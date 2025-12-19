import ipdb
import numpy as np
import pandas as pd
import wfdb
import os
import neurokit2 as nk2
from glob import glob
from sklearn.model_selection import train_test_split
from pprint import pprint
from tqdm import tqdm
import multiprocessing as mp
import sys
from melp.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH
from melp.paths import ROOT_PATH as REPO_ROOT_DIR

'''
python preprocess_mimic_iv_ecg.py
'''

def process_report(row):
    ''' Preprocess the report '''
    # Select the relevant columns and filter out NaNs
    report = row[['report_0', 'report_1', 'report_2', 'report_3', 'report_4', 
                  'report_5', 'report_6', 'report_7', 'report_8', 'report_9', 
                  'report_10', 'report_11', 'report_12', 'report_13', 'report_14', 
                  'report_15', 'report_16', 'report_17']].dropna()
    # Concatenate the report
    report = '. '.join(report)
    # Replace and preprocess text
    report = report.replace('EKG', 'ECG').replace('ekg', 'ecg')
    report = report.strip(' ***').strip('*** ').strip('***').strip('=-').strip('=')
    # Convert to lowercase
    report = report.lower()

    # concatenate the report if the report length is not 0
    total_report = ''
    if len(report.split()) != 0:
        total_report = report
        total_report = total_report.replace('\n', ' ')
        total_report = total_report.replace('\r', ' ')
        total_report = total_report.replace('\t', ' ')
        total_report += '.'
    if len(report.split()) == 0:
        total_report = 'empty'
    # Calculate the length of the report in words
    return len(report.split()), total_report


# Function to process each record
def process_record(p, uid, meta_path, save_path):
    ecg_path = os.path.join(meta_path, p)
    record = wfdb.rdsamp(ecg_path)[0]
    record = record.T
    
    if len(np.unique(record)) == 1 and np.unique(record)[0] == 0:
        return

    # Check if the record contains NaN or Inf
    if np.isnan(record).sum() == 0 and np.isinf(record).sum() == 0:
        denoised_record = np.zeros_like(record)
        for i in range(record.shape[0]):
            # denoise each lead using neurokit2
            denoised_record[i] = nk2.ecg_clean(record[i], sampling_rate=500, method="emrich2023")
        
        assert np.isnan(denoised_record).sum() == 0 and np.isinf(denoised_record).sum() == 0, f"Found NaN or Inf in the denoised record: {uid}"

        # store the data
        np.save(save_path + f"/{uid}.npy", denoised_record[:, :5000])
    else:
        # Do not process the record if it contains NaN or Inf
        return 
    
    # if np.isnan(record).sum() == 0 and np.isinf(record).sum() == 0:
    #     # min max normalization
    #     record = (record - record.min()) / (record.max() - record.min())
    #     record *= 1000
    #     record = record.astype(np.int16)
    # else:
    #     if np.isinf(record).sum() != 0:
    #         for i in range(record.shape[0]):
    #             inf_idx = np.where(np.isinf(record[i]))[0]
    #             for idx in inf_idx:
    #                 neighbor_record = record[i, max(0, idx-6):min(idx+6, record.shape[1])]
    #                 if len(neighbor_record[~np.isinf(neighbor_record) & ]) != 0:
    #                     record[i, idx] = np.nanmean(neighbor_record[~np.isinf(neighbor_record)])

    #     if np.isnan(record).sum() != 0:
    #         # i is the index of lead ...
    #         for i in range(record.shape[0]):
    #             nan_idx = np.where(np.isnan(record[i]))[0]
    #             for idx in nan_idx:
    #                 neighbor_record = record[i, max(0, idx-6):min(idx+6, record.shape[1])]
    #                 if len(neighbor_record[~np.isnan(neighbor_record)]) != 0:
    #                     record[i, idx] = np.nanmean(neighbor_record[~np.isnan(neighbor_record)])
    #                 else:
    #                     # If there is no valid value in the neighbor, we use the mean of the whole record
    #                     record[i, idx] = np.nanmean(record[i])

    #     record = (record - record.min()) / (record.max() - record.min())
    #     record *= 1000
    #     record = record.astype(np.int16)


def process_record_wrapper(args):
    return process_record(*args)


def main():
    # Step 1: preprocess reports
    print("Step 1: Preprocessing reports...")
    preprocessed_report_path = PROCESSED_DATA_PATH / "mimic-iv-ecg/preprocessed_reports.csv"
    if not os.path.exists(preprocessed_report_path):
        report_csv = pd.read_csv(RAW_DATA_PATH / 'mimic-iv-ecg/machine_measurements.csv', low_memory=False)
        tqdm.pandas()
        report_csv['report_length'], report_csv['total_report'] = zip(*report_csv.progress_apply(process_report, axis=1))
        # Filter out reports with less than 4 words
        report_csv = report_csv[report_csv['report_length'] >= 4]
        # you should get 771693 here
        print(report_csv.shape)
        report_csv.reset_index(drop=True, inplace=True)
        report_csv.to_csv(preprocessed_report_path, index=False)
    else:
        print("Preprocessed reports already exist. Don't need to preprocess again.")
        report_csv = pd.read_csv(preprocessed_report_path, low_memory=False)

    # Step 2: preprocess records
    print("Step 2: preprocess ECG and save them into numpy array...")
    record_csv = pd.read_csv(RAW_DATA_PATH / 'mimic-iv-ecg/record_list.csv', low_memory=False)
    record_csv["id"] = record_csv["subject_id"].astype(str) + "_" + record_csv["study_id"].astype(str)
    report_csv["id"] = report_csv["subject_id"].astype(str) + "_" + report_csv["study_id"].astype(str)
    # only keep records which have paired reports.
    record_csv = record_csv[record_csv["id"].isin(report_csv["id"])]
    record_csv.reset_index(drop=True, inplace=True)

    save_dir = PROCESSED_DATA_PATH / "mimic-iv-ecg/records"
    os.makedirs(save_dir, exist_ok=True)

    for _, row in tqdm(record_csv.iterrows(), total=len(record_csv)):
        process_record(row["path"], row["id"], str(RAW_DATA_PATH / 'mimic-iv-ecg'), save_dir)

    # Create a pool of workers
    pool = mp.Pool(processes=32)
    meta_path = str(RAW_DATA_PATH / 'mimic-iv-ecg')
    # Process records in parallel
    results = list(tqdm(pool.imap(process_record_wrapper, 
                [(row["path"], row["id"], meta_path, save_dir) for _, row in record_csv.iterrows()]), 
            total=len(record_csv)))
    # Close the pool
    pool.close()
    pool.join()

    print("Step 3: Preprocessing records...")
    saved_npys = glob(str(save_dir + "/*.npy"))
    report_csv = report_csv[report_csv["id"].isin([os.path.basename(f).split(".")[0] for f in saved_npys])]
    report_csv.reset_index(drop=True, inplace=True)

    record_csv = record_csv[["id", "path"]]
    report_csv = report_csv.merge(record_csv, on="id", how="inner")
    
    # split csv to train and val
    split_dir = str(REPO_ROOT_DIR) + "/src/melp/data_split/mimic-iv-ecg"
    os.makedirs(split_dir, exist_ok=True)
    unique_subject_ids = report_csv["subject_id"].unique()
    print(len(unique_subject_ids))
    # split should be done based on subject_id instead of rows ...
    train_subject_ids, val_subject_ids = train_test_split(unique_subject_ids, test_size=0.02, random_state=42)
    train_csv = report_csv[report_csv["subject_id"].isin(train_subject_ids)]
    val_csv = report_csv[report_csv["subject_id"].isin(val_subject_ids)]
    print(f"Train size: {train_csv.shape[0]}, Val size: {val_csv.shape[0]}")
    train_csv.reset_index(drop=True, inplace=True)
    val_csv.reset_index(drop=True, inplace=True)
    train_csv.to_csv(split_dir + "/train.csv", index=False)
    val_csv.to_csv(split_dir + "/val.csv", index=False)


if __name__ == "__main__":
    main()