import torch
from torch.utils.data import Dataset
import ipdb
import numpy as np
import pandas as pd
from einops import rearrange
from sklearn.model_selection import train_test_split
import os
import wfdb
import h5py
import neurokit2 as nk
from scipy.io import loadmat
from melp.paths import SPLIT_DIR
from einops import rearrange

'''
In this code:
PTB-XL has four subset: superclass, subclass, form, rhythm
ICBEB is CPSC2018 dataset mentioned in the original paper
Chapman is the CSN dataset from the original paper
'''

class ECGDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 csv_file: pd.DataFrame, 
                 split: str,
                 dataset_name: str = 'ptbxl', 
                 data_pct: float = 1,
                 ):
        """
        Args:
            data_path (string): Path to store raw data.
            csv_file (string): Path to the .csv file with labels and data path.
            mode (string): ptbxl/icbeb/chapman.
            data_pct (float): Percentage of data to use.
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.split = split
        csv_file = csv_file.sample(frac=data_pct, random_state=42).reset_index(drop=True)
        
        if 'ptbxl' in dataset_name:
            self.labels_name = list(csv_file.columns[6:])
            self.num_classes = len(self.labels_name)

            self.ecg_path = csv_file["filename_hr"]
            # self.ecg_path = csv_file['ecg_id'].apply(lambda x: f"{x}.npy")
            # in ptbxl, the column 0-5 is other meta data, the column 6-end is the label
            self.labels = csv_file.iloc[:, 6:].values
            self.patient_id = csv_file['patient_id']
            self.ecg_id = csv_file['ecg_id']

        elif self.dataset_name == 'icbeb':
            self.labels_name = list(csv_file.columns[7:])
            self.num_classes = len(self.labels_name)

            self.ecg_path = csv_file['filename']
            # in icbeb, the column 0-6 is other meta data, the column 7-end is the label
            self.labels = csv_file.iloc[:, 7:].values
            self.patient_id = csv_file['patient_id']
            self.ecg_id = csv_file['ecg_id']

        elif self.dataset_name == 'chapman': 
            self.labels_name = list(csv_file.columns[3:])
            self.num_classes = len(self.labels_name)

            csv_file['ecg_path'] = csv_file['ecg_path'].apply(lambda x: x.replace('/chapman/', ''))
            # csv_file = csv_file[csv_file["ecg_path"].apply(lambda x: os.path.exists(os.path.join(data_path, x))).values]
            csv_file.reset_index(drop=True, inplace=True)
            self.ecg_path = csv_file['ecg_path']
            self.labels = csv_file.iloc[:, 3:].values
            self.ecg_id = csv_file["ecg_path"].apply(lambda x: x.split('/')[-1].split('.')[0])
            self.patient_id = csv_file["ecg_path"].apply(lambda x: x.split('/')[-2])

        elif self.dataset_name == "code":
            # self.label_name = list(csv_file.columns)
            # "AFIB", "VPC", "NORM", "1AVB", "CRBBB", "STE", "PAC", "CLBBB", "STD"
            self.labels_name = ["1AVB", "RBBB", "LBBB", "SB", "AFIB", "ST"]
            self.num_classes = len(self.labels_name)

            with h5py.File(os.path.join(self.data_path, "ecg_tracings.hdf5"), "r") as f:
                self.x = np.array(f['tracings'])
            
            self.labels = csv_file.values
        elif self.dataset_name == 'anzhen':
            # 假设最后20列是one-hot标签，前面有ecg_id, patient_id, record_name等元数据
            self.labels_name = list(csv_file.columns[-18:])  # 最后19列是标签
            self.num_classes = len(self.labels_name)
            self.labels = csv_file.iloc[:, -18:].values  # 最后19列作为标签

            # patient_id默认为record_name
            self.patient_id = csv_file['record_name']
            self.ecg_id = csv_file['record_name']
            self.ecg_path = csv_file['record_name'].apply(lambda x: 'wfdb_output/' + x)
        else:
            raise ValueError("dataset_type should be either 'ptbxl' or 'icbeb' or 'chapman")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        if 'ptbxl' in self.dataset_name:
            ecg_path = os.path.join(self.data_path, self.ecg_path.iloc[idx])
            # the wfdb format file include ecg and other meta data
            # the first element is the ecg data

            ecg = wfdb.rdsamp(ecg_path)[0]
            ecg = ecg.T
            ecg = ecg[:, :5000]
            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            ecg_id = self.ecg_id[idx]
            patient_id = self.patient_id[idx]
            uid = f"{patient_id}_{ecg_id}"
        
        elif self.dataset_name == 'anzhen':
            ecg_path = os.path.join(self.data_path, self.ecg_path.iloc[idx])

            ecg = wfdb.rdsamp(ecg_path)[0]
            ecg = ecg.T
            ecg = ecg[:, :2500]

            # padding to 5000 to match the pre-trained data length
            ecg = np.pad(ecg, ((0, 0), (0, 2500)), 'constant', constant_values=0)
            ecg = ecg[:, :5000]
            
            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)

            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            ecg_id = self.ecg_id[idx]
            patient_id = self.patient_id[idx]
            uid = f"{patient_id}_{ecg_id}"

        elif self.dataset_name == 'icbeb':
            ecg_path = os.path.join(self.data_path, self.ecg_path.iloc[idx])
            # icbeb has dat file, which is the raw ecg data
            # ecg = np.load(ecg_path)
            ecg = wfdb.rdsamp(ecg_path)[0]
            ecg = ecg.T
            # icbeb has different length of ecg, so we need to preprocess it to the same length
            # we only keep the first 2500 points as METS did
            ecg = ecg[:, :2500]
            
            # padding to 5000 to match the pre-trained data length
            ecg = np.pad(ecg, ((0, 0), (0, 2500)), 'constant', constant_values=0)
            ecg = ecg[:, :5000]

            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
            
            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            ecg_id = self.ecg_id[idx]
            patient_id = self.patient_id[idx]
            uid = f"{patient_id}_{ecg_id}"
            
        elif self.dataset_name == 'chapman':
            # chapman ecg_path has / at the start, so we need to remove it
            ecg_path = os.path.join(self.data_path, self.ecg_path.iloc[idx])
            # raw data is (12, 5000), do not need to transform
            ecg = loadmat(ecg_path)['val']
            ecg = ecg.astype(np.float32)
            # ecg = np.load(ecg_path)
            ecg = ecg[:, :5000]
            
            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
            
            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            ecg_id = self.ecg_id[idx]
            patient_id = self.patient_id[idx]
            uid = f"{patient_id}_{ecg_id}"
        
        elif self.dataset_name == 'code':
            ecg = self.x[idx]
            ecg = ecg.T
            new_ecg = []
            for i in range(ecg.shape[0]):
                new_ecg.append(nk.signal_resample(ecg[i], desired_length=5000, sampling_rate=400, desired_sampling_rate=500))
            new_ecg = np.array(new_ecg)
            ecg = new_ecg

            # normalzie to 0-1
            ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
            
            ecg = torch.from_numpy(ecg).float()
            target = self.labels[idx]
            target = torch.from_numpy(target).float()
            ecg_id = idx
            patient_id = idx
            uid = f"{patient_id}_{ecg_id}"

        # switch AVL and AVF
        # In MIMIC-ECG, the lead order is I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6
        # In downstream datasets, the lead order is I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        if self.dataset_name == "code":
            # DI, DII, DIII, AVL, AVF, AVR, V1, V2, V3, V4, V5, V6
            # => I, II, III, aVR, aVF, aVL, V1, V2, V3, V4, V5, V6
            ecg[[3, 5]] = ecg[[5, 3]]
        else:
            ecg[[4, 5]] = ecg[[5, 4]]  
        num_leads = ecg.size(0)

        return {
            "id": uid,
            "patient_id": patient_id,
            "ecg": ecg,
            "label": target
        }


if __name__ == "__main__":
    from melp.paths import RAW_DATA_PATH, SPLIT_DIR
    dataset = ECGDataset(data_path=str(RAW_DATA_PATH / "code"),
                         csv_file=pd.read_csv(SPLIT_DIR / 'code/code_test.csv'), 
                         split="test",
                         dataset_name='code',
                         )
    print(len(dataset))
    sample = dataset[0]
    # print(sample['ecg_patch'].size()) 
    print(sample['ecg'].size())
    print(sample['label'].size())

    # import wfdb
    # wave, meta = wfdb.rdsamp('/home/*/Documents/MM-ECG-FM/data/raw/mimic-iv-ecg/files/p1001/p10010058/s41184297/41184297')
    # ipdb.set_trace()