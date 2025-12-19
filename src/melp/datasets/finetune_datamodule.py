import ipdb
import os
import pandas as pd
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from melp.datasets.finetune_dataset import ECGDataset
from melp.paths import SPLIT_DIR, RAW_DATA_PATH, PROCESSED_DATA_PATH


class ECGDataModule(LightningDataModule):
    def __init__(self, dataset_dir: str, dataset_name: str,
                 batch_size: int, num_workers: int,
                 train_data_pct: float):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data_pct = train_data_pct
        assert dataset_name in ["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm", 
                                "icbeb", "chapman", "code", "anzhen"], f"Invalid dataset name {self.dataset_name} found"
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        if "ptbxl" in dataset_name:
            task_name = dataset_name.replace('ptbxl_', '')
            self.dataset_dir = os.path.join(self.dataset_dir, "ptbxl")
            self.split_dir = SPLIT_DIR / "ptbxl" / task_name
        else:
            self.dataset_dir = os.path.join(self.dataset_dir, f"{dataset_name}")
            self.split_dir = SPLIT_DIR / dataset_name

    def train_dataloader(self):
        train_dataset = ECGDataset(
            data_path=self.dataset_dir,
            csv_file=pd.read_csv(self.split_dir / f"{self.dataset_name}_train.csv"),
            split="train",
            dataset_name=self.dataset_name,
            data_pct=self.train_data_pct
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False
        )

        return train_dataloader

    def val_dataloader(self):

        val_dataset = ECGDataset(
            data_path=self.dataset_dir,
            csv_file=pd.read_csv(self.split_dir / f"{self.dataset_name}_val.csv"),
            split="val",
            dataset_name=self.dataset_name,
            data_pct=1
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False
        )

        return val_dataloader
    
    def test_dataloader(self):
        
        test_dataset = ECGDataset(
            data_path=self.dataset_dir,
            csv_file=pd.read_csv(self.split_dir / f"{self.dataset_name}_test.csv"),
            split="test",
            dataset_name=self.dataset_name,
            data_pct=1
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            drop_last=False
        )

        return test_dataloader


if __name__ == "__main__":
    dm = ECGDataModule(
        dataset_dir=RAW_DATA_PATH,
        dataset_name="ptbxl_super_class",
        batch_size=4,
        num_workers=1,
        train_data_pct=0.1,
        use_ecg_patch=True
    )
    for batch in dm.train_dataloader():
        break
    print(batch["ecg"].shape)
    print(batch["ecg_patch"].shape)
    print(batch["t_indices"].shape)
    ipdb.set_trace()