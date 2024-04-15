import pandas as pd
import torch
import pytorch_lightning as pl

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NASAPOWERDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning Data Module for handling NASA POWER data for ET0 prediction.
    
    This class is responsible for preparing and loading the dataset used in the et0 prediction model.
    It includes methods for data preparation, setup, and creating PyTorch DataLoaders for the training, v
    alidation, and test datasets.
    
    Attributes:
        train_dir (str): The file path to the training dataset.
        test_dir (str): The file path to the testing dataset.
        batch_size (int): The size of the batches for the DataLoader.
        train_split (float): The proportion of the data to be used for training.
    
    Methods:
        prepare_data():
            Reads and preprocesses the data from the file paths specified by train_dir and test_dir.
        
        setup(stage=None):
            Prepares the data for training, validation, and testing. This method is responsible for 
            splitting the data and applying any transformations.
        
        train_dataloader():
            Returns a DataLoader for the training dataset.
        
        val_dataloader():
            Returns a DataLoader for the validation dataset.
        
        test_dataloader():
            Returns a DataLoader for the test dataset.
    """
    def __init__(self, train_dir: str, test_dir: str, batch_size=32, train_split=0.7,
                 train_idx=None, val_idx=None):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.train_split = train_split
        self.train_idx = train_idx
        self.val_idx = val_idx

    def prepare_data(self):
        # read, process and split training dataset
        df_train = pd.read_excel(self.train_dir)
        X = df_train.drop('ET', axis=1).values
        y = df_train['ET'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # split data according to train_split or provided indices
        if self.train_idx is not None and self.val_idx is not None:
            self.X_train, self.y_train = X_scaled[self.train_idx], y[self.train_idx]
            self.X_val, self.y_val = X_scaled[self.val_idx], y[self.val_idx]
        else:
            # if indices are not provided, use train_split to split the data
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_scaled, y, 
                                                              test_size=1 - self.train_split, 
                                                              random_state=123)
        
        # convert NumPy arrays to tensors and fix dimensions
        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(-1)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32).unsqueeze(-1)

        # read and process testing dataset
        df_test = pd.read_excel(self.test_dir)
        X_test = df_test.drop('ET', axis=1).values
        y_test = df_test['ET'].values
        X_test_scaled = scaler.transform(X_test)
        self.X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TensorDataset(self.X_train, self.y_train)
            self.val_dataset = TensorDataset(self.X_val, self.y_val)
        if stage == 'test' or stage is None:
            self.test_dataset = TensorDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=15)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset, batch_size=self.batch_size,
    #                       num_workers=15)
    def val_dataloader(self):
        if hasattr(self, 'val_dataset') and self.val_dataset is not None:
            return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15)
        else:
            # Return an empty DataLoader
            return None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=15)