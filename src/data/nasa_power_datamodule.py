import pandas as pd
import torch
import pytorch_lightning as pl

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
    def __init__(self, train_dir: str, test_dir: str, batch_size=32, train_split=0.7):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.train_split = train_split

    def prepare_data(self):
        # read, process and split training dataset
        df_train = pd.read_excel(self.train_dir)
        X = df_train.drop('ET', axis=1).values
        y = df_train['ET'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X) # scale features
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y,
                                                          test_size = 1 - self.train_split,
                                                          random_state=123)
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)

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

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=15)