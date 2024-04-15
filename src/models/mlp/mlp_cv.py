import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# import custom modules
from models.mlp.mlp import MLP
from models.mlp.litmodel import LitModel
from data.nasa_power_datamodule import NASAPOWERDataModule

# set seeds for reproducibility
seed_everything(1996, workers=True)

# check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

# set parameters
epochs = 50
splits = 10
kf = KFold(n_splits=splits, shuffle=True, random_state=1996)

# initialize lists to store metrics
rmse_scores = []
nrmse_scores = []
r2_scores = []

# define dataset directory
train_dir = 'data/train_set.xlsx'
test_dir = 'data/test_set.xlsx'

# get number of samples
df_train = pd.read_excel(train_dir)
num_samples = len(df_train)

# perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(num_samples))):
      print(f"Training fold {fold+1}/{splits}")
      
      # initialize data module
      data_module = NASAPOWERDataModule(train_dir=train_dir, 
                                    test_dir=test_dir,
                                    batch_size=8, 
                                    train_idx=train_idx,
                                    val_idx=val_idx)
      data_module.prepare_data()
      data_module.setup(stage='fit')
      
      # initialize model
      mlp_instance = MLP(input_size=11, dropout_rate=0.026226075828239083)
      model = LitModel(mlp_instance, 
                        lr=0.002334575812078728, 
                        optimizer='adam', 
                        weight_decay=3.601652384566776e-08)
      
      # set early stopping
      early_stop_callback = EarlyStopping(monitor="val_loss", 
                                          min_delta=0.00, 
                                          patience=20, 
                                          verbose=True, 
                                          mode="min")
      
      # initialize trainer and start training
      trainer = Trainer(max_epochs=epochs, 
                        log_every_n_steps=10, 
                        accelerator=device.type, 
                        devices=1 if device.type == 'cuda' else 0, 
                        callbacks=[early_stop_callback],
                        num_sanity_val_steps=0)
      
      trainer.fit(model, data_module)

      # retrieve metrics
      metrics = trainer.callback_metrics
      rmse = torch.sqrt(metrics['val_loss']).item() 
      r2 = trainer.callback_metrics.get('val_r2_epoch', 0) 
      print(f"Fold {fold+1} R2: {r2}")


      # calculate nRMSE
      y_val = data_module.y_val.numpy()  
      nrmse = rmse / (np.max(y_val) - np.min(y_val))
      
      # store metrics
      rmse_scores.append(rmse)
      nrmse_scores.append(nrmse)
      r2_scores.append(r2)

# Print all metrics after cross-validation
print("Cross-Validation Metrics:")
print("RMSE scores:", rmse_scores)
print("nRMSE scores:", nrmse_scores)
print("R2 scores:", r2_scores)