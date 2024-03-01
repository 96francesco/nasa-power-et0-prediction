import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import StepLR

class LitModel(pl.LightningModule):
    """
    A PyTorch Lightning Module for the et0 prediction model using NASA POWER data.

    This class encapsulates the model architecture, training step, validation step, and configuration
      of optimizers. It's designed for easy experimentation with different model architectures and 
      training routines using the PyTorch Lightning framework.

    Attributes:
        model (torch.nn.Module): The neural network model to be trained.
        loss_fn (Callable): The loss function used for training the model.
        lr (float): Learning rate for the optimizer.

    Methods:
        forward(x):
            Defines the forward pass of the model.
        
        training_step(batch, batch_idx):
            Conducts a single training step, including forward pass, loss calculation, and logging.
        
        validation_step(batch, batch_idx):
            Conducts a single validation step, including forward pass and loss calculation.
        
        configure_optimizers():
            Configures the model's optimizers and learning rate scheduler.
    """
    def __init__(self, model, lr=0.01):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.test_outputs = []

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)

        # check for NaN values
        if torch.isnan(loss):
            print(f"NaN detected at epoch {self.current_epoch}, batch {batch_idx}")

        self.log('train_loss', loss,
                 prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        self.log('val_loss', loss,
                 prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss # !!!

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        self.test_outputs.append({'y_pred': y_pred.detach(), 'y_true': y.detach()})

        return {'y_pred': y_pred.detach(), 'y_true': y.detach()}

    def on_test_epoch_end(self):
        # Aggregate outputs
        all_outputs = self.all_gather(self.test_outputs)

        # Concatenate all y_pred and y_true from each test step
        y_pred = torch.cat([tmp['y_pred'] for tmp in all_outputs], dim=0)
        y_true = torch.cat([tmp['y_true'] for tmp in all_outputs], dim=0)

        # Convert to numpy arrays for calculation with sklearn
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        # Calculate metrics
        r2 = r2_score(y_true_np, y_pred_np)
        rmse = mean_squared_error(y_true_np, y_pred_np, squared=False)
        nrmse = rmse / np.mean(y_true_np)

        # Log metrics
        self.log('test_r2', r2)
        self.log('test_rmse', rmse)
        self.log('test_nrmse', nrmse)

        # store predictions and actual observations
        self.predictions = y_pred_np
        self.actuals = y_true_np

        # store metrics
        self.metrics = {'R2': r2, 'RMSE': rmse, 'nRMSE': nrmse}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                    self.learning_rate,
                                    weight_decay=0.01)
        return optimizer