import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import StepLR

from models.mlp import MLP

class LitModel(pl.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating models using NASA POWER data.
    
    This class encapsulates the model architecture, data handling, training loop, validation
    loop, and testing loop within the PyTorch Lightning framework. It supports dynamic 
    selection of optimizers, learning rate schedulers, and the ability to load the model 
    from a checkpoint for continued training or inference.

    Attributes:
        model (torch.nn.Module): The neural network model.
        lr (float): Learning rate for the optimizer.
        optimizer_name (str): Name of the optimizer to use ('adam', 'sgd', etc.).
        weight_decay (float): Weight decay (L2 penalty) for the optimizer.
        checkpoint_path (str, optional): Path to the checkpoint file from which model weights will be loaded.

    Methods:
        forward(x): Implements the forward pass of the model.
        training_step(batch, batch_idx): Processes a single batch during training.
        validation_step(batch, batch_idx): Processes a single batch during validation.
        test_step(batch, batch_idx): Processes a single batch during testing.
        configure_optimizers(): Sets up the optimizer and learning rate scheduler.
    """
    def __init__(self, model=None, lr=0.01, optimizer='sgd', weight_decay=1e-5):
        super().__init__()
        self.model = model
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.test_outputs = []
        self.save_hyperparameters()

        # load from checkpoint if path is provided
        if model is None:
            # initalize model
            self.model = MLP(input_size=11)
            pass

        if optimizer == 'sgd':
                self.optimizer = torch.optim.SGD
        elif optimizer == 'adam':
                self.optimizer = torch.optim.Adam
        else:
            raise ValueError(f'Unkwnown optimizer') 

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
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # collect batch's outputs for use in on_validation_epoch_end
        if not hasattr(self, 'validation_outputs'):
            self.validation_outputs = []
        self.validation_outputs.append({'preds': y_pred.detach(), 'targets': y.detach()})
        
        return loss

       
    def on_validation_epoch_end(self):
        # access the validation step outputs stored in self during validation_step
        preds = torch.cat([x['preds'] for x in self.validation_outputs], dim=0)
        targets = torch.cat([x['targets'] for x in self.validation_outputs], dim=0)

        # calculate the R2 value over all validation outputs
        r2 = r2_score(targets.cpu().numpy(), preds.cpu().numpy())
        self.log('val_r2', r2, prog_bar=True, logger=True)

        self.validation_outputs = []


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.MSELoss()(y_pred, y)
        self.test_outputs.append({'y_pred': y_pred.detach(), 'y_true': y.detach()})

        return {'y_pred': y_pred.detach(), 'y_true': y.detach()}

    def on_test_epoch_end(self):
        # aggregate outputs
        all_outputs = self.all_gather(self.test_outputs)

        # concatenate all y_pred and y_true from each test step
        y_pred = torch.cat([tmp['y_pred'] for tmp in all_outputs], dim=0)
        y_true = torch.cat([tmp['y_true'] for tmp in all_outputs], dim=0)

        # Convert to numpy arrays for calculation with sklearn
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        # Calculate metrics
        r2 = r2_score(y_true_np, y_pred_np)
        rmse = mean_squared_error(y_true_np, y_pred_np, squared=False)
        nrmse = rmse / np.mean(y_true_np)

        # log metrics
        self.log('test_r2', r2)
        self.log('test_rmse', rmse)
        self.log('test_nrmse', nrmse)

        # store predictions and actual observations
        self.predictions = y_pred_np
        self.actuals = y_true_np

        # store metrics
        self.metrics = {'R2': r2, 'RMSE': rmse, 'nRMSE': nrmse}

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]