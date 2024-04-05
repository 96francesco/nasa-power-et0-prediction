# import libraries
import pytorch_lightning as pl
import optuna
import torch

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from models.litmodel import LitModel
from models.mlp import MLP
from data.nasa_power_datamodule import NASAPOWERDataModule


def objective(trial):
      # parameters to optimize
      lr = trial.suggest_float('lr', 1e-5, 1e-1)
      batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
      optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
      weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3)
      dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

      # print trial information
      print(f"Starting trial {trial.number}")
      print(f"Hyperparameters: lr={lr}, batch_size={batch_size}, optimizer={optimizer}, weight_decay={weight_decay}, dropout_rate={dropout_rate}")

      # initialize model and data model
      mlp_instance = MLP(input_size=11, dropout_rate=dropout_rate)
      model = LitModel(mlp_instance, lr=lr, optimizer=optimizer, weight_decay=weight_decay)

      train_dir = 'data/train_set.xlsx'
      test_dir = 'data/test_set.xlsx'
      data_module = NASAPOWERDataModule(train_dir=train_dir, test_dir=test_dir,
                                        batch_size=batch_size)

      # initialize PL trainer
      trainer = pl.Trainer(max_epochs=20,
                           logger=False,
                           enable_progress_bar=True,
                           enable_checkpointing=False,
                           accelerator='gpu',
                           devices=1)
      
      # fit the model trying to catch any exceptions
      try:
            trainer.fit(model, data_module)
      except Exception as e:
            print(f"Trial {trial.number} failed with exception {e}")
            return float('inf'), 0  # return a large value to indicate failure

      # return validation loss and R2
      val_loss = trainer.callback_metrics['val_loss'].item()
      val_r2 = trainer.callback_metrics['val_r2'].item()

      # Print trial results
      print(f"Trial {trial.number} completed: val_loss={val_loss}, val_r2={val_r2}")

      return val_loss, val_r2

# setup the study
storage = optuna.storages.RDBStorage(url='sqlite:///src/optimization/optimization.db')
pruner = MedianPruner(n_startup_trials=5,
                      n_warmup_steps=10,
                      interval_steps=1)
study = optuna.create_study(directions=['minimize', 'maximize'],
                            sampler=TPESampler(seed=42),
                            pruner=pruner,
                            storage=storage,
                            study_name='mlp_optimization',
                            load_if_exists=True)

# start study
study.optimize(objective, n_trials=50, gc_after_trial=True)






