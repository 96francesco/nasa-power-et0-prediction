import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# import custom modules
from data.nasa_power_datamodule import NASAPOWERDataModule
from models.mlp import MLP
from models.litmodel import LitModel

# set seeds for reproducibility
seed_everything(1996, workers=True)

# check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")


# define dataset directory
train_dir = 'data/train_set.xlsx'
test_dir = 'data/test_set.xlsx'

# define model and data module
epochs = 50
data_module = NASAPOWERDataModule(train_dir=train_dir, test_dir=test_dir, batch_size=32)
mlp_instance = MLP(input_size=11, dropout_rate=0.0017917784324016656)
model = LitModel(mlp_instance,
                 lr=0.0061020681060527885,
                 optimizer='adam',
                 weight_decay=1.3455019173512806e-07)

filename = "mlp-optimized-trial13-{epoch:02d}-{val_loss:.2f}-{val_r2:.2f}"

# define callabacks
early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=0.00,
                                    patience=20,
                                    verbose=True,
                                    mode="min")

checkpoint_callback = ModelCheckpoint(
    dirpath="models/checkpoints",
    filename=filename,
    save_top_k=3,
    verbose=False,
    monitor='val_loss',
    mode='min')

# define trainer and start training
trainer = pl.Trainer(max_epochs=epochs,
                     log_every_n_steps=10,
                     enable_checkpointing=True,
                     accelerator='gpu',
                     devices=1,
                     detect_anomaly=False,
                     callbacks=[early_stop_callback, checkpoint_callback])

trainer.fit(model, data_module)