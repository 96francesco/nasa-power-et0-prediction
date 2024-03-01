import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import StepLR

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

num_epochs = 100
data_module = NASAPOWERDataModule(train_dir=train_dir, test_dir=test_dir)
mlp_instance = MLP(input_size=11)


model = LitModel(mlp_instance,
                 lr=0.012022644346174132)
early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=0.00,
                                    patience=10,
                                    verbose=True,
                                    mode="min")
trainer = pl.Trainer(max_epochs=num_epochs,
                     log_every_n_steps=10,
                     detect_anomaly=False,
                     callbacks=[early_stop_callback])

trainer.fit(model, data_module)