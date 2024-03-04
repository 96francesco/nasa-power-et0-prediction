import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pytorch_lightning import seed_everything

# import custom modules
from data.nasa_power_datamodule import NASAPOWERDataModule
from models.litmodel import LitModel
from models.mlp import MLP

# set seeds for reproducibility
seed_everything(1996, workers=True)

# define dataset directory
train_dir = 'data/train_set.xlsx'
test_dir = 'data/test_set.xlsx'

# define model and data module
epochs = 50
data_module = NASAPOWERDataModule(train_dir=train_dir, test_dir=test_dir, batch_size=16)

# load model
filename = 'mlp-optimized-trial13-epoch=37-val_loss=0.35.ckpt'
model = LitModel.load_from_checkpoint(checkpoint_path=f'models/checkpoints/{filename}')
print(model.hparams)


# set the model for evaluation
model.eval()
model.freeze()

# test the model
trainer = pl.Trainer()
trainer.test(model, data_module)

# extract predictions and actual observations
y_test_predictions = model.predictions
y_test_actual = model.actuals

# make plots
plt.figure(figsize=(6, 6))
plt.scatter(model.actuals, model.predictions, alpha=0.5)
plt.xlabel('Predicted ETo (mm/day)', fontsize=12)
plt.ylabel('Observed ETo (mm/day)', fontsize=12)
plt.title('Predicted vs Observed (Testing)', fontsize=16)
plt.plot([min(model.actuals), max(model.actuals)], 
         [min(model.actuals), max(model.actuals)], 
         'r--')

# put metrics in the plot
metrics_text = '\n'.join([f'{key}: {value:.2f}' for key, value in model.metrics.items()])
plt.annotate(metrics_text, xy=(0.05, 0.8), 
             xycoords='axes fraction', 
             bbox=dict(boxstyle="round", 
                       fc="w"))
plt.savefig(f'reports/figures/output/output_{filename}.png')
plt.show()
