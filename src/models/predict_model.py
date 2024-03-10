import torch
import pandas as pd

from sklearn.preprocessing import StandardScaler
from pytorch_lightning import seed_everything

# set seeds for reproducibility
seed_everything(1996, workers=True)

# import model and set it to evaluation mode
model = torch.load('models/full_model_mlp-optimized-trial33-epoch=33-val_loss=0.35-val_r2=0.88.ckpt.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# load and scale dataset to predict
df = pd.read_excel("data/prova.xlsx")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# convert the dataset to a tensor and sennd it to the device
dataset_tensor = torch.tensor(df_scaled, dtype=torch.float32)
dataset_tensor = dataset_tensor.to(device)

# do predictions
with torch.no_grad():
    predictions = model(dataset_tensor)

predictions_df = pd.DataFrame(predictions.cpu().numpy(), columns=['predicted_ET0'])

# export predictions
predictions_df.index = df.index
final_df = pd.concat([df, predictions_df], axis=1) # concantenate predictions with features
final_df.to_excel("models/predictions.xlsx")