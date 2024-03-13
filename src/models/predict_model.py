import torch
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from pytorch_lightning import seed_everything

# define dataset and output directory
dataset_directory = 'data/df_spain'
output_directory = 'models/predictions/predictions_spain'

# set seeds for reproducibility
seed_everything(1996, workers=True)

# import model and set it to evaluation mode
model = torch.load('models/full_model_mlp.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# define scaler
scaler = StandardScaler()

# iterate over the files in the dataset directory
for filename in os.listdir(dataset_directory):
    if filename.endswith(".csv"): 
        file_path = os.path.join(dataset_directory, filename)
        
        # load and process dataset to predict
        df = pd.read_csv(file_path)
        original_columns = df.iloc[:, :7]
        features = df.iloc[:, 7:]
        features_scaled = scaler.fit_transform(features)
        
        # convert the features to a tensor and send it to the device
        features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
        features_tensor = features_tensor.to(device)
        
        # run predictions
        with torch.no_grad():
            predictions = model(features_tensor)
        
        # convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions.cpu().numpy(), columns=['predicted_ET0'])
        
        # concatenate the original columns with predictions
        final_df = pd.concat([original_columns, predictions_df], axis=1)
        
        # save the file
        output_file_path = os.path.join(output_directory, f'predictions_{filename}')
        final_df.to_csv(output_file_path, index=False)

print("Predictions finished.")