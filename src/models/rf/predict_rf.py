import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler

# define dataset and output directory
dataset_directory = 'data/df_italy'
output_directory = 'data/df_italy/predictions'

# make sure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# load the trained RF model
model_filename = 'models/rf_model.joblib'
model = joblib.load(model_filename)

# define scaler
scaler = StandardScaler()

# iterate over the files in the dataset directory
for filename in os.listdir(dataset_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(dataset_directory, filename)
        
        # load and process dataset to predict
        df = pd.read_csv(file_path)
        original_columns = df.iloc[:, :7]  # adjust the number as per your requirement
        features = df.iloc[:, 7:]  # adjust the number as per your requirement
        features_scaled = scaler.fit_transform(features)

        # make predictions
        predictions = model.predict(features_scaled)
        
        # convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions, columns=['predicted_ET0'])
        
        # concatenate the original columns with predictions
        final_df = pd.concat([original_columns, predictions_df], axis=1)
        
        # save the file
        output_file_path = os.path.join(output_directory, f'predictions_{filename}')
        final_df.to_csv(output_file_path, index=False)

print("Predictions finished.")
