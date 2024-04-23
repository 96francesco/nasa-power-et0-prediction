import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# set seeds for repdocibility
seed = 1996

# load data
train_df = pd.read_excel('data/train_set.xlsx')
test_df = pd.read_excel('data/test_set.xlsx')

# prepare features and target
X_train = train_df.drop(columns=['ET'])
y_train = train_df['ET']
X_test = test_df.drop(columns=['ET'])
y_test = test_df['ET']

# standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# define model with best hparams configuration identified
model = SVR(
    C=70.50630937484068,
    kernel='rbf', 
    gamma='scale',
    epsilon=0.3326181470372891 
)

# fit model
model.fit(X_train_scaled, y_train)

# calculate metrics on the training set
y_pred_train = model.predict(X_train_scaled)
train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
train_r2 = r2_score(y_train, y_pred_train)
train_nrmse = train_rmse / (y_train.max() - y_train.min())
print(f"Training RMSE: {train_rmse}")
print(f"Training nRMSE: {train_nrmse}")
print(f"Training R2: {train_r2}")

# evaluate the model on the test set
y_pred_test = model.predict(X_test_scaled)
test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
test_nrmse = test_rmse / (y_test.max() - y_test.min())
test_r2 = r2_score(y_test, y_pred_test)
print(f"Test RMSE: {test_rmse}")
print(f"Training nRMSE: {test_nrmse}")
print(f"Test R2: {test_r2}")

# plot for the training set
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_train, y_train, alpha=0.5)
plt.xlabel('Predicted ETo (mm/day)', fontsize=12)
plt.ylabel('Observed ETo (mm/day)', fontsize=12)
plt.title('Predicted vs Observed (Training)', fontsize=16)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
metrics_text_training = f'R2: {train_r2:.2f}\nRMSE: {train_rmse:.2f}\nnRMSE: {train_nrmse:.2f}'
plt.annotate(metrics_text_training, xy=(0.05, 0.8), xycoords='axes fraction',
              bbox=dict(boxstyle="round", fc="w"))
plt.savefig('reports/figures/output/svm_model_training.png')
plt.show()

# plot for the test set
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_test, y_test, alpha=0.5)
plt.xlabel('Predicted ETo (mm/day)', fontsize=12)
plt.ylabel('Observed ETo (mm/day)', fontsize=12)
plt.title('Predicted vs Observed (Testing)', fontsize=16)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
metrics_text_testing = f'R2: {test_r2:.2f}\nRMSE: {test_rmse:.2f}\nnRMSE: {test_nrmse:.2f}'
plt.annotate(metrics_text_testing, xy=(0.05, 0.8), xycoords='axes fraction', 
             bbox=dict(boxstyle="round", fc="w"))
plt.savefig('reports/figures/output/svm_model_testing.png')
plt.show()