import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# set seeds for repdocibility
seed = 1996

# load data
df = pd.read_excel('data/train_set.xlsx')
X = df.drop(columns=['ET'])
y = df['ET']

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# set-up cross validation
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# scores lists
rmse_scores = []
nrmse_scores = []
r2_scores = []

# define model
model = RandomForestRegressor(
      criterion='squared_error',
      n_estimators=160,  
      max_depth=32,
      min_samples_split=15,
      min_samples_leaf=5,
      random_state=seed
)

# perform K-fold cross validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    # calculate RMSE
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    rmse_scores.append(rmse)

    # calculate nRMSE
    nrmse = rmse / (y_val.max() - y_val.min())
    nrmse_scores.append(nrmse)

    # calculate R2 Score
    r2 = r2_score(y_val, y_pred)
    r2_scores.append(r2)

print("RMSE scores:", rmse_scores)
print("nRMSE scores:", nrmse_scores)
print("R2 scores:", r2_scores)