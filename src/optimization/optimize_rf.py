# import libraries
import optuna
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# load training data
df = pd.read_excel('data/train_set.xlsx')

# split features and target
X = df.drop(columns=['ET'])
y = df['ET']

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def objective(trial):
      # get validation set from training data
      X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, 
                                                            test_size=0.3, 
                                                            random_state=96)
      
      # parameters to optimize
      n_estimators = trial.suggest_int('n_estimators', 10, 200, step=5)
      max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
      min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
      min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 50)
    
      try:
            # initialize and fit RF model
            model = RandomForestRegressor(
                  criterion='squared_error',
                  n_estimators=n_estimators,
                  max_depth=max_depth,
                  min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf,
                  random_state=96
            )
            model.fit(X_train, y_train)
            
            # make predictions on the validation set 
            y_pred = model.predict(X_val)
            
            # calculate MSE and R2 on the validation set
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)

      except Exception as e:
            print(f"Trial {trial.number} failed with exception {e}")
            return float('inf'), 0
      
      return mse, r2

# setup the study
storage = optuna.storages.RDBStorage(url='sqlite:///src/optimization/optimization.db')
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
study = optuna.create_study(directions=['minimize', 'maximize'],
                            sampler=TPESampler(seed=96),
                            pruner=pruner,
                            storage=storage,
                            study_name='rf_optimization',
                            load_if_exists=True)

# start study
study.optimize(objective, n_trials=50, gc_after_trial=True)