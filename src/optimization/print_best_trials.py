import optuna

# database file path
db_file_path = 'sqlite:////home/francesco/nasa_power_et0_prediction/src/optimization/optimization.db'

# load the study
study = optuna.load_study(study_name='rf_optimization', storage=db_file_path)

# retrieve the best trials
pareto_front_trials = study.best_trials

# print the details of the Pareto front trials
for trial in pareto_front_trials:
    print(f'Trial {trial.number}:')
    print(f'  Values: {trial.values}')
    print(f'  Params: {trial.params}')