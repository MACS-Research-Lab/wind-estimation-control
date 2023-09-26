import optuna
import numpy as np
import multiprocessing

train_dataloader = None # load the dataloader and let each process share it
validation_dataloader = None

def train_and_evaluate(hyperparams):
    lstm = train(hyperparams)

    save_model(lstm)

    # calculate validation loss
    validation_error = None
    return validation_error

def train(hyperparams):
    lstm = None # make sure to move it to the right device here so it can train on the GPU

def save_model(model):
    # save each model after training for simplicity
    raise NotImplementedError

def objective(trial):
    hyperparams = {
        'batch': trial.suggest_categorical('batch', [32, 64, 128]),
        'use_motors': trial.suggest_categorical('use_motors', [0,1]),
        'lr': trial.suggest_loguniform('lr', 1e-6, 1e-3),
        'epochs': trial.suggest_int('epochs', 1, 15),
        'num_lstm': trial.suggest_categorical('num_lstm', [1,2]),
        'timestep': trial.suggest_int('timestep', 3, 10),
    }

    score = train_and_evaluate(hyperparams)

    print(f"Validation MAE for current model: {score}")
    return score


if __name__=="__main__":
    study = optuna.load_study(
        study_name="lstm-hpo", storage="mysql://lstm.db", direction="minimize"
    )

    n_processes = 8 
    num_trials = 200
    with multiprocessing.Pool(processes=n_processes) as pool:
        study.optimize(objective, n_trials=num_trials//n_processes, n_jobs=n_processes)