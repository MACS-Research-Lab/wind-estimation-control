# Hybrid Control Framework of UAVs Under Varying Wind and Payload Conditions

See `src/setup.txt` to set up the Python environment.

## Reproducing Results

To reproduce results, any of the trained models can be used by running the notebook files. All results included in the paper can be obtained by running the respective notebook file. Interesting ones may be the `___ Wind Impact.ipynb` files and `Plot Trajectories.ipynb` files. 

To train your own models or reproduce our trained models, make the appropriate changes to `scripts/opt_multirotorenv.py` and run that file as denoted in `src/README.txt`. These models will take about a day to arrive at the same results as shown in the paper on a modern GPU with 10 parallel processes. To train the LSTM model, the training data will linked here shortly. 

See `Demo.ipynb` for basic usage of the UAV without any disturbances.

## Hyperparameters

The final hyperparameters for each of the models can be seen by running `optuna dashboard sqlite:///filename.db` from the `studies/` directory where `filename` is the model you want to inspect the hyperparmeters. In this dashboard, you can see the impacts of each hyperparmeter along with other useful information. The final models chosen are the models with the highest `value` in this dashboard.