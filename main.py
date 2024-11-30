import os
import pandas as pd
#from DMDc import run_DMDc
from DMDc_normalized import run_DMDc
from SINDYc import run_psindyc

# Experiment
experiment_name = 'Experiment_name'
training_name = 'Training'
states = ['D5', 'D10', 'D50', 'D90']
inputs = ['AFR', 'LFR', 'BFR']

# Directory
plot_dir = 'EXPERIMENT'
os.makedirs(plot_dir, exist_ok=True)

# Load data
dt = pd.read_csv('datasets.csv')  # Training data
state_data_train = dt[['D5_T', 'D10_T', 'D50_T', 'D90_T']].values.T
input_data_train = dt[['AFR_T', 'LFR_T', 'BFR_T']].values.T

dv = pd.read_csv('datasets.csv')  # Validation data

# Validation
validation_configs = [
    {'name': 'Validation_1', 'states': ['D5_V1', 'D10_V1', 'D50_V1', 'D90_V1'], 'inputs': ['AFR_V1', 'LFR_V1', 'BFR_V1']},
    {'name': 'Validation_2', 'states': ['D5_V2', 'D10_V2', 'D50_V2', 'D90_V2'], 'inputs': ['AFR_V2', 'LFR_V2', 'BFR_V2']},
    {'name': 'Validation_3', 'states': ['D5_V3', 'D10_V3', 'D50_V3', 'D90_V3'], 'inputs': ['AFR_V3', 'LFR_V3', 'BFR_V3']}
]

for config in validation_configs:
    validation_name = config['name']
    state_data_val = dv[config['states']].values.T
    input_data_val = dv[config['inputs']].values.T
    
    # Run DMDc
    A_DMDc, B_DMDc = run_DMDc(
        plot_dir, state_data_train, input_data_train,
        state_data_val, input_data_val, experiment_name,
        training_name, validation_name, states, inputs
    )
    
    # Run SINDy
    A_SINDyc, B_SINDyc = run_psindyc(
        plot_dir, dt, dv, state_data_train, input_data_train,
        state_data_val, input_data_val, experiment_name,
        training_name, validation_name, states, inputs
    )
    
    print(f"Processed {validation_name} successfully.")
