import time
import os
import csv
import numpy as np
import pandas as pd
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def run_psindyc(plot_dir,dt,dv,StateData, InputData, StateData_v, InputData_v,Experiment_name,training_name, validation_name, states, inputs):

    # Define datasets    
    os.makedirs(plot_dir, exist_ok=True)

    # Estimation data
    xout = StateData.T
    u = InputData.T
    sampling_interval = 1
    num_samples = StateData.T.shape[0]
    tout = np.arange(0, num_samples * sampling_interval, sampling_interval)

    # Validation data
    xoutv = StateData_v.T
    uv = InputData_v.T
    sampling_interval = 1
    num_samples = StateData_v.T.shape[0]
    toutv = np.arange(0, num_samples * sampling_interval, sampling_interval)

    # Initialize and fit SINDy model
    sindy_library = ps.PolynomialLibrary(degree=1, include_bias=False,include_interaction=False)
    optimizer = ps.STLSQ(threshold=0.006) 
    model = ps.SINDy(feature_library=sindy_library, optimizer=optimizer,discrete_time=True)
    model.fit(xout,t=1, u=u)

    # Print state space equations
    Xi = model.coefficients()
   
    # Get A and B matrices 
    r = Xi.shape[0]
    A_SINDyc = Xi[:r, :r]
    B_SINDyc = Xi[:r, r:]
    Eigenvalues=np.sort(np.linalg.eigvals(Xi[:4, :4]))
   
   #Prediction for training
    x_pred = xout[0, :]  
    x_pred_e = [x_pred]
    for u_t in u:
        x_pred = A_SINDyc @ x_pred + B_SINDyc @ u_t  
        x_pred_e.append(x_pred)
    x_pred_e = np.array(x_pred_e[:-1])

   #Prediction for validation
    x_predv = xoutv[0, :]  
    x_pred_v = [x_predv]
    for u_t in uv:
        x_predv = A_SINDyc @ x_predv + B_SINDyc @ u_t  
        x_pred_v.append(x_predv)
    x_pred_v = np.array(x_pred_v[:-1])

    # Calculate metrics for training data
    r2_values = []
    mre_values = []
    mse_values = []
    
    for i in range(xout.shape[1]):
        r2 = r2_score(xout[:len(x_pred_e), i], x_pred_e[:, i])
        r2_values.append(r2)

        mre = np.mean(np.abs((xout[:len(x_pred_e), i] - x_pred_e[:, i]) / xout[:len(x_pred_e), i])) * 100
        mre_values.append(mre)

        mse = mean_squared_error(xout[:len(x_pred_e), i], x_pred_e[:, i])
        mse_values.append(mse)

    # Calculate metrics for validation data
    for i in range(xoutv.shape[1]):
        r2 = r2_score(xoutv[:len(x_pred_v), i], x_pred_v[:, i])
        r2_values.append(r2)
     
        mre = np.mean(np.abs((xoutv[:len(x_pred_v), i] - x_pred_v[:, i]) / xoutv[:len(x_pred_v), i])) * 100
        mre_values.append(mre)

        mse = mean_squared_error(xoutv[:len(x_pred_v), i], x_pred_v[:, i])
        mse_values.append(mse)
      
    # Text file
    sindy_summary = {
        'Experiment': Experiment_name,
        'Training dataset':training_name,
        'Validation dataset': validation_name,
        'A_SINDyc': A_SINDyc.tolist(),
        'B_SINDyc': B_SINDyc.tolist(),
        'Eigenvalues':Eigenvalues,
        'R-squared': r2_values,
        'MRE': mre_values,
        'MSE': mse_values
    }

    txt_file_path = os.path.join(plot_dir, 'SINDYc_state_space.txt')
    with open(txt_file_path, 'a') as txt_file:
        for key, value in sindy_summary.items():
            txt_file.write(f"{key}: {value}\n")

    custom_labels = [f'{input}_T' for input in inputs] + [f'{state}_T' for state in states] + [f'{state}_T_PSYNDIC' for state in states] + \
                [f'{input}_V' for input in inputs] + [f'{state}_V' for state in states] +[f'{state}_V_PSYNDIC' for state in states] 
    csv_file_path = os.path.join(plot_dir, f'Prediction {validation_name} SINDYc_Results.csv')

# CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(custom_labels)
        for i in range(min(len(u), len(xout), len(x_pred_e)) - 1):
            row = list(u[i]) + list(xout[i]) + list(x_pred_e[i]) + \
                list(uv[i]) + list(xoutv[i]) + list(x_pred_v[i])
            writer.writerow(row)
    
    return A_SINDyc, B_SINDyc

