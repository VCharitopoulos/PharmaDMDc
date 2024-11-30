import os
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydmd.plotter import plot_eigs
from numpy import dot, multiply, diag, power
from numpy.linalg import inv, eig, pinv
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from numpy import real, imag
from numbers import Number
import warnings

def run_DMDc(plot_dir,StateData, InputData, StateData_v, InputData_v,Experiment_name,training_name, validation_name, states, inputs):

    #Data matrixes for DMD
    X = StateData[:, :-1]
    X_v = StateData_v[:, :-1]

    Xp = StateData[:, 1:]
    Xp_v = StateData_v[:, 1:]

    Ups = InputData[:, :-1]
    Ups_v = InputData_v[:, :-1]
       
    Omega = np.vstack((X, Ups))

    #SVD for Omega matrix

    U, Sig, V = np.linalg.svd(Omega, full_matrices=False)
    V = V.conj().T

    # Truncation
    svd_rank=-1
    svd_rank_Xp=-1
    def _svht(sigma_svd: np.ndarray, rows: int, cols: int) -> int:
        beta = np.divide(*sorted((rows, cols)))
        omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
        tau = np.median(sigma_svd) * omega
        rank = np.sum(sigma_svd > tau)

        if rank == 0:
            warnings.warn(
                "SVD optimal rank is 0. The largest singular values are "
                "indistinguishable from noise. Setting rank truncation to 1.",
                RuntimeWarning,
         )
            rank = 1

        return rank

    def _compute_rank(
        sigma_svd: np.ndarray, rows: int, cols: int, svd_rank: Number) -> int:
        if svd_rank == 0:
            rank = _svht(sigma_svd, rows, cols)
        elif 0 < svd_rank < 1:
            cumulative_energy = np.cumsum(sigma_svd**2 / (sigma_svd**2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, sigma_svd.size)
        else:
            rank = min(rows, cols)

        return rank

    rtil = _compute_rank(Sig, Omega.shape[0], Omega.shape[1], svd_rank)

    Util = U[:, :rtil]

    Sigtil = Sig[:rtil]

    Vtil = V[:, :rtil]

    # SVD for Xp matrix

    U, Sig, V = np.linalg.svd(Xp, full_matrices=False)

    # Truncation

    r = _compute_rank(Sig, Xp.shape[0], Xp.shape[1], svd_rank_Xp)

    V = V.conj().T
    Uhat = U[:, :r]
    Uhat_MPC= Uhat

    Sighat = Sig[:r]

    Vbar = V[:, :r]

    n = X.shape[0]

    q = Ups.shape[0]

    U_1 = Util[:n, :]

    U_2 = Util[n:, :]

    #SMatrix A and B from DMDc

    approxA = Uhat.T.conj() @ (Xp) @ Vtil @ np.diag(np.reciprocal(Sigtil)) @ U_1.T.conj() @ Uhat
    approxA_MPC= (np.dot((np.dot(Uhat, approxA)),Uhat.T))
    approxB = Uhat.T.conj() @ (Xp) @ Vtil @ np.diag(np.reciprocal(Sigtil))  @ U_2.T.conj()
    approxB_MPC=np.dot(Uhat, approxB)
    
    # DMD modes
    D, W = np.linalg.eig(approxA)
    Phi = Xp @ Vtil @ np.diag(np.reciprocal(Sigtil)) @ U_1.T.conj() @ Uhat @ W
    #Amplitud
    b = dot(pinv(Phi), X[:,0]) 
    #Prediction results
    X_DMDc = np.zeros((X.shape[0], X.shape[1]))
    X_DMDcv = np.zeros((X_v.shape[0], X_v.shape[1]))
    X_DMDc[:, 0] = X[:, 0]
    X_DMDcv[:, 0] = X_v[:, 0]

    for k in range(X.shape[1] - 1):
        X_DMDc[:, k + 1] = Uhat @ (approxA @ (Uhat.T @ X_DMDc[:, k]) + approxB @ InputData[:, k])

    for k in range(X_v.shape[1] - 1):
        X_DMDcv[:, k + 1] = Uhat @ (approxA @ (Uhat.T @ X_DMDcv[:, k]) + approxB @ InputData_v[:, k])

    reconstruction_e = X_DMDc.real
    reconstruction_v = X_DMDcv.real

    #Error metrics
    def _statistics(reconstruction_e,X):
        r2_values = []
        mre_values = []
        mse_values = []
        for i in range(X.shape[0]):     
            r2 = r2_score(X[i, :], reconstruction_e.real[i, :])
            r2_values.append(r2)
    
            mre = np.mean(np.abs((X[i, :] - reconstruction_e.real[i, :]) / X[i, :])) * 100
            mre_values.append(mre)
    
            mse = mean_squared_error(X[i, :], reconstruction_e.real[i, :])
            mse_values.append(mse)
        return r2_values,mre_values,mse_values
    
    r2_values,mre_values,mse_values=_statistics(reconstruction_e,X)
    r2_values_v,mre_values_v,mse_values_v=_statistics(reconstruction_v,X_v)

     # Text file
    dmd_summary = {
        'Experiment':Experiment_name,
        'Training dataset':training_name,
        'Validation dataset': validation_name,
        'R-squared_training': r2_values,
        'MRE_training': mre_values,
        'MSE_training': mse_values,
        'R-squared_validation': r2_values_v,
        'MRE_training_validation': mre_values_v,
        'MSE_training_validation': mse_values_v,
        'A_matrix': approxA_MPC,
        'B_matrix': approxB_MPC,
        'Eigenvalues': D.tolist(),
        'Amplitud': b.tolist(),
        'Modes': Phi.flatten().tolist()
    }

    txt_file_path = os.path.join(plot_dir, 'DMDc_state_space.txt')

    with open(txt_file_path, 'a') as txt_file:
        for key, value in dmd_summary.items():
            txt_file.write(f"{key}: {value}\n")
  
          
    custom_labels = [f'{input}_T' for input in inputs] + [f'{state}_T' for state in states] + [f'{state}_T_DMDC' for state in states] + \
                [f'{input}_V' for input in inputs] + [f'{state}_V' for state in states] +[f'{state}_V_DMDC' for state in states] 
    csv_file_path = os.path.join(plot_dir, f'Prediction {validation_name} DMDc_Results.csv')

# CSV file 
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(custom_labels)

        for i in range(len(Ups.T)):
            row = list(Ups.T[i]) + list(StateData.T[i]) + list(reconstruction_e.T[i]) + \
                list(InputData_v.T[i]) + list(StateData_v.T[i]) + list(reconstruction_v.T[i])
            writer.writerow(row)

    return  approxA_MPC, approxB_MPC

if __name__ == "__main__":
    A_DMDc, B_DMDc = run_DMDc()
