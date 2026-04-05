import numpy as np
import pandas as pd

def calculate_robust_kmo(dataset):
    """
    Calcula el KMO Global de manera robusta usando la pseudoinversa 
    para evitar el cuelgue con matrices singulares (One-Hot Encoding).
    """
    # Matriz de correlación
    corr_mat = dataset.corr().values
    
    # Pseudoinversa (inmune a multicolinealidad perfecta)
    inv_corr_mat = np.linalg.pinv(corr_mat)
    
    # Matriz de correlaciones parciales
    A = np.ones(corr_mat.shape)
    for i in range(0, corr_mat.shape[0]):
        for j in range(i, corr_mat.shape[1]):
            A[i,j] = -(inv_corr_mat[i,j]) / (np.sqrt(inv_corr_mat[i,i]*inv_corr_mat[j,j]))
            A[j,i] = A[i,j]
            
    corr_mat = np.asarray(corr_mat)
    kmo_num = np.sum(np.square(corr_mat)) - np.sum(np.square(np.diagonal(corr_mat)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    
    return kmo_num / kmo_denom