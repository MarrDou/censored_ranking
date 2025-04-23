# Import
import numpy as np

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import math

import torch

from sklearn.base import BaseEstimator, RegressorMixin

from typing import Callable, Dict, List, Tuple


# Data generation
def generate_cox_model_multivariate(
    number_of_samples: int, 
    h: Callable[[np.ndarray], np.ndarray], 
    features_dimension: int = 4, 
    lambda_tuning: float = 1
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic data following a Cox proportional hazards model.

    Parameters:
    -----------
    number_of_samples : int
        The number of samples to generate.
    h : Callable[[np.ndarray], np.ndarray]
        A callable function that takes a 2D feature matrix (number_of_samples, features_dimension)
        and returns a 1D array (number_of_samples) of hazard rates.
    features_dimension : int, optional (default=4)
        The number of features for the feature matrix `X`.
    lambda_tuning : float, optional (default=1)
        A tuning parameter to scale the coefficients `beta_C`.

    Returns:
    --------
    Dict[str, np.ndarray]
        A dictionary containing the following keys:
        - 'T_tilde': np.ndarray
            The observed times (min(T, C)).
        - 'delta': np.ndarray
            Indicator array where 1 means the event was observed (not censored) and 0 means censored.
        - 'T': np.ndarray
            The true event times.
        - 'C': np.ndarray
            The censoring times.
        - 'X': np.ndarray
            The feature matrix.
    """
    # Generate beta coefficients with alternating signs and scaled by lambda_tuning
    beta_C = lambda_tuning * (np.array([(-1) ** i for i in range(features_dimension)]) / 2 + 1 / 2)

    # Generate feature matrix X
    X = np.random.uniform(size=(number_of_samples, features_dimension))

    # Generate true event times T using the hazard function h
    T = np.random.exponential(scale=np.exp(-h(X)), size=number_of_samples)

    # Generate censoring times C based on X and beta_C
    C = np.random.exponential(scale=np.exp(-X @ beta_C), size=number_of_samples)

    # Determine observed events (T <= C)
    delta = (T <= C).astype(int)  # 1 if observed, 0 if censored

    return {
        'T_tilde': np.minimum(T, C),  # Observed time (min of T and C)
        'delta': delta,               # Censorship indicator
        'T': T,                       # True event times
        'C': C,                       # Censoring times
        'X': X                        # Feature matrix
    }

def maximum_of_C_index_cox_model(
    h: Callable[[np.ndarray], np.ndarray], 
    features_dimension: int, 
    number_of_monte_carlo_iteration: int
) -> float:
    """
    Compute the theoretical maximum of the C-Index for a Cox Model using Monte Carlo sampling.

    Parameters:
    -----------
    h : Callable[[np.ndarray], np.ndarray]
        A scoring function that takes a 2D feature matrix (number_of_samples, features_dimension)
        and returns a 1D array of predicted scores.
    features_dimension : int
        The number of features for the randomly generated samples.
    number_of_monte_carlo_iteration : int
        The number of Monte Carlo iterations (i.e., pairs of samples) to approximate the C-Index.

    Returns:
    --------
    float
        The estimated theoretical maximum of the C-Index.
    """
    # Generate random feature matrices for two groups of samples
    X1 = np.random.rand(number_of_monte_carlo_iteration, features_dimension)
    X2 = np.random.rand(number_of_monte_carlo_iteration, features_dimension)

    # Compute predicted scores for each group
    h_X1 = h(X1)
    h_X2 = h(X2)

    # Compute the pairwise probabilities and the psi function
    probabilities = 1 / (1 + np.exp(h_X1 - h_X2))
    psi_values = np.maximum(probabilities, 1 - probabilities)  # Vectorized calculation of psi

    # Return the mean of psi values as the maximum C-index
    return psi_values.mean()

def monte_carlo_cox_concordance_index(
    f: Callable[[np.ndarray], np.ndarray], 
    number_of_monte_carlo_iteration: int, 
    features_dimension: int, 
    h: Callable[[np.ndarray], np.ndarray]
) -> float:
    """
    Estimate the concordance index (C-Index) for a given function `f` 
    when data follow a Cox proportional hazards model.

    Parameters:
    -----------
    f : Callable[[np.ndarray], np.ndarray]
        A scoring function that takes a 2D feature matrix (number_of_samples, features_dimension)
        and returns a 1D array of predicted scores.
    number_of_monte_carlo_iteration : int
        The number of Monte Carlo iterations (i.e., the number of samples to generate).
    features_dimension : int
        The number of features in the generated samples.
    h : Callable[[np.ndarray], np.ndarray]
        A function representing the hazard model, which takes a 2D feature matrix (number_of_samples, features_dimension)
        and returns a 1D array of hazard values.

    Returns:
    --------
    float
        The estimated concordance index (C-Index), which ranges from 0 to 1. 
        Returns `np.nan` if no valid pairs are found during the calculation.
    """
    # Generate two sets of random samples
    X1 = np.random.uniform(size=(number_of_monte_carlo_iteration, features_dimension))
    T1 = np.random.exponential(scale=np.exp(-h(X1)), size=number_of_monte_carlo_iteration)
    
    X2 = np.random.uniform(size=(number_of_monte_carlo_iteration, features_dimension))
    T2 = np.random.exponential(scale=np.exp(-h(X2)), size=number_of_monte_carlo_iteration)

    # Generate all pairs of indices
    i, j = np.meshgrid(
        np.arange(number_of_monte_carlo_iteration), 
        np.arange(number_of_monte_carlo_iteration), 
        indexing='ij'
    )

    # Mask for pairs where T1[i] < T2[j]
    mask = T1[i] < T2[j]

    # Apply the scoring function to the masked pairs
    f1 = f(X1[i[mask]])
    f2 = f(X2[j[mask]])

    # Count concordant pairs
    concordant = np.sum(f1 < f2)
    total_pairs = np.sum(mask)

    # Return concordance index or np.nan if no valid pairs are found
    return concordant / total_pairs if total_pairs > 0 else np.nan

# Estimation of the C-Index
def hat_conditional_T_survival_values(
    X: np.ndarray, 
    delta: np.ndarray, 
    T_tilde: np.ndarray
) -> List[np.ndarray]:
    """
    Compute conditional survival values for each individual in the dataset.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    delta : np.ndarray
        Censorship indicator (1 if event occurred, 0 if censored).
    T_tilde : np.ndarray
        Observed event or censoring times.

    Returns:
    --------
    List[np.ndarray]
        A list of size `n_samples`, where each element is a 1D array containing 
        the predicted survival probabilities for the corresponding individual.
    """
    # Fit a Random Survival Forest on the data
    survival_forest = RandomSurvivalForest().fit(X, Surv.from_arrays(delta, T_tilde))
    
    # Predict survival functions for all individuals in a vectorized way
    survival_functions = survival_forest.predict_survival_function(X, return_array=True)
    
    # Convert survival functions into a list of 1D numpy arrays
    conditional_survival_values = [func.flatten() for func in survival_functions]
    
    return conditional_survival_values


def integral_of_hatS_dhatS(
    X: np.ndarray, 
    delta: np.ndarray, 
    T_tilde: np.ndarray
) -> np.ndarray:
    """
    Compute the matrix of integrals:
    (  ∫ S_T(t | X_j) dS_T(t | X_i)  )_{i, j}

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    delta : np.ndarray
        Censorship indicator (1 if event occurred, 0 if censored).
    T_tilde : np.ndarray
        Observed event or censoring times.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (n_samples, n_samples) representing the computed integrals.
    """
    # Compute conditional survival values
    conditional_hatS_values = np.array(hat_conditional_T_survival_values(X, delta, T_tilde))
    
    # Compute jumps for survival probabilities
    conditional_hatS_jump = np.diff(
        np.concatenate([conditional_hatS_values, np.zeros((len(conditional_hatS_values), 1))], axis=1),
        axis=1
    )
    
    # Compute the integral matrix
    integral_matrix = conditional_hatS_jump@conditional_hatS_values.T
    
    return integral_matrix

def hatSi_cond_Xi(X,delta,T_tilde):
    n = len(X)
    conditional_survival_values = []
    conditional_survival_func = RandomSurvivalForest().fit(X, Surv.from_arrays(delta, T_tilde))
    
    for i in range(n):
        conditional_survival_func_i = conditional_survival_func.predict_survival_function((X[i].reshape(1,-1)))[0]
        cond = np.where(conditional_survival_func_i.x <= T_tilde[i])[0]
        if len(cond) > 0:
            conditional_survival_values.append( conditional_survival_func_i.y[cond[-1]] )
        else:
            conditional_survival_values.append( conditional_survival_func_i.y[0] )
    return np.array(conditional_survival_values)


class SHCIM(BaseEstimator, RegressorMixin):
    ''' Compute the SHCIM estimator of the C-Index '''
    
    def __init__(self, X:torch.Tensor, delta:torch.Tensor, T_tilde:torch.Tensor, device):
        self.X = X
        self.delta = delta
        self.T_tilde = T_tilde
        self.weights = None
        self.device = device
        
    def fit(self):
        self.weights = torch.tensor(integral_of_hatS_dhatS(self.X.cpu(), self.delta.cpu(), self.T_tilde.cpu()), dtype=torch.float32, device=self.device)
        return self
    
    def predict(self, indicatrice_function_estimate:Callable, f:Callable) -> torch.float:
        n = self.X.size(0)
        f_X = f(self.X).view(-1)
        diff_matrix = f_X.unsqueeze(1) - f_X.unsqueeze(0) #( f(X_j)-f(X_i) )_{i,j}
        diff_ind_matrix = indicatrice_function_estimate(diff_matrix)#( \phi( f(X_j)-f(X_i) ) )_{i,j}
        weighted_sigmoid = diff_ind_matrix * self.weights 
        return 2*weighted_sigmoid.sum() / (n * (n - 1))
    
class NaiveCIndex(BaseEstimator, RegressorMixin):
    ''' Compute the Naive estimator of the C-Index '''

    def __init__(self, X: torch.Tensor, delta: torch.Tensor, T_tilde: torch.Tensor, device: torch.device):
        """
        X: Tensor of features (size: n_samples x n_features)
        delta: Tensor of event indicators (1 if event occurred, 0 if censored)
        T_tilde: Tensor of observed or censored times
        device: Device to run computations on (cpu or cuda)
        """
        self.X = X.to(device)
        self.delta = delta.to(device)
        self.T_tilde = T_tilde.to(device)
        self.device = device

    def fit(self):
        # No fitting needed for the NaiveCIndex, it's just a predictor.
        return self

    def predict(self, indicatrice_function_estimate: Callable, f: Callable) -> torch.float:
        """
        Computes the Naive C-index prediction based on the given functions.
        
        indicatrice_function_estimate: Function to apply the indicator (phi) on the difference
        f: Function that returns the risk scores for each sample (e.g., model's output)
        """
        # Compute risk scores
        f_X = f(self.X).view(-1).to(self.device)

        # Calculate difference matrix (f(X_j) - f(X_i)) for all pairs
        diff_matrix = f_X.unsqueeze(1) - f_X.unsqueeze(0)

        # Apply the indicator function (phi)
        diff_ind_matrix = indicatrice_function_estimate(diff_matrix)

        # Compute the difference of times T_tilde
        T_tilde_diff = self.T_tilde.unsqueeze(1) - self.T_tilde.unsqueeze(0)

        # Create the delta matrix (apply delta to the difference of times)
        delta_matrix = self.delta.unsqueeze(0) * torch.ones_like(T_tilde_diff).to(self.device)

        # Mask valid pairs (where T_tilde_diff > 0)
        valid_pairs_mask = (T_tilde_diff > 0).float().to(self.device)

        # Compute the weighted sum for the C-index numerator and denominator
        sigma_f_diff = diff_ind_matrix * valid_pairs_mask
        numerator = torch.sum(delta_matrix * sigma_f_diff)
        denominator = torch.sum(delta_matrix * valid_pairs_mask)

        return numerator / denominator if denominator != 0 else torch.tensor(0.0, device=self.device)
    
class IPCWCIndex(BaseEstimator, RegressorMixin):
    ''' Compute the IPCW ranking estimator of the C-Index '''
    
    def __init__(self, X: torch.Tensor, delta: torch.Tensor, T_tilde: torch.Tensor, device: torch.device):
        """
        X: Tensor of features (size: n_samples x n_features)
        delta: Tensor of event indicators (1 if event occurred, 0 if censored)
        T_tilde: Tensor of observed or censored times
        device: Device to run computations on (cpu or cuda)
        """
        self.X = X.to(device)
        self.delta = delta.to(device)
        self.T_tilde = T_tilde.to(device)
        self.weights = None
        self.device = device

    def fit(self):
        # Compute the weights on the appropriate device
        self.weights = torch.tensor(hatSi_cond_Xi(self.X.cpu().numpy(), 1-self.delta.cpu().numpy(), self.T_tilde.cpu().numpy()), dtype=torch.float32, device=self.device)
        eps = torch.finfo(torch.float32).eps
        self.weights = torch.where(self.weights == 0, torch.tensor(eps, device=self.device), self.weights)
        return self
    
    def predict(self, indicatrice_function_estimate: Callable, f: Callable) -> torch.float:
        """
        Computes the IPCW C-index prediction based on the given functions.
        
        indicatrice_function_estimate: Function to apply the indicator (phi) on the difference
        f: Function that returns the risk scores for each sample (e.g., model's output)
        """
        n = self.delta.size(0)
        
        # Compute the risk scores
        f_X = f(self.X).view(-1).to(self.device)

        # Calculate difference matrix (f_j - f_i) for all pairs
        diff_matrix = f_X.unsqueeze(0) - f_X.unsqueeze(1)
        # Apply the indicator function (phi)
        diff_ind_matrix = indicatrice_function_estimate(diff_matrix)
        # Create the delta matrix (delta_i * delta_j)
        delta_matrix = self.delta.unsqueeze(0) * self.delta.unsqueeze(1)

        # Indicator matrix for comparing T_tilde (T_tilde_j > T_tilde_i)
        indicator_matrix = (self.T_tilde.unsqueeze(0) > self.T_tilde.unsqueeze(1)).float()

        # Compute the weights matrix (weights_i * weights_j)
        weights_matrix = self.weights.unsqueeze(0) * self.weights.unsqueeze(1)

        # Compute the final C-index value
        numerator = torch.sum(delta_matrix * indicator_matrix * diff_ind_matrix / weights_matrix)
        denominator = n * (n - 1)
        
        return 2 * numerator / denominator if denominator != 0 else torch.tensor(0.0, device=self.device)
    
class IPCWRegression(BaseEstimator, RegressorMixin):
    ''' Compute the IPCW regression estimator '''
    
    def __init__(self, X: torch.Tensor, delta: torch.Tensor, T_tilde: torch.Tensor, device: torch.device):
        """
        X: Tensor of features (size: n_samples x n_features)
        delta: Tensor of event indicators (1 if event occurred, 0 if censored)
        T_tilde: Tensor of observed or censored times
        device: Device to run computations on (cpu or cuda)
        """
        self.X = X.to(device)
        self.delta = delta.to(device)
        self.T_tilde = T_tilde.to(device)
        self.weights = None
        self.device = device

    def fit(self):
        # Compute the weights on the appropriate device
        self.weights = torch.tensor(hatSi_cond_Xi(self.X.cpu().numpy(), 1-self.delta.cpu().numpy(), self.T_tilde.cpu().numpy()), dtype=torch.float32, device=self.device)
        eps = torch.finfo(torch.float32).eps
        self.weights = torch.where(self.weights == 0, torch.tensor(eps, device=self.device), self.weights)
        return self
    
    def predict(self, indicatrice_function_estimate: Callable, f: Callable) -> torch.float:
        """
        Computes the IPCW regression prediction based on the given functions.
        
        indicatrice_function_estimate: Function to apply the indicator (phi) on the difference
        f: Function that returns the risk scores for each sample (e.g., model's output)
        """
        # Compute the risk scores
        f_X = f(self.X).view(-1).to(self.device)

        # Calculate loss numerator: delta * (T_tilde - f(X))^2
        loss_numerator = self.delta * (self.T_tilde - f_X) ** 2

        # Compute final prediction by summing the loss over weights
        return torch.sum(loss_numerator / self.weights) / self.delta.size(0)