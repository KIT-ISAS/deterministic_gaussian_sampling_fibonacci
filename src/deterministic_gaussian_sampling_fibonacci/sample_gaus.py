import numpy as np
from scipy.stats import norm

from .grid_util import _get_fitting_grid


SUPPORED_DIM = [2, 3, 4, 5, 6]
FIB_TYPES = ['ImprovedFrolov', 'ClassicalFrolov', 'Fibonacci', 'Galois']

def _transform_grid_gaussian(grid, mu, cov):
	eps = 1e-9
	grid = np.clip(grid, eps, 1 - eps) # avoid inf in ppf

	gaus = norm.ppf(grid)

	var = np.mean(gaus**2, axis=0)

	gaus = gaus / np.sqrt(var)

	# scale with eigen decomposition
	ew, V = np.linalg.eig(cov)

	D = np.diag(np.sqrt(ew))	

	gaus = gaus.T	# (dim,L)

	gaus = V @ D @ gaus # (dim,dim) @ (dim,dim) @ (dim,L) -> (dim,L)

	gaus = gaus.T # (L,dim)

	for i in range(gaus.shape[1]):
		gaus[:,i] += mu[i]

	return gaus, V, D

def _mean_correction(samples, mu):
	samples = samples + mu
	return samples
	

def _fast_cholesky_covariance_correction(samples, V, D):
	# see [JAIF23_Frisch] V.E

	# variance correction
	L = samples.shape[0]
	v_d = 1 / L * np.sum(samples**2, axis=0)  # shape (dim,)

	X_stdD = samples / np.sqrt(v_d)

	# Fast Cholesky Covariance Correction
	C_stdD = 1 / L * (X_stdD.T @ X_stdD)
	L_stdD = np.linalg.cholesky(C_stdD)
	L_stdD_inv = np.linalg.inv(L_stdD)

	X_Gauss = V @ D @ L_stdD_inv @ X_stdD.T  # (dim,dim) @ (dim,dim) @ (dim,dim) @ (dim,L) -> (dim,L)
	X_Gauss = X_Gauss.T  # (L,dim)
	return X_Gauss


def _check_parameters(mu, cov, LVol, type):
	if type not in FIB_TYPES:
		raise ValueError(f"Type {type} is not supported. Supported types are {FIB_TYPES}.")

	dim = mu.shape[0]
	if dim not in SUPPORED_DIM:
		raise ValueError(f"Dimension {dim} is not supported. Supported dimensions are {SUPPORED_DIM}.")

	if cov.shape != (dim, dim):
		raise ValueError(f"Covariance matrix shape {cov.shape} does not match mu shape {mu.shape}.")
	



def sample_gaussian_fibonacci(mu: list | np.ndarray, cov: np.ndarray, sample_count: int = 100, type: str = 'Fibonacci') -> np.ndarray:
	"""
	Generate deterministic Gaussian samples using Fibonacci/Frolov sequences.
	
	Args:
		mu: Mean vector of shape (dim,). Can be a list or numpy array.
		cov: Covariance matrix of shape (dim, dim).
		sample_count: Number of samples to generate. Default is 100.
		type: Type of sequence ('Fibonacci', 'ClassicalFrolov', or 'ImprovedFrolov'). Default is 'Fibonacci'.
	
	Returns:
		Gaussian samples of shape (sample_count, dim) as a numpy array.
	"""
	mu = np.asarray(mu)
	_check_parameters(mu, cov, sample_count, type)

	dim = mu.shape[0]
	grid = _get_fitting_grid(dim, sample_count, type)

	samples, V, D = _transform_grid_gaussian(grid, mu, cov)

	# center for fast cholesky correction
	samples = samples - np.mean(samples, axis=0)
	samples = _fast_cholesky_covariance_correction(samples, V, D)
	samples = _mean_correction(samples, mu)
	return samples