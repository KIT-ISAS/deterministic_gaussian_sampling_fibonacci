import numpy as np
import h5py
from importlib import resources
from scipy.stats import norm
from functools import lru_cache


SUPPORED_DIM = [2, 3, 4, 5, 6]
FIB_TYPES = ['ImprovedFrolov', 'ClassicalFrolov', 'Fibonacci']

"""returns np array of shape (N, dim)"""
@lru_cache
def _load_data(dim, LVol, type='Fibonacci'):
	filename = f"dim={dim:02d}_LVol={LVol}_{type}.mat"
	mat_res = resources.files("deterministic_gaussian_sampling_fibonacci.data") / filename
	try:
		with resources.as_file(mat_res) as mat_path:
			with h5py.File(mat_path, "r") as f:
				data = f.get('X')
				data = np.array(data)
	except Exception as e:
		raise RuntimeError(f"{type} Samples for dim={dim} and LVol={LVol} are not available. please check the docuentation for supported dimensions and LVol values.")
	return data

def transform_grid_gaussian(grid, mu, cov):
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

	return gaus

def _check_parameters(mu, cov, LVol, type):
	if type not in FIB_TYPES:
		raise ValueError(f"Type {type} is not supported. Supported types are {FIB_TYPES}.")

	dim = mu.shape[0]
	if dim not in SUPPORED_DIM:
		raise ValueError(f"Dimension {dim} is not supported. Supported dimensions are {SUPPORED_DIM}.")

	if cov.shape != (dim, dim):
		raise ValueError(f"Covariance matrix shape {cov.shape} does not match mu shape {mu.shape}.")

def sample_gaussian_fibonacci(mu, cov, LVol=100, type='Fibonacci'):
	mu = np.asarray(mu)
	_check_parameters(mu, cov, LVol, type)

	dim = mu.shape[0]
	grid = _load_data(dim, LVol, type)

	grid = grid + 0.5

	samples = transform_grid_gaussian(grid, mu, cov)
	return samples
