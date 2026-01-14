from functools import lru_cache
from importlib import resources
import h5py
import numpy as np

"""
Get a uniform grid within [0,1]^dim of given sample_count and type.

Args:
	dim: Dimension of the grid.
	sample_count: Number of samples to generate.
	type: Type of sequence ('Fibonacci', 'ClassicalFrolov', or 'ImprovedFrolov').
Returns:
	Grid points of shape (sample_count, dim) as a numpy array.
"""
def get_uniform_grid(dim: int, sample_count: int, type: str) -> np.ndarray:
	# try to find min grid size >= sample_count
	# as a sideeffect, grids with size < sample_count are also cached
	LVols = [100, 1000, 10_000, 100_000, 1_000_000]

	for LVol in LVols:
		try:
			grid = _load_data(dim, LVol, type)
			if grid.shape[0] >= sample_count:
				break
		except RuntimeError:
			continue
	else:
		raise RuntimeError(f"{type} Samples for dim={dim} and sample_count={sample_count} are not available. please check the docuentation for supported dimensions and sample counts.")
	

	grid = grid[np.argsort(grid[:,0])]  # sort by first dimension

	# TODO: allow randomised intervall selection
	# drop extra points to fit sample_count
	intervall = [0, sample_count] # exlusive end
	start, end = intervall
	shifted_grid = grid[start:end, :]


	# shift 0th dim to middle for rescaling
	max_0 = shifted_grid[-1,0]
	min_0 = shifted_grid[0,0]
	shifted_grid[:,0] = shifted_grid[:,0] - (min_0 + max_0)/2


	# Rescale 
	if sample_count == 1:
		return shifted_grid + 0.5  # only one point, return center point
	
	border_wanted = 1/2 - 1/(2*sample_count)
	fac = np.max(shifted_grid, axis=0) / border_wanted
	if np.all(np.isfinite(fac)):
		shifted_grid = shifted_grid / fac
	
	shifted_grid = shifted_grid + 0.5

	return  shifted_grid




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
