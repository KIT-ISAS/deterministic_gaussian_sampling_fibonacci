import pytest
import numpy as np
from deterministic_gaussian_sampling_fibonacci import get_uniform_grid


FIB_DIMS = [2, 3, 4, 5, 6]
FIB_TYPES = ['ImprovedFrolov', 'ClassicalFrolov', 'Fibonacci']


@pytest.mark.parametrize("type", FIB_TYPES)
@pytest.mark.parametrize("dim", FIB_DIMS)
def test_001_gaus_sample_test(type, dim):
	if dim == 6:
		sampcount = 1000 # dim = 6, LVol = 10000, type = 'ClassicalFrolov' is unsupported
	else:
		sampcount = 10000

	grid = get_uniform_grid(dim, sampcount, type)

	assert grid.shape == (sampcount, dim)