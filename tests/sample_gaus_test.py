from deterministic_gaussian_sampling_fibonacci import sample_gaussian_fibonacci
import pytest
import numpy as np

FIB_TYPES = ['ImprovedFrolov', 'ClassicalFrolov', 'Fibonacci']
FIB_DIMS = [2, 3, 4, 5, 6]


@pytest.mark.parametrize("type", FIB_TYPES)
@pytest.mark.parametrize("dim", FIB_DIMS)
def test_001_gaus_sample_test(type, dim):
	if dim == 6:
		sampcount = 1000 # dim = 6, LVol = 10000, type = 'ClassicalFrolov' is unsupported
	else:
		sampcount = 10000


	np.random.seed(0)
	mu = np.random.randn(dim)
	A = np.random.randn(dim, dim)
	cov = A @ A.T  # Ensures positive semi-definite covariance
	samp = sample_gaussian_fibonacci(mu, cov, sampcount, type)

	mu_pred = np.mean(samp, axis=0) # shape (3,)
	C_pred = np.cov(samp, rowvar=False, bias=True)

	assert np.all(np.isclose(mu, mu_pred, 10**-2))
	if sampcount > 10000:
		assert np.all(np.isclose(cov, C_pred, 10**-2))

@pytest.mark.parametrize("type", FIB_TYPES)
@pytest.mark.parametrize("dim", FIB_DIMS)
def test_002_gaus_sample_count_edgecase(type, dim):
	sampcount = 1


	np.random.seed(0)
	mu = np.random.randn(dim)
	A = np.random.randn(dim, dim)
	cov = A @ A.T  # Ensures positive semi-definite covariance
	samp = sample_gaussian_fibonacci(mu, cov, sampcount, type)

	mu_pred = np.mean(samp, axis=0) # shape (3,)
	assert np.all(np.isclose(mu, mu_pred, 10**-6))

	assert samp.shape == (sampcount, dim)


@pytest.mark.parametrize("type", FIB_TYPES)
@pytest.mark.parametrize("dim", FIB_DIMS)
def test_002_gaus_0_sample_count(type, dim):
	sampcount = 0


	np.random.seed(0)
	mu = np.random.randn(dim)
	A = np.random.randn(dim, dim)
	cov = A @ A.T  # Ensures positive semi-definite covariance
	samp = sample_gaussian_fibonacci(mu, cov, sampcount, type)

	assert samp.shape == (sampcount, dim)

