# Deterministic Gaussian Sampling

Deterministic sampling via orthogonal inverse transform of low-discrepancy Fibonacci grids.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/KIT-ISAS/deterministic_gaussian_sampling_fibonacci.git
```

## Usage example

```python
from deterministic_gaussian_sampling_fibonacci import sample_gaussian_fibonacci
import numpy as np

# sample_gaussian_fibonacci(mean, covariance_matrix, L_Vols, sample_type)
samples = sample_gaussian_fibonacci([1, 1], np.eye(2), 100, "Fibonacci")
```

## Supported sample sizes and types

| dim | ClassicalFrolov (LVols)    | ImprovedFrolov (LVols)     | Fibonacci (LVols)          | Galois (LVols)             |
| --- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| 2   | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | -                          |
| 3   | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | -                          |
| 4   | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | -                          |
| 5   | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` |
| 6   | `1000`                     | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` |
