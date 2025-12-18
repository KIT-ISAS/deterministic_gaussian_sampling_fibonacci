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

# sample_gaussian_fibonacci(mean, covariance_matrix, sample_count, sample_type)
samples = sample_gaussian_fibonacci([1, 1], np.eye(2), 100, "Fibonacci")
```

## Maximum supported sample counts by dimension and type

| dim | ClassicalFrolov | ImprovedFrolov | Fibonacci | Galois |
| --- | --------------: | -------------: | --------: | -----: |
| 2   |          100003 |         100003 |     99997 |      - |
| 3   |           99997 |         100007 |     99995 |      - |
| 4   |          100009 |          99965 |     99997 |      - |
| 5   |          100217 |         100017 |     99989 | 105021 |
| 6   |            5987 |          99969 |    100005 | 104997 |

