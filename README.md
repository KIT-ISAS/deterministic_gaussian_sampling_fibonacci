 # Deterministic gaussian sampling

 Deterministic sampling via orthogonal inverse transform of low-discrepancy Fibonacci grids. 

 # Usage Example

```python
from deterministic_gaussian_sampling_fibonacci import sample_gaussian_fibonacci
import numpy as np

samples = sample_gaussian_fibonacci([1, 1], np.eye(2), 100, "Fibonacci")
```

# Supported sample sizes and types
| dim | ClassicalFrolov (LVols)    | ImprovedFrolov (LVols)     | Fibonacci (LVols)          | Galois (LVols)             |
| --- | -------------------------- | -------------------------- | -------------------------- | -------------------------- |
| 2   | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | -                          |
| 3   | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | -                          |
| 4   | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | -                          |
| 5   | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` |
| 6   | `1000`                     | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` | `100, 1000, 10000, 100000` |
