# SVD Benchmark

This project measures the performance of different SVD solvers.

### Dependencies
* numpy
* scipy
* cupy
* pytest
* pytest-benchmark

#### Optional
* [torch]
* [arrayfire]

## How to run
```
pytest tests/test_svd_cpu.py --sizes=100,200 --dtypes=float32,float64
pytest tests/test_svd_gpu.py --sizes=100,200 --dtypes=float32,complex32
```