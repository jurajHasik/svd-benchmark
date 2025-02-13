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
* [slepc]
* [primme]

## How to run
```
pytest tests/test_svd_cpu.py --sizes=100,200 --dtypes=float32,float64
pytest tests/test_svd_gpu.py --sizes=100,200 --dtypes=float32,complex32
```

For partial SVD solvers, `--ratio` specifies fraction of largest singular triplets to compute

```
pytest tests/test_partial_svd_cpu.py --sizes=2000 --dtypes=float32,float64 --ratio 0.04
pytest tests/test_partial_svd_gpu.py --sizes=2000 --dtypes=float32,complex32
```