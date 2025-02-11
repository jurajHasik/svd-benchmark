from setuptools import setup, find_packages
setup(
    name='svd-benchmarks',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    'numpy',
    'scipy',
    'cupy',
    'pytest',
    'pytest-benchmark'
]
)
