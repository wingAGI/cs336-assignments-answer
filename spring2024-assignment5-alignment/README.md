# CS336 Spring 2024 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2024_assignment5_alignment.pdf](./cs336_spring2024_assignment5_alignment.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

0. Set up a conda environment and install packages:

``` sh
conda create -n cs336_alignment python=3.10 --yes
conda activate cs336_alignment
pip install -e .'[test]'
```

1. Install Flash-Attention 2:

``` sh
export CUDA_HOME=/usr/local/cuda

pip install flash-attn --no-build-isolation
```

2. Run unit tests:

``` sh
pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

