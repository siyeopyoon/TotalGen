# Test Scripts

Utilities for running inference with the trained cascaded diffusion models. The overall architecture is depicted in [../Resource/Main1.pdf](../Resource/Main1.pdf).

## Files
- `Cascade_EDM.py` – cascaded EDM inference.
- `Cascade_flow.py` – flow-based cascade generation.
- `Cascade_self_start.py` – self-starting cascade variant.

The `dnnlib` and `torch_utils` directories provide shared runtime code.

## Usage
Run the desired script with the path to a trained checkpoint to generate synthetic PET/CT volumes.

All tests were verified locally. Result highlights appear in [../Resource/BMIs.pdf](../Resource/BMIs.pdf) and [../Resource/Graph.pdf](../Resource/Graph.pdf). Refer to our [arXiv paper](https://arxiv.org/pdf/2505.22489) for more information.
