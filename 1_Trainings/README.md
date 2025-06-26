# Training Modules

This directory contains the scripts and configurations for model training. The overall procedure is visualized in [../Resource/Main1.pdf](../Resource/Main1.pdf).

## Subdirectories
- `EDM2_LowRes` – baseline low-resolution EDM model.
- `EDM2_SR` – super-resolution EDM training.
- `Flow_LowRes` – low-resolution flow model.
- `Flow_SR` – super-resolution flow model.

## Workflow
1. Train the low-resolution EDM model.
2. Train the flow model at the same resolution.
3. Use the super-resolution stages (`EDM2_SR` and `Flow_SR`) to refine the outputs.

All training scripts have been validated locally. Final results can be found in [../Resource/BMIs.pdf](../Resource/BMIs.pdf) and [../Resource/Graph.pdf](../Resource/Graph.pdf). See the [arXiv manuscript](https://arxiv.org/pdf/2505.22489) for complete training details.
