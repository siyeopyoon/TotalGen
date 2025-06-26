# TotalGen

This repository accompanies the paper ["Cascaded 3D Diffusion Models for Whole-body 18F-FDG PET/CT Synthesis from Demographics"](https://arxiv.org/pdf/2505.22489). The overall workflow is illustrated in [Resource/Main1.pdf](Resource/Main1.pdf), while quantitative results are summarized in [Resource/BMIs.pdf](Resource/BMIs.pdf) and [Resource/Graph.pdf](Resource/Graph.pdf).

## Project Layout

- **0_Preprocess** – data conversion, segmentation, and intensity scaling scripts.
- **1_Trainings** – training configurations for cascaded diffusion and flow models.
- **2_Tests** – inference utilities and shared runtime libraries.
- **3_Evaluations** – segmentation-based evaluation and statistical analysis.
- **Resource** – diagrams and result figures referenced throughout the READMEs.

All codes were verified locally before publication.

## Getting Started

1. Follow the preprocessing steps in [0_Preprocess](0_Preprocess/README.md).
2. Train the models as described in [1_Trainings](1_Trainings/README.md).
3. Run inference using [2_Tests](2_Tests/README.md).
4. Evaluate results with [3_Evaluations](3_Evaluations/README.md).

See the [arXiv manuscript](https://arxiv.org/pdf/2505.22489) for methodology and experimental details.
