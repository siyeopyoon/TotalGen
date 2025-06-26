# TotalGen

Implementation of cascaded 3D diffusion models for whole-body 18F-FDG PET/CT synthesis from demographics as described in our [arXiv paper](https://arxiv.org/pdf/2505.22489).

Contact: [yooneige@gmail.com](mailto:yooneige@gmail.com) or [syoon5@mgh.harvard.edu](mailto:syoon5@mgh.harvard.edu)

## Overview
The pipeline combines conditional diffusion and flow-based models in a cascaded architecture to synthesize paired PET/CT volumes. Training begins at a low spatial resolution and progressively refines predictions using super-resolution stages. The method is designed for demographic conditioning and generates realistic whole-body images.

![Pipeline overview](Resource/Main1.pdf)
*Figure 1. Diagram of the cascaded diffusion and flow modules.*

![BMI analysis](Resource/BMIs.pdf)
*Figure 2. BMI distribution of the dataset and generated cohort.*

![Performance graph](Resource/Graph.pdf)
*Figure 3. Quantitative comparison across different model variants.*

## Project Layout
- **0_Preprocess** – data conversion, segmentation, and intensity scaling scripts.
- **1_Trainings** – training configurations for cascaded diffusion and flow models.
- **2_Tests** – inference utilities and shared runtime libraries.
- **3_Evaluations** – segmentation-based evaluation and statistical analysis.
- **Resource** – diagrams and result figures.

All codes were verified locally before publication.

## Getting Started
1. Follow the preprocessing steps in [0_Preprocess](0_Preprocess/README.md).
2. Train the models as described in [1_Trainings](1_Trainings/README.md).
3. Run inference using [2_Tests](2_Tests/README.md).
4. Evaluate results with [3_Evaluations](3_Evaluations/README.md).

See the [arXiv manuscript](https://arxiv.org/pdf/2505.22489) for detailed methodology and experimental setup.
