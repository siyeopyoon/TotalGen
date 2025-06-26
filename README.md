# Cascaded 3D Diffusion Models for Whole-body ¹⁸F-FDG PET/CT Synthesis
**Link to full paper (PDF):** https://arxiv.org/pdf/2505.22489
## Abstract
This repository implements a cascaded volumetric image synthesis pipeline that combines conditional diffusion and flow-based super-resolution modules to generate realistic whole-body ¹⁸F-FDG PET/CT scans from demographic variables (height, weight, sex, age). The approach proceeds from low-resolution diffusion-based generation to high-resolution refinement, enabling anatomy-preserving, demographically conditioned volumes. Comprehensive preprocessing, training, inference, and evaluation scripts facilitate reproduction and extension.

## Features
- Demographic conditioning on height, weight, sex, and age  
- Two-stage generation: low-resolution diffusion → super-resolution flow  
- Modular scripts for data handling, training, inference, and evaluation  
- Quantitative metrics: Dice, Hausdorff, intensity statistics  

## Repository Structure
```
0_Preprocess/       Data conversion, segmentation, and intensity scaling
1_Trainings/        Training scripts and configs for diffusion & flow modules
2_Tests/            Inference utilities and shared runtime libraries
3_Evaluations/      Evaluation metrics and statistical analysis
Resource/           Pipeline diagrams and result figures
```

## Requirements
- Python 3.8 or later  
- PyTorch 2.0+ with CUDA support  
- NumPy, SciPy, scikit-learn, scikit-image  
- imageio, imageio-ffmpeg, pyspng, pillow, nibabel, click, requests, tqdm, psutil  
- See `requirements.txt` for exact versions  


## Figures
![Pipeline overview](Resource/Picture3.png)  
*Figure 1. Overview of the cascaded diffusion and flow modules.*

![BMI distribution](Resource/Picture1.png)  
*Figure 2. Dataset and generated cohort BMI distributions.*

![Quantitative results](Resource/Picture2.png)  
*Figure 3. Performance comparison across model variants.*

## Citation
```bibtex
@article{yoon2025cascaded,
  title={Cascaded 3D Diffusion Models for Whole-body 3D 18-F FDG PET/CT synthesis from Demographics},
  author={Yoon, Siyeop and Song, Sifan and Jin, Pengfei and Tivnan, Matthew and Oh, Yujin and Kim, Sekeun and Wu, Dufan and Li, Xiang and Li, Quanzheng},
  journal={arXiv preprint arXiv:2505.22489},
  year={2025}
}
```

## Contact
- yooneige@gmail.com  
- syoon5@mgh.harvard.edu  

## License
See the [LICENSE] files for under Training folder.

