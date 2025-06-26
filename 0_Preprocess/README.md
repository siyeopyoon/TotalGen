# Preprocess Pipeline

Scripts in this folder prepare the raw DICOM data before model training. The overall preprocessing workflow is visualized in [../Resource/Main1.pdf](../Resource/Main1.pdf).

## Workflow
1. **1_DCM2Meta.py** – export DICOM metadata to plain text for reproducibility.
2. **2_DCM2Nifti_PET-SUV.py** – convert PET scans to SUV NIfTI volumes.
3. **3_Totalsegment_population.py** – run TotalSegmentator across the entire cohort.
4. **4_Cropping_segs_SUV.py** – crop segmentations and SUV volumes to the working region.
5. **5_Compute_rescaler.py** / **5-1_compute_rescaler_using_existing.py** – compute intensity scaling factors from `ratio.txt`.

All scripts were verified locally. Main results are summarized in [../Resource/BMIs.pdf](../Resource/BMIs.pdf) and [../Resource/Graph.pdf](../Resource/Graph.pdf). For a complete description, consult our [arXiv paper](https://arxiv.org/pdf/2505.22489).
