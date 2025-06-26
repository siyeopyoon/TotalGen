
import totalsegmentator
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import numpy as np


import torch
import os

import nibabel
import platform


if __name__ == "__main__":
        
    if torch.cuda.is_available():
        use_gpu=True
        external="/external"
    else:
        use_gpu=False
        external="/Volumes/HOMEDIR$"
    
    # inference dataset
    modes=["cascade-Flow-3D-img224","cascade-EDM2-3D-img224"]
    
    
    for mode in modes:
            
        path = f"{external}/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Generated_Data/{mode}"
        outpath=f"{external}/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Generated_Data/{mode}-segment/"
        os.makedirs(outpath,exist_ok=True)
        patient_ids = os.listdir(path)

        for f in patient_ids:
            if "."==f[0]:
                continue
            
            
            basename=os.path.basename(f)
            
            if "CT" in basename:
                
                inpath=os.path.join(path,basename)
                CT_nifti=nibabel.load(inpath)
                CT=CT_nifti.get_fdata()
                CT=np.flip(CT,axis=-1)
                CT[CT<-1000]=-1000
                CT[CT>1000]=1000
                
                reoriented_nifti = nibabel.Nifti1Image(CT, CT_nifti.affine)



                # 2. Run segmentation for specific organs only
                output_img = totalsegmentator(
                    reoriented_nifti,
                    fast=True,
                    roi_subset=["heart", "liver", "kidney_left", "kidney_right","spleen"]
                )
                output_path=os.path.join(outpath,basename)
                nibabel.save(output_img, output_path)
