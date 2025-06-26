import numpy as np
import os

import nibabel
import platform

import torch
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
            
        data_path=f"{external}/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Generated_Data/{mode}"
        segmentation_path =f"{external}/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Generated_Data/{mode}-segment/"
        
        path_out=f"{external}/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Generated_Data/recal/"
        os.makedirs(path_out,exist_ok=True)
        
        outname=f"{mode}.txt"
        seg_ids = os.listdir(segmentation_path)

        
        
        patient_ids = os.listdir(data_path)
        uniq_ids = []

        for f in seg_ids:
            if "."==f[0]:
                continue
            
            
            basename=os.path.basename(f)
            
            # Split the filename to extract ID and modality
            # Filename format: PETCT_<id>-<modality>.nii.gz
            # Example: PETCT_9837205b34-PET.nii.gz

            
            _, uniq_id = basename.split("images_CT_")  # e.g. uniq_id="9837205b34", modality_ext="PET.nii.gz"
            uniq_id, _ = uniq_id.split(".nii.gz")       # e.g. modality_ext="PET.nii.gz" -> modality="PET"

            uniq_ids.append(uniq_id)

        
        for patient_id in uniq_ids:
        
                
                
            
            height,weight,sex,age=patient_id.split("_")

            age=float(age)
            group= 'Elder' if age>65 else 'Adult'
            if '1' in sex:
                sex= "Male"
            else:
                sex="Female"
                
            mask_path=os.path.join(segmentation_path,"images_CT_"+patient_id+".nii.gz")
            SUV_path=os.path.join(data_path,"images_PET_"+patient_id+".nii.gz")
            
            
            SUV_nifti=nibabel.load(SUV_path)
            Mask_nifti=nibabel.load(mask_path)
            

            SUV=SUV_nifti.get_fdata()      
            SUV[SUV<0.0]=0.0
            SUV[SUV>25.0]=25.0 
            SUV=np.flip(SUV,axis=-1)
            mask=Mask_nifti.get_fdata()

            
            
            organdict={
                'liver':[5],
                'heart':[51],
                'kidney':[2,3],
                'spleen':[1]
            }
            
            
            for organ in organdict:
                
                mask_organ=np.zeros_like(SUV)
                
                for oid in organdict[organ]:
                    mask_organ=mask_organ+np.where(mask==oid,1,0)
                
                masked=SUV*mask_organ
                
                mean_SUV=np.mean(SUV,where=mask_organ.astype(bool))
                peak_SUV=np.max(masked)
                
                
                voxel_volume = 8.0  # in mm³
                num_voxels = np.sum(mask_organ)
                volume = num_voxels * voxel_volume  # in mm³
                volume=volume/1000.0
            
                
                print (f"{age}, {group}, {sex}, {weight}, {height}, {organ}, {volume}, {mean_SUV}, {peak_SUV}")
                
                    
                log_path = os.path.join(path_out, outname)

                log=open(log_path, "a")
                log.write(f"{age}, {group}, {sex}, {weight}, {height}, {organ}, {volume}, {mean_SUV}, {peak_SUV}\n")
                log.close()
