
import totalsegmentator
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
import numpy as np


import torch
import os

import nibabel
import platform


if __name__ == "__main__":
    # option 1: provide input and output as file paths
    meta_path = "/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/MetaData"
    data_path = "/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Resized_PETCTSUV-2mm"
    segmentation_path = "/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Resized_PETCTSUV-2mm_mask"
    os.makedirs(segmentation_path,exist_ok=True)
    
    patient_ids = os.listdir(data_path)

    grouped_files = {}

    for f in patient_ids:
        if "."==f[0]:
            continue
        file_path = os.path.join(data_path, f)
        item_ids = os.listdir(file_path)
        for f_item in item_ids:

            if "."==f_item[0]:
                continue

            # Split the filename to extract ID and modality
            # Filename format: PETCT_<id>-<modality>.nii.gz
            # Example: PETCT_9837205b34-PET.nii.gz
            base = f_item.split("PETCT_")[1] 
            
            uniq_id, modality_ext = base.split("_", 1)  # e.g. uniq_id="9837205b34", modality_ext="PET.nii.gz"
            modality = modality_ext.split(".")[0]       # e.g. modality_ext="PET.nii.gz" -> modality="PET"

            if uniq_id not in grouped_files:
                grouped_files[uniq_id] = {}

            #grouped_files[uniq_id][modality] = f_item
            grouped_files[uniq_id][modality] = os.path.join(file_path, f_item)
#                print (modality,os.path.join(file_path, f_item))

    # Convert to desired list of dictionaries
    expected_output = []
    for uniq_id, data in grouped_files.items():
        # Ensure both CT and PET are present before adding
        if "CT" in data and "SUV" in data:
            expected_output.append({"uniq_id":uniq_id,"CT": data["CT"], "SUV": data["SUV"]})
        else:
            print (f"{uniq_id} don't have pair" )

    


    meta_filenames = os.listdir(meta_path)
    #meta_filenames=sorted(meta_filenames)
    #meta_filenames=meta_filenames[int(0.8*len(meta_filenames)):]
    
    print (len(meta_filenames))
    for meta_filename in meta_filenames:
        if meta_filename[0]==".":
            continue
        
        for patient_id in expected_output:
            
            if patient_id['uniq_id'].lower()  not in meta_filename:
                continue 

            
            
            basename=os.path.basename(patient_id["CT"])
            output_path=os.path.join(segmentation_path,meta_filename[:-4]+".nii.gz")
            if os.path.exists(output_path):
                
                
                
                break
            print(output_path) 
            

            textfile= os.path.join(meta_path,meta_filename)
            
            key_value_dict = {}
            with open(textfile, "r") as file:
                for line in file:
                    # Split the line into key and value
                    k, v = line.strip().split(":")
                    # Add to the dictionary
                    key_value_dict[k] = float(v)

            
            CT_nifti=nibabel.load(patient_id["CT"])
            CT=CT_nifti.get_fdata()

            CT[CT<-1000]=-1000
            CT[CT>1000]=1000
            
            reoriented_nifti = nibabel.Nifti1Image(CT, CT_nifti.affine)
            output_img = totalsegmentator(reoriented_nifti,fast=True)
            nibabel.save(output_img, output_path)

