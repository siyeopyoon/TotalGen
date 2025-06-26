
import multiprocessing
import os
import nibabel as nib
import pydicom
import numpy as np

import matplotlib.pyplot as plt
import platform
import scipy.ndimage as ndimage


def find_largest_component(thresholded_volume):
    labeled_volume, num_features = ndimage.label(thresholded_volume)
    sizes = ndimage.sum(thresholded_volume, labeled_volume, range(1, num_features + 1))
    largest_component_label = np.argmax(sizes) + 1
    largest_component = labeled_volume == largest_component_label
    return largest_component

def process_patient(pack):
    
    input_folder,out_CT_folder, out_SUV_folder, out_mask_folder, pid=pack

    dirpath=os.path.join(input_folder,pid)
    
    filename=pid+"_CT.nii.gz"
    
    CT_path= os.path.join(dirpath, filename)    
    
    filename_SUV=pid+"_SUV.nii.gz"
    SUV_path= os.path.join(dirpath, filename_SUV)     
    
    filename_mask=pid+".nii.gz"  
    mask_path= os.path.join(mask_folder, filename_mask)
    
    
    
    CT_output_path = os.path.join(out_CT_folder, f"{pid}.nii.gz")
    SUV_output_path = os.path.join(out_SUV_folder, f"{pid}.nii.gz")
    mask_output_path = os.path.join(out_mask_folder, f"{pid}.nii.gz")
    
    if os.path.exists(CT_output_path):
        return
    
    
    
    CT=nib.load(CT_path)
    CT=CT.get_fdata()
    
    SUV=nib.load(SUV_path)
    SUV=SUV.get_fdata()
    
    mask=nib.load(mask_path)
    mask=mask.get_fdata()
    
    organdict={
        'heart':[51],
        'liver':[5],
        'pancreas':[7],
        'kidney':[2,3],
        'spleen':[1],
        'sacrum':[25],
        'claviculs':[73,74],
        'scapuls':[71,72],
        'hip':[77,78],
        
    }
    
    mask_organ=np.zeros_like(CT)
    for organ in organdict:
        for oid in organdict[organ]:
            
            mval=1            
            mask_organ=mask_organ+np.where(mask==oid,mval,0)
    
    mask_organ_bin=np.where(mask_organ>0,1,0)
    
    min_coords = np.min(np.where(mask_organ_bin), axis=1)
    max_coords = np.max(np.where(mask_organ_bin), axis=1)
    
    min_coords = tuple(map(int, min_coords))  # (z_min, y_min, x_min)
    max_coords = tuple(map(int, max_coords))  # (z_max, y_max, x_max)

    
    middle_loc= (min_coords[2]+max_coords[2] + 1)//2
    window=480
    
    zstart=min_coords[2]-20 if min_coords[2]-20 >0 else min_coords[2]
    zend=max_coords[2]+20 if max_coords[2]+20 <CT.shape[2] else max_coords[2]
    
    cropped_arr = CT[:,:, zstart:zend]
    cropped_mask = mask[:,:, zstart:zend]
    cropped_suv = SUV[:,:, zstart:zend]
    
    
    
    blob=np.where(cropped_arr < -999, 0, 1).astype(np.uint8)
    largest_component = find_largest_component(blob)
    compact_CT=ndimage.binary_fill_holes(largest_component).astype(np.uint8)

    min_coords = np.min(np.where(compact_CT), axis=1)
    max_coords = np.max(np.where(compact_CT), axis=1)
    
    min_coords = tuple(map(int, min_coords))  # (z_min, y_min, x_min)
    max_coords = tuple(map(int, max_coords))  # (z_max, y_max, x_max)

    compact_CT = cropped_arr[
        min_coords[0]:max_coords[0] + 1,
        min_coords[1]:max_coords[1] + 1,
        min_coords[2]:max_coords[2] + 1,
    ]
    compact_SUV = cropped_suv[
        min_coords[0]:max_coords[0] + 1,
        min_coords[1]:max_coords[1] + 1,
        min_coords[2]:max_coords[2] + 1,
    ]
    compact_mask = cropped_mask[
        min_coords[0]:max_coords[0] + 1,
        min_coords[1]:max_coords[1] + 1,
        min_coords[2]:max_coords[2] + 1,
    ]
    
    
    z=cropped_arr.shape[-1]
    targetShape=[224,224,z]
    expected_CT= -1000 * np.ones(shape=targetShape)
    expected_SUV= np.zeros(shape=targetShape)
    expected_mask=np.zeros(shape=targetShape)
    
    diff = np.array(targetShape)-compact_CT.shape
    wing = (diff//2).astype(np.int16).tolist()
    
    #put the compact_CT into center of expected_CT
        
    x0,y0,z0=wing
    if x0<0:
        nx0=-x0
        compact_CT=compact_CT[nx0:nx0+targetShape[0],:,:]
        compact_SUV=compact_SUV[nx0:nx0+targetShape[0],:,:]  
        compact_mask=compact_mask[nx0:nx0+targetShape[0],:,:]
        x0=0
        
        diff = np.array(targetShape)-compact_CT.shape
        wing = (diff//2).astype(np.int16).tolist()
        x0,y0,z0=wing
        
        print (nx0,compact_CT.shape)
    if y0<0:
        ny0=-y0
        compact_CT=compact_CT[:,ny0:ny0+targetShape[1],:]
        compact_SUV=compact_SUV[:,ny0:ny0+targetShape[1],:]
        compact_mask=compact_mask[:,ny0:ny0+targetShape[1],:]
        y0=0
        diff = np.array(targetShape)-compact_CT.shape
        wing = (diff//2).astype(np.int16).tolist()
        x0,y0,z0=wing
    
    print (compact_CT.shape)
    expected_CT[
        x0 : x0 + compact_CT.shape[0],
        y0 : y0 + compact_CT.shape[1],
        z0 : z0 + compact_CT.shape[2]
    ] = compact_CT
    
    expected_SUV[
        x0 : x0 + compact_CT.shape[0],
        y0 : y0 + compact_CT.shape[1],
        z0 : z0 + compact_CT.shape[2]
    ] = compact_SUV
    
    
    
    expected_mask[
        x0 : x0 + compact_CT.shape[0],
        y0 : y0 + compact_CT.shape[1],
        z0 : z0 + compact_CT.shape[2]
    ] = compact_mask
    


    #masked=CT*mask_organ
    
        
    
    affine=2.0*np.eye(4,4)
    affine[3,3]=1.0
    
    
    expected_CT=np.flip(expected_CT,axis=[0,1])
    expected_mask=np.flip(expected_mask,axis=[0,1])

    expected_SUV=np.flip(expected_SUV,axis=[0,1])
    
    expected_CT=expected_CT.astype(np.int16)
    expected_CT = nib.Nifti1Image(expected_CT, affine)
   
    nib.save(expected_CT, CT_output_path)
    
 
    expected_SUV = nib.Nifti1Image(expected_SUV, affine)
   
    nib.save(expected_SUV, SUV_output_path)
    
    
    expected_mask = nib.Nifti1Image(expected_mask, affine)

    nib.save(expected_mask, mask_output_path)
    
    
    
    

if __name__ == "__main__":
    
    
        
    input_folder=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Resized_PETCTSUV-2mm/"
    meta_folder=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/MetaData/"
    mask_folder=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Resized_PETCTSUV-2mm_mask/"
    

    out_mask_folder=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Cropped_PETCTSUV-2mm/mask/"
    os.makedirs(out_mask_folder,exist_ok=True)
    
    out_CT_folder=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Cropped_PETCTSUV-2mm/CT/"
    os.makedirs(out_CT_folder,exist_ok=True)

    out_SUV_folder=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Cropped_PETCTSUV-2mm/SUV/"
    os.makedirs(out_SUV_folder,exist_ok=True)

    
    patient_ids = os.listdir(input_folder)
    
    
    
    
    packs=[]
    for pid in patient_ids:
        if pid[0]=='.':
            continue

        packs.append([input_folder,out_CT_folder, out_SUV_folder, out_mask_folder, pid])
        process_patient([input_folder,out_CT_folder, out_SUV_folder, out_mask_folder, pid])
        
    #total_cpu=multiprocessing.cpu_count()
    #runcpu= total_cpu//2 if total_cpu-6 < total_cpu//2 else total_cpu-6
    #print (runcpu)

  
    #with multiprocessing.Pool(processes=runcpu) as pool:
    #    pool.map(process_patient, packs)