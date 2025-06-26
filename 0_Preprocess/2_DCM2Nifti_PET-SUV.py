import os
import sys
import math
import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk
from datetime import datetime
from pydicom.datadict import keyword_for_tag
import platform
from scipy.ndimage import zoom
import time

import multiprocessing
from nilearn.image import resample_to_img
import shutil

import dicom2nifti




from scipy import ndimage
def sitk_to_nib(sitk_img):
    """Convert an in-memory SimpleITK image to a nibabel Nifti1Image."""
    arr = sitk.GetArrayFromImage(sitk_img)  # shape [Z, Y, X]
    # nibabel uses an affine for world coordinates, which we build from SITK metadata
    # However, building a 4x4 affine from SITK's origin/direction/spacing can be done:
    spacing = list(sitk_img.GetSpacing())
    origin  = list(sitk_img.GetOrigin())
    direction = np.array(sitk_img.GetDirection()).reshape((3, 3))

    # Build 4x4 affine
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3,  3] = origin

    # nibabel’s default orientation expects [X, Y, Z] in the data array, 
    # but SITK is [Z, Y, X]. We’ll need to flip or reorder axes if you want
    # them to match typical nibabel orientation. For simplicity, do a transpose:
    arr_nib = np.flip(arr, axis=0)  # or reorder with np.transpose(...)

    # The orientation specifics can get tricky; adapt as needed.
    return nib.Nifti1Image(arr_nib, affine)
def nib_to_sitk(nib_img):
    """
    Convert nibabel Nifti1Image back to SimpleITK.Image.

    Assumes that, in sitk_to_nib, we did:
       arr_nib = np.flip(sitk_array, axis=0)
    so here we flip back along axis=0 before building the SITK image.

    Then we parse the nib affine to derive:
       - origin
       - spacing
       - direction
    for the SITK image.
    """
    import numpy as np
    import SimpleITK as sitk

    # --- 1) Recover the data array and flip back ---
    arr_nib = nib_img.get_fdata()  # shape [X, Y, Z], etc.
    arr_sitk = np.flip(arr_nib, axis=0).astype(np.float32)  # Undo the flip from sitk_to_nib

    # Build the SimpleITK image from the NumPy array [Z, Y, X]
    # (SITK expects the first dimension to be Z, so shape is reversed from nib's default).
    out_sitk = sitk.GetImageFromArray(arr_sitk)

    # --- 2) Parse the nibabel affine to get origin, spacing, direction ---
    affine = nib_img.affine  # 4x4
    # The top-left 3x3 is the rotation+scaling, the rightmost 3x1 is the translation
    R = affine[:3, :3]
    t = affine[:3, 3]

    # Spacing is the magnitude of each column vector (if there's no shear).
    # direction_matrix will be the normalized columns of R.
    spacing = [np.linalg.norm(R[:, 0]),
               np.linalg.norm(R[:, 1]),
               np.linalg.norm(R[:, 2])]

    # Handle the case that one column might have zero length, etc.
    # For safety, add a small epsilon when dividing.
    eps = 1e-12
    direction_matrix = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        length = max(np.linalg.norm(R[:, i]), eps)
        direction_matrix[:, i] = R[:, i] / length

    # Convert direction to a flat tuple in row-major order, as SITK expects
    direction_flat = tuple(direction_matrix.ravel(order='F'))  # or 'C' – see note below

    # --- 3) Set the metadata on out_sitk ---
    out_sitk.SetSpacing(spacing)
    out_sitk.SetOrigin(tuple(t.tolist()))
    # SITK direction is stored in row-major, but nib's columns often correspond to axes:
    # If you see issues, you may need to transpose the direction matrix or reorder axes.
    out_sitk.SetDirection(direction_flat)

    return out_sitk

################################################################################
#                            UTILITY FUNCTIONS
################################################################################

def dicom_lookup(group, element):
    """
    Returns the DICOM keyword for a given (group, element).
    E.g., dicom_lookup(0x0028, 0x0051) -> 'CorrectedImage'
    """
    tag = (group, element)
    keyword = keyword_for_tag(tag)
    if keyword is None:
        return 'Unknown'
    return keyword

def time_sub(time1, time2):
    """
    Calculates the difference in seconds between two DICOM date-time strings.
      time1, time2: 'YYYYMMDDHHMMSS.FFFFFF'
    """
    try:
        fmt = '%Y%m%d%H%M%S'
        dt1 = datetime.strptime(time1.split('.')[0], fmt)
        dt2 = datetime.strptime(time2.split('.')[0], fmt)
        return (dt2 - dt1).total_seconds()
    except Exception as e:
        print(f"Error parsing date/time: {e}")
        sys.exit(1)

def conv_time(time_str):
    # function for time conversion in DICOM tag
    return (float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13]))

def calculate_suv_factor(ds):
    # reads a PET dicom file and calculates the SUV conversion factor

    total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    acq_time = ds.AcquisitionTime
    weight = ds.PatientWeight
    # decay correct the acquisition to the actual series start time
    # for backward compatibility and consistency to the challenge data,
    # we keep the old equation, but future implementations should adjust this 
    #if ds.RadiopharmaceuticalInformationSequence[0].DecayCorrection == 'START':
    #   acq_time = ds.SeriesTime
    time_diff = conv_time(acq_time) - conv_time(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    suv_factor = 1000 * weight / act_dose
    return suv_factor


def calculate_suv(pet_array, dicom_info):
    suv_factor=calculate_suv_factor(dicom_info)
    suv_array = pet_array*suv_factor
    return suv_array



def load_dicom_series(dicom_dir):
    """
    Load a DICOM series from dicom_dir using SimpleITK.
    Returns (sitk.Image, list_of_filenames).
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in {dicom_dir}")
    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
    reader.SetFileNames(file_names)
    image = reader.Execute()
    return image, file_names

def resample_sitk_image(image, spacing=(2.0, 2.0, 2.0), interp=sitk.sitkLinear):
    """
    Resample a SimpleITK image to 'spacing' using specified interpolation.
    """
    orig_spacing = np.array(image.GetSpacing(), dtype=float)
    orig_size = np.array(image.GetSize(), dtype=int)
    
    # Maintain same physical size
    orig_phys_size = orig_spacing * orig_size
    new_spacing = np.array(spacing, dtype=float)
    new_size = np.ceil(orig_phys_size / new_spacing).astype(int)
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(interp)
    
    # Background value
    arr = sitk.GetArrayFromImage(image)
    background_val = float(arr.min())
    resampler.SetDefaultPixelValue(background_val)
    
    return resampler.Execute(image)

def crop_volume_around_center_nophysical(image, output_size=(256,256,256)):
    """
    Crop 'image' to 'output_size' around its physical center.
    """
    size = np.array(image.GetSize(), dtype=int)
    crop_size = np.array(output_size, dtype=int)

    center_idx = (size - 1) / 2.0
    half_crop_size = (crop_size - 1) / 2.0
    start_idx = np.floor(center_idx - half_crop_size).astype(int)

    # Clamp to valid range
    start_idx = np.maximum(start_idx, 0)
    end_idx = start_idx + crop_size
    end_idx = np.minimum(end_idx, size)
    final_crop_size = end_idx - start_idx

    extract_filter = sitk.ExtractImageFilter()
    extract_filter.SetSize(final_crop_size.tolist())
    extract_filter.SetIndex(start_idx.tolist())
    cropped_image = extract_filter.Execute(image)

    return cropped_image

def crop_sitk_image_with_bounding_box(image_sitk, min_coords, max_coords):
    """
    Crop 'image_sitk' to the bounding box specified by 
    'min_coords' and 'max_coords' (both in index space).
    
    Parameters
    ----------
    image_sitk : sitk.Image
        3D SimpleITK image to be cropped.
    min_coords : tuple or list of int
        (z_min, y_min, x_min).
    max_coords : tuple or list of int
        (z_max, y_max, x_max).

    Returns
    -------
    cropped_sitk : sitk.Image
        Cropped SimpleITK image corresponding to the bounding box.
    """
    min_coords = tuple(map(int, min_coords))  # (z_min, y_min, x_min)
    max_coords = tuple(map(int, max_coords))  # (z_max, y_max, x_max)

    
    # Use NumPy-like slicing to create the cropped array
    arr = sitk.GetArrayFromImage(image_sitk)  # shape [z, y, x]
    cropped_arr = arr[
        min_coords[0]:max_coords[0] + 1,
        min_coords[1]:max_coords[1] + 1,
        min_coords[2]:max_coords[2] + 1,
    ]

    # 3) Convert back to SimpleITK
    cropped_sitk = sitk.GetImageFromArray(cropped_arr)
    # Optionally copy metadata (e.g. direction, origin) from the original,
    # but realize that the "physical location" changes, so you may want
    # to update the origin accordingly if you want the cropped region
    # to remain aligned in world coordinates.

    # For a quick approach, keep the same direction and spacing,
    # but shift the origin so the cropped region is in the correct place.
    origin = image_sitk.TransformIndexToPhysicalPoint((min_coords[2], min_coords[1], min_coords[0]))
    cropped_sitk.SetOrigin(origin)
    cropped_sitk.SetSpacing(image_sitk.GetSpacing())
    cropped_sitk.SetDirection(image_sitk.GetDirection())

    return cropped_sitk
def apply_mask_sitk(image_sitk, mask_sitk, outside_val=0):
    """
    Apply a binary mask (0/1) to 'image_sitk'. Outside mask -> 'outside_val'.
    """
    image_arr = sitk.GetArrayFromImage(image_sitk)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    mask_bin = (mask_arr > 0).astype(image_arr.dtype)
    
    masked_arr = image_arr * mask_bin + outside_val * (1 - mask_bin)
    out_img = sitk.GetImageFromArray(masked_arr)
    out_img.CopyInformation(image_sitk)
    return out_img

# (2) Find the largest 3D component from the result of (1)
def find_largest_component(thresholded_volume):
    labeled_volume, num_features = ndimage.label(thresholded_volume)
    sizes = ndimage.sum(thresholded_volume, labeled_volume, range(1, num_features + 1))
    largest_component_label = np.argmax(sizes) + 1
    largest_component = labeled_volume == largest_component_label
    return largest_component

# (3) Find the 3D bounding box of the largest 3D component from the result of (2)
def find_bounding_box(largest_component):
    min_coords = np.min(np.where(largest_component), axis=1)
    max_coords = np.max(np.where(largest_component), axis=1)
    return min_coords, max_coords


################################################################################
#                   MAIN PATIENT PROCESSING (CROP BEFORE MASK)
################################################################################

def process_patient(pack):
    
    input_folder, output_folder_original, output_folder_2mm, patient_id= pack
    """
    1. Load CT and PET from DICOM.
    2. Compute SUV from PET if possible.
    3. Resample CT, PET, (and SUV) to 2.0 mm.
    4. Crop CT, PET, (and SUV) around the center to (256,256,256).
    5. Create mask from *cropped CT* (HU==-1024 ->0, else 1).
    6. Apply mask to CT, PET, SUV.
    7. Save results.
    """
    patient_path = os.path.join(input_folder, patient_id)
    if not os.path.isdir(patient_path):
        print(f"[{patient_id}] Not a directory. Skipping.")
        return

    subfolders = [f for f in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, f))]
    if not subfolders:
        print(f"[{patient_id}] No subfolders found.")
        return
    dump_path = os.path.join(patient_path, subfolders[0])

    ct_path, pet_path = None, None
    modalities = [m for m in os.listdir(dump_path) if os.path.isdir(os.path.join(dump_path, m))]
    for mod in modalities:
        if "PET" in mod.upper():
            pet_path = os.path.join(dump_path, mod)
        elif "CT" in mod.upper() or "GK" in mod.upper():
            ct_path = os.path.join(dump_path, mod)
    
    if not pet_path:
        print(f"[{patient_id}] No PET folder found.")
        return
    if not ct_path:
        print(f"[{patient_id}] No CT folder found.")
        return


    out_dir = os.path.join(output_folder_original, patient_id)
    os.makedirs(out_dir, exist_ok=True)
    
    ct_out = os.path.join(out_dir, f"{patient_id}_CT.nii.gz")
    pet_out = os.path.join(out_dir, f"{patient_id}_PET.nii.gz")
    suv_out = os.path.join(out_dir, f"{patient_id}_SUV.nii.gz")
    
    if os.path.exists(ct_out):
        ct_nib  = nib.load(ct_out)
    else:
            # ------------------ Load CT ------------------ #
        try:
            ct_image, ct_fnames = load_dicom_series(ct_path)
        except Exception as e:
            print(f"[{patient_id}] Could not load CT: {e}")
            return
        sitk.WriteImage(ct_image, ct_out)
        ct_nib  = nib.load(ct_out)
        
    if os.path.exists(pet_out) and os.path.exists(suv_out):
        pet_nib = nib.load(pet_out)
        pet_image = sitk.ReadImage(pet_out)
        suv_image = sitk.ReadImage(suv_out)

        
    else:
        
        # ------------------ Load PET ------------------ #
        try:
            pet_image, pet_fnames = load_dicom_series(pet_path)
        except Exception as e:
            print(f"[{patient_id}] Could not load PET: {e}")
            return
        sitk.WriteImage(pet_image, pet_out)
        pet_nib = nib.load(pet_out)
        pet_image = sitk.ReadImage(pet_out)
        
        # For SUV calculation
        try:
            first_pet_dcm = pydicom.dcmread(pet_fnames[0], stop_before_pixels=True)
        except Exception as e:
            print(f"[{patient_id}] Could not read PET metadata for SUV: {e}")
            return
    
        pet_arr = sitk.GetArrayFromImage(pet_image)  # [Z, Y, X]
        pet_arr=pet_arr.astype(np.float32)
        
        suv_arr = calculate_suv(pet_arr, first_pet_dcm)
        suv_arr=suv_arr.astype(np.float32)
        
        # Convert SUV array -> SITK image
        suv_image = None
        if suv_arr is not None:
            suv_image = sitk.GetImageFromArray(suv_arr)
            suv_image.CopyInformation(pet_image)
            print ('suv_image is computed')

        if suv_image is not None:

            sitk.WriteImage(suv_image, suv_out)
            suv_image = sitk.ReadImage(suv_out)



            



    

    ct_mapped = os.path.join(out_dir, f"{patient_id}_CT-mapped.nii.gz")
    ct_cropped  = resample_to_img(source_img=ct_nib, target_img=pet_nib, interpolation='continuous',fill_value=-1024, copy_header=True)
    nib.save(ct_cropped,ct_mapped)
    
    del ct_nib, pet_nib 
    ct_image = sitk.ReadImage(ct_mapped) 
   
        
    # ------------------ Resample CT, PET, SUV to 2mm ------------------ #
    new_spacing = (2.0, 2.0, 2.0)
    ct_resampled = resample_sitk_image(ct_image, spacing=new_spacing, interp=sitk.sitkLinear)
    pet_resampled = resample_sitk_image(pet_image, spacing=new_spacing, interp=sitk.sitkLinear)
    suv_resampled = None
    if suv_image is not None:
        suv_resampled = resample_sitk_image(suv_image, spacing=new_spacing, interp=sitk.sitkLinear)


   
        
     # ------------------ Create Mask from CROPPED CT ------------------ #
    # Convert cropped CT to array, build mask (0 if HU=-1024 else 1)
    ct_cropped_arr = sitk.GetArrayFromImage(ct_resampled)
    pet_cropped_arr = sitk.GetArrayFromImage(pet_resampled)
    
    mask_ct_arr = np.where(ct_cropped_arr < -1000, 0, 1).astype(np.uint8)
    mask_pet_arr = np.where(pet_cropped_arr < 20.0, 0, 1).astype(np.uint8) #counts
    
    mask_arr = mask_ct_arr*mask_pet_arr
    #mask_arr = np.ones_like(mask_ct_arr)
    mask_arr = np.where(mask_arr == 0, 0, 1).astype(np.uint8)
    

    largest_component = find_largest_component(mask_arr)
    
    mask_arr=ndimage.binary_fill_holes(largest_component).astype(np.uint8)
    
    
    mask_image = sitk.GetImageFromArray(mask_arr)
    mask_image.CopyInformation(ct_resampled)

    # ------------------ Apply Mask to CT, PET, SUV ------------------ #
    ct_final = apply_mask_sitk(ct_resampled, mask_image,outside_val=-1000)
    pet_final = apply_mask_sitk(pet_resampled, mask_image,outside_val=0.0)
    
    
    min_coords, max_coords = find_bounding_box(mask_arr)
    ct_final=crop_sitk_image_with_bounding_box(ct_final, min_coords, max_coords )
    pet_final=crop_sitk_image_with_bounding_box(pet_final, min_coords, max_coords )
    suv_final = None
    if suv_resampled is not None:
        suv_final = apply_mask_sitk(suv_resampled, mask_image,outside_val=0.0)
        suv_final=crop_sitk_image_with_bounding_box(suv_final, min_coords, max_coords )
   
        

    # ============================================================================
    #     (1) Ensure x,y,z are divisible by 16
    #         - pad x,y with min value
    #         - crop z from 0-th index
    # ============================================================================
    ct_final_arr = sitk.GetArrayFromImage(ct_final)
    pet_final_arr = sitk.GetArrayFromImage(pet_final)
    if suv_final is not None:
        suv_final_arr = sitk.GetArrayFromImage(suv_final)
    else:
        suv_final_arr = None

    def pad_crop_divisible(image_arr, min_val):
        """
        Given a 3D NumPy array [Z, Y, X], make each dimension divisible by 16:
          - For z: crop from 0-th index if needed (no padding).
          - For y,x: pad with `min_val` if needed (no cropping).
        Returns the adjusted 3D array.
        """
        z, y, x = image_arr.shape
        
        # 1) Crop z dimension (from index 0 upwards)
        new_z = (z // 32) * 32  # largest multiple of 16 that is <= z

        if new_z < z:
            zstart=z-new_z
            image_arr = image_arr[zstart:, :, :]
        
        # 2) Pad y dimension if needed
        z, y, x = image_arr.shape  # update shape after z-crop
        if y % 32 != 0:
            new_y = ((y // 32) + 1) * 32
            pad_y = new_y - y
            # Pad equally on "top" or "bottom"? 
            # For simplicity, here pad at the "end" only:
            image_arr = np.pad(
                image_arr,
                pad_width=((0, 0), (0, pad_y), (0, 0)),
                mode='constant',
                constant_values=min_val
            )
        
        # 3) Pad x dimension if needed
        z, y, x = image_arr.shape
        if x % 32 != 0:
            new_x = ((x // 32) + 1) * 32
            pad_x = new_x - x
            image_arr = np.pad(
                image_arr,
                pad_width=((0, 0), (0, 0), (0, pad_x)),
                mode='constant',
                constant_values=min_val
            )
        
        return image_arr

    # Apply pad/crop for CT
    ct_min = ct_final_arr.min()  # min HU, e.g., might be -1022 or lower
    ct_final_arr_16 = pad_crop_divisible(ct_final_arr, min_val=ct_min)

    # Apply pad/crop for PET
    pet_min = pet_final_arr.min()  # min PET activity, possibly 0.0
    pet_final_arr_16 = pad_crop_divisible(pet_final_arr, min_val=pet_min)
    
    # And for SUV if present
    if suv_final_arr is not None:
        suv_min = suv_final_arr.min()
        suv_final_arr_16 = pad_crop_divisible(suv_final_arr, min_val=suv_min)
    else:
        suv_final_arr_16 = None

    # ------------------ Save Results ------------------ #
    out_dir = os.path.join(output_folder_2mm, patient_id)
    os.makedirs(out_dir, exist_ok=True)

    ct_out = os.path.join(out_dir, f"{patient_id}_CT.nii.gz")
    pet_out = os.path.join(out_dir, f"{patient_id}_PET.nii.gz")
    
    
    ct_final_arr_16 = sitk.GetImageFromArray(ct_final_arr_16)
    # Optionally transfer direction, origin, spacing from the original 
    ct_final_arr_16.SetDirection(ct_final.GetDirection())
    ct_final_arr_16.SetOrigin(ct_final.GetOrigin())
    ct_final_arr_16.SetSpacing(ct_final.GetSpacing())
    
    pet_final_arr_16 = sitk.GetImageFromArray(pet_final_arr_16)
    # Optionally transfer direction, origin, spacing from the original 
    pet_final_arr_16.SetDirection(pet_final.GetDirection())
    pet_final_arr_16.SetOrigin(pet_final.GetOrigin())
    pet_final_arr_16.SetSpacing(pet_final.GetSpacing())
    
    sitk.WriteImage(ct_final_arr_16, ct_out)
    sitk.WriteImage(pet_final_arr_16, pet_out)

    if suv_final is not None:
        suv_out = os.path.join(out_dir, f"{patient_id}_SUV.nii.gz")
        suv_final_arr_16 = sitk.GetImageFromArray(suv_final_arr_16)
        # Optionally transfer direction, origin, spacing from the original 
        suv_final_arr_16.SetDirection(suv_final.GetDirection())
        suv_final_arr_16.SetOrigin(suv_final.GetOrigin())
        suv_final_arr_16.SetSpacing(suv_final.GetSpacing())
        sitk.WriteImage(suv_final_arr_16, suv_out)
        print(f"[{patient_id}] CT, PET, SUV saved.")
    else:
        print(f"[{patient_id}] CT, PET saved. (No SUV)")      



    

   

def convert_dicom_to_nifti(meta_folder,input_folder, output_folder_original, output_folder_2mm):
    """
    Iterates over each patient, processes CT+PET, saves final volumes.
    """
    if not os.path.exists(output_folder_original):
        os.makedirs(output_folder_original)
    if not os.path.exists(output_folder_2mm):
        os.makedirs(output_folder_2mm)
    patient_ids = [
        p for p in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, p)) and not p.startswith('.')
    ]
    
    meta_filenames = os.listdir(meta_folder)        
            
    #meta_filenames=['PETCT_2e44706eaf']
    
    packs=[]
    for pid in patient_ids:
        for meta_filename in meta_filenames:
            if pid in meta_filename:
                packs.append([input_folder,output_folder_original, output_folder_2mm, pid])
                process_patient([input_folder, output_folder_original, output_folder_2mm, pid])
    
    #total_cpu=multiprocessing.cpu_count()
    #runcpu= total_cpu//2 if total_cpu-6 < total_cpu//2 else total_cpu-6
    
    #with multiprocessing.Pool(processes=runcpu) as pool:
    #    pool.map(process_patient, packs)
    
    print (f"Processed {len(packs)} patients.")
if __name__ == "__main__":
    # Example usage
    
    input_folder=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/FDG-PET-CT-Lesions/"
    meta_folder=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/MetaData/"
    
    
    output_folder_orig=f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Original_PETCTSUV/"
    output_folder_2mm =f"/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/Resized_PETCTSUV-2mm/"
    
    
    
    convert_dicom_to_nifti(meta_folder,input_folder,output_folder_orig, output_folder_2mm)
    print("Finished processing all patients.")
    
