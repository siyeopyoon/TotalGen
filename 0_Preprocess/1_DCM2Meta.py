import multiprocessing
import os
import nibabel as nib
import pydicom
import numpy as np


def saver(tuples):
    input_folder, output_folder,patient_id=tuples
    ct_output_path = os.path.join(output_folder, f"{patient_id}.txt")

    if os.path.exists(ct_output_path):
        print (ct_output_path,"skipped")
        return
    else:
        print(ct_output_path, "-processing")

    patient_folder_A = os.path.join(input_folder, patient_id)
    dump = os.listdir(patient_folder_A)
    
    check=False
    for dump_s in dump:
        
        if dump_s[0] == '.':
            continue
        
        if check:
            continue
        
        patient_folder=os.path.join(patient_folder_A, dump_s)

        modalities = os.listdir(patient_folder)


        for modality in modalities:
            
            if modality[0] == '.':
                continue
            
            if "CT" in modality.upper() or "GK" in modality.upper() or "THA" in modality.upper():
                modality_folder = os.path.join(patient_folder, modality)

                slices = os.listdir(modality_folder)
                slices=sorted(slices)

                medianslice=slices[len(slices)//2]
                dcm_path = os.path.join(modality_folder, medianslice)
                dicom_data = pydicom.dcmread(dcm_path,stop_before_pixels=True)



                if hasattr(dicom_data, 'PatientSize'):
                    print (f"PatientSize:{dicom_data.PatientSize}\n")
                    PatientSize=float(dicom_data.PatientSize)
                else:
                    print (f"{dump_s} missing PatientSize\n")
                    continue

                if hasattr(dicom_data, 'PatientWeight'):
                    print(f"PatientWeight:{dicom_data.PatientWeight}\n")
                    PatientWeight = float(dicom_data.PatientWeight)
                else:
                    print (f"{dump_s} missing PatientWeight\n")
                    continue

                if hasattr(dicom_data, 'PatientAge'):
                    print(f"PatientAge:{dicom_data.PatientAge[:-1]}\n")
                    PatientAge = float(dicom_data.PatientAge[:-1])
                else:
                    print (f"{dump_s} missing PatientAge\n")
                    continue


                if hasattr(dicom_data, 'PatientSex'):
                    print(f"PatientAge:{dicom_data.PatientSex}\n")
                    PatientSex = 0 if dicom_data.PatientSex == "F" else 1
                else:
                    print (f"{dump_s} missing PatientSex\n")
                    continue


                with open(ct_output_path, "w") as file:
                    file.write(f"height:{PatientSize}\n")
                    file.write(f"weight:{PatientWeight}\n")
                    file.write(f"age:{PatientAge}\n")
                    file.write(f"sex:{PatientSex}\n")
                check=True

   
   


def saver_from_dicom(args):


    dimcompath,output_folder,patient_id=args
    dicom_data = pydicom.dcmread(dimcompath,stop_before_pixels=True)
    meta_output_path = os.path.join(output_folder, f"{patient_id}.txt")


    if hasattr(dicom_data, 'PatientSize'):
        print (f"PatientSize:{dicom_data.PatientSize}\n")
        PatientSize=float(dicom_data.PatientSize)
    else:
        print (f"{dimcompath} missing PatientSize\n")
        PatientSize=-1

    if hasattr(dicom_data, 'PatientWeight'):
        print(f"PatientWeight:{dicom_data.PatientWeight}\n")
        PatientWeight = float(dicom_data.PatientWeight)
    else:
        print (f"{dimcompath} missing PatientWeight\n")
        PatientWeight=-1

    if hasattr(dicom_data, 'PatientAge'):
        print(f"PatientAge:{dicom_data.PatientAge[:-1]}\n")
        PatientAge = float(dicom_data.PatientAge[:-1])
    else:
        print (f"{dimcompath} missing PatientAge\n")
        PatientAge=-1


    if hasattr(dicom_data, 'PatientSex'):
        print(f"PatientAge:{dicom_data.PatientSex}\n")
        PatientSex = 0 if dicom_data.PatientSex == "F" else 1
    else:
        print (f"{dimcompath} missing PatientSex\n")
        PatientSex=-1


    with open(meta_output_path, "w") as file:
        file.write(f"height:{PatientSize}\n")
        file.write(f"weight:{PatientWeight}\n")
        file.write(f"age:{PatientAge}\n")
        file.write(f"sex:{PatientSex}\n")
    

     
def convert_dicom_to_nifti(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    patient_ids = [name for name in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, name))]

    packs=[]
    for pids in patient_ids:
        packs.append([input_folder, output_folder,pids])

        saver([input_folder, output_folder,pids])

if __name__ == "__main__":
    input_folder="/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/FDG-PET-CT-Lesions/"
    output_folder="/Volumes/HOMEDIR$/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/MetaData/"
    convert_dicom_to_nifti(input_folder=input_folder,output_folder=output_folder)