# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import nibabel
import scipy.ndimage as ndimage

import random

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        use_labels  = True,     # Enable conditioning labels? False = label dimension is zero.
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:]
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self): # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = anything goes.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------





class Dataset_train_low_PETCT(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 use_labels=False,dry=False):


        meta_path = os.path.join(path,"meta")
        meta_filenames= os.listdir(meta_path)
   
   
        ct_path = os.path.join(path,"CT")
        suv_path = os.path.join(path,"SUV")
        
        self.dataset=[]
        self.num_channels=2 # CT,PET
        
        maxz=384
        
        meta_filename_true=[]
        for meta_filename in meta_filenames:
            if meta_filename[0]==".":
                continue
            meta_filename_true.append(meta_filename)
        
        if dry:
            meta_filename_true=random.sample(meta_filename_true,1) 
        else:
            meta_filename_true=random.sample(meta_filename_true,10) 
        for meta_filename in meta_filename_true:
            if meta_filename[0]==".":
                continue
            
            textfile= os.path.join(meta_path,meta_filename)
            
            key_value_dict = {}
            with open(textfile, "r") as file:
                for line in file:
                    # Split the line into key and value
                    k, v = line.strip().split(":")
                    # Add to the dictionary
                    key_value_dict[k] = float(v)
 
            niftiname=meta_filename[:-4]+".nii.gz"
            
            CT=nibabel.load(os.path.join(ct_path,niftiname))
            CT=CT.get_fdata()


            PET=nibabel.load(os.path.join(suv_path,niftiname))
            PET=PET.get_fdata()

                
            CT = np.flip(CT, axis=2)
            PET = np.flip(PET, axis=2)
            
            
            
                        
            scaler_z= maxz/float(CT.shape[-1])          
            CT=ndimage.zoom(CT,zoom=(1,1,scaler_z),cval=-1024.0)
            
            PET=ndimage.zoom(PET,zoom=(1,1,scaler_z),cval=0.0)
            if CT.shape[2]>maxz:
                CT=CT[:,:,:maxz]
                PET=PET[:,:,:maxz]
            elif CT.shape[2]<maxz:
                minpad_CT= np.min(CT) * np.ones(shape=(CT.shape[0],CT.shape[1],maxz))
                minpad_PET= np.min(PET) * np.ones(shape=(PET.shape[0],PET.shape[1],maxz))
                
                minpad_CT[:,:,:CT.shape[2]]= CT
                minpad_PET[:,:,:PET.shape[2]]= PET 
                
                
                CT=minpad_CT
                PET=minpad_PET
                    
            PET=np.clip(PET,0,25)  # 0-25     
            PET=PET+1.0 #1-26
            
            PET=np.log2(PET) # 0-4.7004...
            PET=PET/np.log2(26.0) #0-1.0
            PET=PET.astype(np.float32)    
            
                
            CT[CT<-500]=-500 
            CT[CT>500]=500 # -500~500
            CT=CT+500.0 # 0~1000
            CT=CT/1000.0 # 0~1.0
            CT=CT.astype(np.float32)


            for x_off in range(4):
                for y_off in range(4):
                    for z_off in range(4):
                        CT_sample = CT[x_off::4, y_off::4, z_off::4]
                        PET_sample = PET[x_off::4, y_off::4, z_off::4]
                        
            
                
                        CT_sample=np.clip(CT_sample,0,1.0)
                        PET_sample=np.clip(PET_sample,0,1.0)
                        
                        new_key_value_dict = {
                                "CT": CT_sample.copy(),
                                "PET": PET_sample.copy()
                            }
                        for key_value in key_value_dict:
                            new_key_value_dict[key_value]=key_value_dict[key_value]
                            

                        self.dataset.append(new_key_value_dict)
        
        self.ones= np.ones(shape=(56,56,96))

    

    def normalizer_minmax(self, data, minval, maxval):
        return (data - minval) / (maxval - minval)

    def patchfy(self, data):


        height = self.ones * (data["height"]-1.0)
        weight = self.ones * (data["weight"]-30)/100.0
        sex = self.ones* data["sex"]
        age = self.ones * data["age"]/100.0

        demographics=np.stack ([height,weight,sex,age],axis=0)
        target=np.stack ([data["CT"],data["PET"]],axis=0)
        
        label=0
        return target,demographics, label

    def __getitem__(self, index: int):

        return self.patchfy(self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)


class Dataset_train_high_PETCT(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 use_labels=False,
                 dry=False):


        meta_path = os.path.join(path,"MetaData")
        meta_filenames= os.listdir(meta_path)
   
   
        ct_path = os.path.join(path,"Cropped_PETCTSUV-2mm/CT")
        suv_path = os.path.join(path,"Cropped_PETCTSUV-2mm/SUV")
        
        self.dataset=[]
        self.num_channels=2 # CT,PET
        
        maxz=384
                
        self.ones= np.ones(shape=(56,56,96))
        
        meta_filename_true=[]
        for meta_filename in meta_filenames:
            if meta_filename[0]==".":
                continue
            meta_filename_true.append(meta_filename)
            
        if dry:
            meta_filename_true=random.sample(meta_filename_true,1) 
        else:
            meta_filename_true=random.sample(meta_filename_true,min(20,len(meta_filename_true))) 
            
        for meta_filename in meta_filename_true:
            if meta_filename[0]==".":
                continue
            
            textfile= os.path.join(meta_path,meta_filename)
            
            key_value_dict = {}
            with open(textfile, "r") as file:
                for line in file:
                    # Split the line into key and value
                    k, v = line.strip().split(":")
                    # Add to the dictionary
                    key_value_dict[k] = float(v)
 
            niftiname=meta_filename[:-4]+".nii.gz"
            
            CT=nibabel.load(os.path.join(ct_path,niftiname))
            CT=CT.get_fdata()


            PET=nibabel.load(os.path.join(suv_path,niftiname))
            PET=PET.get_fdata()

                
            CT = np.flip(CT, axis=2)
            PET = np.flip(PET, axis=2)
            
            
            
                        
            scaler_z= maxz/float(CT.shape[-1])          
            CT=ndimage.zoom(CT,zoom=(1,1,scaler_z),cval=-1024.0)
            
            PET=ndimage.zoom(PET,zoom=(1,1,scaler_z),cval=0.0)
            if CT.shape[2]>maxz:
                CT=CT[:,:,:maxz]
                PET=PET[:,:,:maxz]
            elif CT.shape[2]<maxz:
                minpad_CT= np.min(CT) * np.ones(shape=(CT.shape[0],CT.shape[1],maxz))
                minpad_PET= np.min(PET) * np.ones(shape=(PET.shape[0],PET.shape[1],maxz))
                
                minpad_CT[:,:,:CT.shape[2]]= CT
                minpad_PET[:,:,:PET.shape[2]]= PET 
                
                
                CT=minpad_CT
                PET=minpad_PET
                    
            PET=np.clip(PET,0,25)  # 0-25     
            PET=PET+1.0 #1-26
            
            PET=np.log2(PET) # 0-4.7004...
            PET=PET/np.log2(26.0) #0-1.0
            PET=PET.astype(np.float32)    
            
                
            CT[CT<-500]=-500 
            CT[CT>500]=500 # -500~500
            CT=CT+500.0 # 0~1000
            CT=CT/1000.0 # 0~1.0
            CT=CT.astype(np.float32)


            CT_sample = ndimage.zoom(CT,zoom=(1/4,1/4,1/4),order=1,cval=0.0,)
            PET_sample = ndimage.zoom(PET,zoom=(1/4,1/4,1/4),order=1,cval=0.0)
            
            CT_sample=ndimage.zoom(CT_sample,zoom=(4,4,4),order=1,cval=0.0)
            PET_sample=ndimage.zoom(PET_sample,zoom=(4,4,4),order=1,cval=0.0)
        
    
        
            
            CT_sample=np.clip(CT_sample,0,1.0)
            PET_sample=np.clip(PET_sample,0,1.0)
            
            new_key_value_dict = {
                    "CT": CT_sample.copy(),
                    "PET": PET_sample.copy(),
                    "CT_high": CT.copy(),
                    "PET_high": PET.copy()
                    
                }
            for key_value in key_value_dict:
                new_key_value_dict[key_value]=key_value_dict[key_value]
                        

            self.dataset.append(new_key_value_dict)

        print( f"datasize :{len(self.dataset)}\n")
    

    def normalizer_minmax(self, data, minval, maxval):
        return (data - minval) / (maxval - minval)

    def patchfy(self, data):


        x_len, y_len, z_len = data["CT"].shape
        patchsize =self.ones.shape

        start_idxs=[0,0,0]

        start_idxs[0]=np.random.randint(0,x_len-patchsize[0])
        start_idxs[1]=np.random.randint(0,y_len-patchsize[1])
        start_idxs[2]=np.random.randint(0,z_len-patchsize[2])
        
        x_range = np.linspace(start_idxs[0], start_idxs[0]+patchsize[0]-1, patchsize[0])/x_len
        y_range = np.linspace(start_idxs[1], start_idxs[1]+patchsize[1]-1, patchsize[1])/y_len
        z_range = np.linspace(start_idxs[2], start_idxs[2]+patchsize[2]-1, patchsize[2])/z_len

        pos_x, pos_y, pos_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        


        height = self.ones * (data["height"]-1.0)
        weight = self.ones * (data["weight"]-30)/100.0
        sex = self.ones* data["sex"]
        age = self.ones * data["age"]/100.0




        CT_patch=data["CT_high"][start_idxs[0]:start_idxs[0]+patchsize[0],
                        start_idxs[1]:start_idxs[1]+patchsize[1],
                        start_idxs[2]:start_idxs[2]+patchsize[2]]
        
        PET_patch=data["PET_high"][start_idxs[0]:start_idxs[0]+patchsize[0],
                    start_idxs[1]:start_idxs[1]+patchsize[1],
                    start_idxs[2]:start_idxs[2]+patchsize[2]] 
        
        CT_low_patch=data["CT"][start_idxs[0]:start_idxs[0]+patchsize[0],
                    start_idxs[1]:start_idxs[1]+patchsize[1],
                    start_idxs[2]:start_idxs[2]+patchsize[2]] 
 
        PET_low_patch=data["PET"][start_idxs[0]:start_idxs[0]+patchsize[0],
                    start_idxs[1]:start_idxs[1]+patchsize[1],
                    start_idxs[2]:start_idxs[2]+patchsize[2]] 




        lowres_demographics_pose=np.stack ([CT_low_patch,PET_low_patch,
                                            height,weight,sex,age,
                                            pos_x, pos_y, pos_z],axis=0)
        target=np.stack ([CT_patch, PET_patch],axis=0)
        
        label=0
        return target, lowres_demographics_pose,label
    def __getitem__(self, index: int):

        return self.patchfy(self.dataset[index])

    def __len__(self) -> int:
        return len(self.dataset)