# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
import nibabel

from flow_matching.path import CondOTProbPath
from torchdiffeq import odeint
import scipy.ndimage as ndimage
warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention')

#----------------------------------------------------------------------------
# Configuration presets.



#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net, noise, conditions=None, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next
def edm_sampler_conditional(
    net, noise, conditions=None, outfolder=None, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def denoise(x, t,conditions):
        Dx = net(x, t,conditions, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t,conditions, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        denoised=denoise(x_hat, t_hat,conditions)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised=denoise(x_next, t_next,conditions)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

                    
    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Guiding network. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    
    if use_gpu:
        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        device= torch.device('cuda')
    else:
        device=torch.device('cpu')

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size
        
    if use_gpu:
        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')


    for batch_idx, indices in enumerate(rank_batches):
        r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
        r.seeds = [seeds[idx] for idx in indices]
        if len(r.seeds) > 0:
            
            
            # Pick noise and labels.
            rnd = StackedRandomGenerator(device, r.seeds)
            r.noise = rnd.randn([len(r.seeds), net.img_channels, 56, 56, 96], device=device)
            
            
            demos=[]
            for ridx in range (len(r.seeds)):
                data={}
                data["sex"]=1
                data["height"]=1.5
                data["weight"]=65.0
                data["age"]=60.0
                
                ones= np.ones(shape=(56,56,96))
                height = ones * (data["height"]-1.0)
                weight = ones * (data["weight"]-30.0)/100.0
                sex = ones* data["sex"]
                age = ones * data["age"]/100.0

                demographics=np.stack ([height,weight,sex,age],axis=0)
                demos.append(demographics)
            
            demos=np.stack(demos,axis=0)
            r.conditions = torch.from_numpy(demos).to(device)
            
            
            
            
            r.labels = 0
            if net.label_dim > 0:
                r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                if class_idx is not None:
                    r.labels[:, :] = 0
                    r.labels[:, class_idx] = 1



            # Generate images.
            latents = dnnlib.util.call_func_by_name(func_name=edm_sampler_conditional, net=net, noise=r.noise, outfolder=outdir,
                                                    conditions=r.conditions, labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)
            r.images = encoder.decode(latents)


            for ridx in range (len(r.seeds)):
                for cidx in range (net.img_channels):
                
                        
                    output = r.images[ridx, cidx, ...]
                    

                    # Save images.
                    output = output.cpu().numpy()
                    if cidx == 0:
                        output= np.clip(output,0,1)
                        output=output*1000.0
                        output= output-500.0
                        output=output.astype(np.int16)
                    
                    if cidx == 1:
                        output= np.clip(output,0,1)
                        output=output*np.log2(26.0)
                        output= np.power(output,2)
                        output= output-1.0
                        
                    
                    output=np.flip(output,axis=2)
                    
                    affine= 8.0 * np.eye(4)
                    affine[3,3]=1.0
                    new_image = nibabel.Nifti1Image(output, affine=affine)
                    nibabel.save(new_image, f'{outdir}images_{str(ridx)}_{str(cidx)}.nii.gz')
    

def generate_images_edm(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Guiding network. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    
    if use_gpu:
        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        device= torch.device('cuda')
    else:
        device=torch.device('cpu')

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size
        
    if use_gpu:
        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    
    
        meta_path = os.path.join(data_dir,"meta")
        meta_filenames= os.listdir(meta_path)
   
   
      
        dataset=[]
        
        meta_filename_true=[]
        for meta_filename in meta_filenames:
            if meta_filename[0]==".":
                continue
            meta_filename_true.append(meta_filename)
        
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
 
            
            dataset.append(key_value_dict)
        
        

    for idx in range (len(dataset)):
        data=dataset[idx]
        # Pick noise and labels.
      
        noise = torch.randn([1, net.img_channels, 56, 56, 96], device=device)
        ones= np.ones(shape=(56,56,96))
        height = ones * (data["height"]-1.0)
        weight = ones * (data["weight"]-30.0)/100.0
        sex = ones* data["sex"]
        age = ones * data["age"]/100.0

        demographics=np.stack ([height,weight,sex,age],axis=0)
        
        demographics=np.expand_dims(demographics,axis=0)
        
        conditions = torch.from_numpy(demographics).to(device)
    
        labels = 0
  
        # Generate images.
        latents = dnnlib.util.call_func_by_name(func_name=edm_sampler_conditional, net=net, noise=noise, outfolder=outdir,
                                                conditions=conditions, labels=labels, gnet=gnet, randn_like=torch.randn_like, **sampler_kwargs)
        images = encoder.decode(latents)


        
        output = images[0, ...]
        output = output.cpu().numpy()
        output= np.clip(output,0,1)
        
        
        affine= 8.0 * np.eye(4)
        affine[3,3]=1.0
        new_image = nibabel.Nifti1Image(output, affine=affine)
        nibabel.save(new_image, f'{outdir}images_{data["height"]}_{data["weight"]}_{data["sex"]}_{data["age"]}.nii.gz')
            

def flow_sampler_conditional(
    net, noise, conditions=None, outfolder=None, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    
    def ode_func(t, x):
        # Here, t and x are provided by the solver.
        return net(x=x, sigma=t, condition=conditions)
        
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    
    time_vec = (t_steps / (1 + t_steps)).squeeze()
    time_grid = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)

    
    
    ode_opts = {}
    atol= 1e-5,
    rtol = 1e-5,
    with torch.set_grad_enabled(False):
        # Approximate ODE solution with numerical ODE solver
        sol = odeint(
            ode_func,
            noise,
            time_grid,
            method="heun2",
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )
                    
    return sol[-1]


         


def flow_sampler_conditional_jump(
    net, noise, jumper=None, conditions=None, outfolder=None, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    
    def ode_func(t, x):
        # Here, t and x are provided by the solver.
        return net(x=x, sigma=t, condition=conditions)
        
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    
    time_vec = (t_steps / (1 + t_steps)).squeeze()
    time_grid = 1.0 - torch.clip(time_vec, min=0.0, max=1.0)
    
    path = CondOTProbPath()
    
    #50% jumping
    
    jumpt=num_steps//2
    noise = path.sample(t=time_grid[jumpt:jumpt+1], x_0=noise, x_1=jumper)
    noise = noise.x_t
    
    time_grid = time_grid [jumpt:]
    ode_opts = {}
    
    atol= 1e-5,
    rtol = 1e-5,
    with torch.set_grad_enabled(False):
        # Approximate ODE solution with numerical ODE solver
        sol = odeint(
            ode_func,
            noise,
            time_grid,
            method="heun2",
            options=ode_opts,
            atol=atol,
            rtol=rtol,
        )
                    
    return sol[-1]



            
def generate_cascade(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    srnet,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Guiding network. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    
    if use_gpu:
        # Rank 0 goes first.
        if dist.get_rank() != 0:
            torch.distributed.barrier()
        device= torch.device('cuda')
    else:
        device=torch.device('cpu')

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardEncoder')
    assert net is not None



     # Load main network.
    if isinstance(srnet, str):
        if verbose:
            dist.print0(f'Loading main network from {srnet} ...')
        with dnnlib.util.open_url(srnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        srnet = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardEncoder')
    assert srnet is not None



    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size
        
    if use_gpu:
        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')



    x_range = np.linspace(0, 223, 224)/224
    y_range = np.linspace(0, 223, 224)/224
    z_range = np.linspace(0, 383, 384)/384

    pos_x, pos_y, pos_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    ones= np.ones(shape=(56,56,96))
    SR_ones= np.ones(shape=(224, 224, 384))
    
    noise = torch.randn([1, net.img_channels, 56, 56, 96], device=device)
    SR_noise = torch.randn([1, net.img_channels, 224, 224, 384], device=device)

    Weight_Range=[54,66.5,80]

    for Weight in Weight_Range:
        demos=[]
        data={}
        data["sex"]=0
        data["height"]=1.65
        data["weight"]=Weight
        data["age"]=60.0
        

        height = ones * (data["height"]-1.0)
        weight = ones * (data["weight"]-30.0)/100.0
        sex = ones* data["sex"]
        age = ones * data["age"]/100.0

        demographics=np.stack ([height,weight,sex,age],axis=0)
        demos.append(demographics)
    
        demos=np.stack(demos,axis=0)
        conditions = torch.from_numpy(demos).to(device)
    
        labels = 0


        sampler_kwargs["num_steps"]= 100 
        # Generate images.
        latents = dnnlib.util.call_func_by_name(func_name=flow_sampler_conditional, net=net, noise=noise, outfolder=outdir,
                                                conditions=conditions, labels=labels, gnet=gnet, randn_like=torch.randn_like, **sampler_kwargs)
        images = encoder.decode(latents)
        
        for cidx in range (net.img_channels):
            
                    
            output = images[0, cidx, ...]
            

            # Save images.
            output = output.cpu().numpy()
            height_rescaler_Factor=146.629
            height_rescaler_offset=110.419
    
            scaler_z= (height_rescaler_Factor* float(data["height"])+height_rescaler_offset)/384.0
            output=ndimage.zoom(output,zoom=(1,1,scaler_z),cval=0.0)
            
            if cidx==0:
                modality="CT"
                output=output*1000.0
                output= output-500.0
                output=output.astype(np.int16)
                
            else:
                modality="PET"
                output = 2**(output * np.log2(26.0)) - 1.0
                output=np.clip(output,0,25)  # 0-25   
                output=output.astype(np.float32)
            
            output=np.flip(output,axis=2)
        
            
            
            affine= 8.0 * np.eye(4)
            affine[3,3]=1.0
            new_image = nibabel.Nifti1Image(output, affine=affine)
            nibabel.save(new_image, f'{outdir}/LowResimages_{data["height"]}_{data["weight"]}_{data["sex"]}_{data["age"]}.nii.gz')




        images=images.cpu().numpy()
        images=np.clip(images,0.0,1.0)
        lowres_CT = ndimage.zoom(images[0,0,:,:,:],zoom=(4,4,4),order=1,cval=0.0)
        lowres_PET = ndimage.zoom(images[0,1,:,:,:],zoom=(4,4,4),order=1,cval=0.0)
        images=np.stack ([lowres_CT,lowres_PET],axis=0)
        images=np.clip(images,0.0,1.0)
        demos=[]
        
        
        height = SR_ones * (data["height"]-1.0)
        weight = SR_ones * (data["weight"]-30.0)/100.0
        sex = SR_ones* data["sex"]
        age = SR_ones * data["age"]/100.0

        
        jumper=torch.from_numpy(images).unsqueeze(0).to(device)
        
        demographics=np.stack ([height,weight,sex,age,pos_x, pos_y, pos_z],axis=0)
        demos.append(demographics)
    
        demos=np.stack(demos,axis=0)
        conditions = torch.from_numpy(demos).to(device)

        
        sampler_kwargs["num_steps"]= 100
        
        
        

        N, C, H, W, D = SR_noise.shape

        outputs = []
        current_start = 0
        chunk_size= 384//4
        overlap = 8
        labels = 0
        
        while current_start < D:
            print (current_start," is start")
            # End of the center region for this chunk
            center_end = min(current_start + chunk_size, D)

            # Overlap on left & right (unless we're at the boundaries)
            pad_left = overlap if current_start > 0 else 0
            pad_right = overlap if center_end < D else 0

            # Extended (overlapped) chunk boundaries
            extended_start = max(0, current_start - pad_left)
            extended_end = min(D, center_end + pad_right)

            # Slice out the overlapped chunk
            SR_noise_partial = SR_noise[..., extended_start:extended_end].to(device)
            conditions_partial = conditions[..., extended_start:extended_end].to(device)
            jumper_partial=jumper[..., extended_start:extended_end].to(device)
                
                
            out_extended = dnnlib.util.call_func_by_name(func_name=flow_sampler_conditional_jump, net=srnet, noise=SR_noise_partial, jumper=jumper_partial, outfolder=outdir,
                                                    conditions=conditions_partial, labels=labels, gnet=gnet, randn_like=torch.randn_like, **sampler_kwargs)
            out_extended = encoder.decode(out_extended)
        
            
            local_center_start = pad_left
            local_center_end = local_center_start + (center_end - current_start)
            out_extended = out_extended[..., local_center_start:local_center_end]


            # Squeeze if necessary (depends on your model output shape)
            # If generator outputs (N, C, H, W), do you still need squeeze?
            out_extended = torch.squeeze(out_extended)
            # Save images.
            out_extended = out_extended.cpu().numpy()
            outputs.append(out_extended)
            # Move to the next chunk
            current_start += chunk_size

        full = np.concatenate(outputs, axis=-1)
        
        
        
        
        
        for cidx in range (net.img_channels):
            output = full[cidx, ...]
            
            
            height_rescaler_Factor=146.629 # calculated from the training data
            height_rescaler_offset=110.419 # calculated from the training data
    
            scaler_z= (height_rescaler_Factor* float(data["height"])+height_rescaler_offset)/384.0
            output=ndimage.zoom(output,zoom=(1,1,scaler_z),cval=0.0)
            
            if cidx==0:
                modality="CT"
                output=output*1000.0
                output= output-500.0
                output=output.astype(np.int16)
                
            else:
                modality="PET"
                output = 2**(output * np.log2(26.0)) - 1.0
                output=np.clip(output,0,25)  # 0-25   
                output=output.astype(np.float32)
            output=np.flip(output,axis=2)
            

            
            affine= 2.0 * np.eye(4)
            affine[3,3]=1.0
            new_image = nibabel.Nifti1Image(output, affine=affine)
            nibabel.save(new_image, f'{outdir}/images_{data["height"]}_{data["weight"]}_{data["sex"]}_{data["age"]}.nii.gz')
        
    

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.


if torch.cuda.is_available():
    use_gpu=True
    external="/external"
else:
    use_gpu=False
    external="/Volumes/HOMEDIR$"
    
   

data_dir =f"{external}/syhome/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Dataprocess/MetaData/"
model_root = f"{external}/syhome/1_Codes/20_TotalGenerator/3_Reproduce/ModelWeights/"

out_root = f"{external}/syhome/2_Datasets/20_TCIA_FDG-PET-CT-Lesions/Generated_Data/"


config_presets = {
    'cascade-SelfFlow-3D-img224':  dnnlib.EasyDict(net=f'{model_root}/Flow_LowRes_network.pkl',
                                             srnet=f'{model_root}/Flow_Patch_network.pkl'),      # fid = 3.53=
    }



@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str,  default='cascade-SelfFlownvi-3D-img224')
@click.option('--net',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--srnet',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--gnet',                     help='Guiding network pickle filename', metavar='PATH|URL',             type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, default=out_root,required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='1', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=1, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=5, show_default=True)


@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)
    opts.outdir=os.path.join(out_root,preset)
    os.makedirs(opts.outdir,exist_ok=True)
    
    print (f"Using outdir: {opts.outdir}")

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    # Generate.
    if use_gpu:
        dist.init()
    generate_cascade(**opts)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
