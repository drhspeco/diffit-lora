"""
Forward Diffusion Process

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import torch

from .utils import extract
from .schedulers import linear_beta_schedule


def q_sample(x_start, t, noise, timesteps=500):
    """
    Forward diffusion process: x_t-1 -> x_t
    
    EXACT preservation from original implementation.
    
    Args:
        x_start: Initial clean image
        t: Timestep tensor
        noise: Gaussian noise to add
        timesteps: Total number of timesteps (default 500, same as original T)
        
    Returns:
        Noisy image at timestep t
    """
    # Define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # Define alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # Calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_start.shape)

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
