"""
Reverse Diffusion Process (Sampling)

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .utils import extract
from .schedulers import linear_beta_schedule


@torch.no_grad()
def p_sample(model, x, t, t_index, timesteps=500):
    """
    Single step of reverse diffusion process
    
    EXACT preservation from original implementation.
    
    Args:
        model: Diffusion model
        x: Current noisy image
        t: Current timestep tensor
        t_index: Current timestep index
        timesteps: Total number of timesteps (default 500, same as original T)
        
    Returns:
        Denoised image for previous timestep
    """
    # Define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # Define alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # Calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # Calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, None) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape, timesteps=500):
    """
    Complete reverse diffusion sampling loop
    
    EXACT preservation from original implementation.
    
    Args:
        model: Diffusion model
        shape: Shape of images to generate (batch_size, channels, height, width)
        timesteps: Total number of timesteps (default 500, same as original T)
        
    Returns:
        List of images through the reverse diffusion process
    """
    device = next(model.parameters()).device

    b = shape[0]
    # Start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc="sampling loop time step", total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, timesteps)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, timesteps=500):
    """
    Generate samples from the model
    
    EXACT preservation from original implementation.
    
    Args:
        model: Diffusion model
        image_size: Size of images to generate
        batch_size: Number of images to generate
        channels: Number of channels (default 3 for RGB)
        timesteps: Total number of timesteps (default 500, same as original T)
        
    Returns:
        List of generated images through reverse diffusion
    """
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), timesteps=timesteps)
