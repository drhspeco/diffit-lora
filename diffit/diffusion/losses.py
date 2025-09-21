"""
Diffusion Loss Functions

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import torch
import torch.nn.functional as F

from .forward import q_sample


def p_losses(denoise_model, x_start, t, l=None, timesteps=500):
    """
    Compute diffusion loss
    
    EXACT preservation from original implementation.
    
    Args:
        denoise_model: The denoising model
        x_start: Clean images
        t: Timestep tensor
        l: Labels (optional, for conditional generation)
        timesteps: Total number of timesteps (default 500, same as original T)
        
    Returns:
        Diffusion loss (smooth L1 loss between predicted and actual noise)
    """
    # Get Gaussian noise
    noise = torch.randn_like(x_start)

    # Apply the noise to the image
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise, timesteps=timesteps)
    predicted_noise = denoise_model(x_noisy, t, l)

    # Use smooth L1 loss instead of MSE
    return F.smooth_l1_loss(noise, predicted_noise)
