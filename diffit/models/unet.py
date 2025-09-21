"""
U-shaped DiffiT Network for Image-space Diffusion

EXACT preservation of original algorithms from diffit_image_space_architecture.py
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .components import (
    ResBlockGroup,
    Tokenizer,
    Head,
    Downsample,
    Upsample,
)


class UShapedNetwork(pl.LightningModule):
    """
    U-shaped DiffiT network for image-space diffusion
    
    EXACT preservation from original implementation.
    
    This is the main image-space DiffiT model that uses a U-Net-like architecture
    with transformer blocks for diffusion-based image generation.
    """
    
    def __init__(self, learning_rate: float, d_model: int, num_heads: int, dropout: float,
                 d_ff: int, img_size: int, device, denoising_steps: int,
                 L1: int = 2, L2: int = 2, L3: int = 2, L4: int = 2):
        super().__init__()
        d_model_2 = d_model * 2

        self.learning_rate = learning_rate
        self.denoising_steps = denoising_steps

        self.diffit_res_block_group_1 = ResBlockGroup(d_model, num_heads, dropout, d_ff, L1,
                                                     in_channels=d_model, out_channels=d_model,
                                                     img_size=img_size, device=device)
        self.diffit_res_block_group_2 = ResBlockGroup(d_model, num_heads, dropout, d_ff, L2,
                                                     in_channels=d_model_2, out_channels=d_model_2,
                                                     img_size=img_size // 2, device=device)
        self.diffit_res_block_group_3 = ResBlockGroup(d_model, num_heads, dropout, d_ff, L3,
                                                     in_channels=d_model_2, out_channels=d_model_2,
                                                     img_size=img_size // 4, device=device)

        self.downsample_1 = Downsample(in_channels=d_model, out_channels=d_model_2)
        self.downsample_2 = Downsample(in_channels=d_model_2, out_channels=d_model_2)

        self.upsample_1 = Upsample(in_channels=d_model_2, out_channels=d_model_2)
        self.upsample_2 = Upsample(in_channels=d_model_2, out_channels=d_model)

        self.tokenizer = Tokenizer(out_channels=d_model)
        self.head = Head(in_channels=d_model)

    def uShape(self, xs, t):
        """
        U-shaped forward pass
        
        EXACT preservation from original implementation.
        
        Args:
            xs: Input spatial features after tokenization
            t: Time step
            
        Returns:
            Processed features through U-shaped architecture
        """
        output_downsample_1 = self.downsample_1(self.diffit_res_block_group_1(xs, t))
        output_downsample_2 = self.downsample_2(self.diffit_res_block_group_2(output_downsample_1, t))
        uLeft = output_downsample_2

        uCenter = self.diffit_res_block_group_3(uLeft, t)

        input_upsample_1 = uCenter + uLeft
        input_upsample_2 = (self.diffit_res_block_group_2(self.upsample_1(input_upsample_1), t)
                           + output_downsample_1)
        uRight = self.diffit_res_block_group_1(self.upsample_2(input_upsample_2), t)

        return uRight

    def forward(self, xs, t, l=None):
        """
        Forward pass through complete U-shaped network
        
        Args:
            xs: Input images
            t: Time steps
            l: Labels (unused in image-space model)
            
        Returns:
            Predicted noise/reconstructed images
        """
        return self.head(self.uShape(self.tokenizer(xs), t))

    def training_step(self, batch, batch_idx):
        """
        PyTorch Lightning training step
        
        EXACT preservation from original implementation.
        """
        t = torch.randint(0, self.denoising_steps, (len(batch),), device=self.device).long()
        
        # Import here to avoid circular dependencies
        from ..diffusion import p_losses
        loss = p_losses(self, batch, t)

        # Log every 25 batches (preserved from original)
        if batch_idx % 25 == 0:
            print(f"loss: {loss:.2f}")

        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """
        PyTorch Lightning test step
        
        EXACT preservation from original implementation.
        """
        t = torch.randint(0, self.denoising_steps, (len(batch),), device=self.device).long()
        
        # Import here to avoid circular dependencies
        from ..diffusion import p_losses
        loss = p_losses(self, batch, t)

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers for PyTorch Lightning
        
        EXACT preservation from original implementation.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
