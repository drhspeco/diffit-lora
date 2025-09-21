"""
Data Loading for DiffiT

EXACT preservation of original dataset loading algorithms
"""

import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from datasets import load_dataset
from torchvision import transforms
from typing import Optional, List, Dict, Any


class DiffiTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for DiffiT datasets
    
    EXACT preservation of original dataset loading logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data_config = config.get('data', {})
        
        # Dataset parameters
        self.dataset_name = self.data_config.get('name', 'CIFAR')
        self.batch_size_train = self.data_config.get('batch_size_train', 64)
        self.batch_size_test = self.data_config.get('batch_size_test', 16)
        self.num_workers = self.data_config.get('num_workers', 2)
        self.pin_memory = self.data_config.get('pin_memory', True)
        
        # Data transforms
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup data transforms based on configuration"""
        # Base transform for CIFAR-10 and Imagenette
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1] range
        ])
        
        # Training transform with optional augmentation
        train_transforms = []
        if self.data_config.get('augmentation', {}).get('enabled', False):
            train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
            train_transforms.append(transforms.RandomRotation(degrees=5))
        
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.train_transform = transforms.Compose(train_transforms)
        
    def prepare_data(self):
        """Download datasets if needed"""
        if self.dataset_name == "CIFAR":
            load_dataset("cifar10")
        elif self.dataset_name == "IMAGENETTE":
            load_dataset("frgfm/imagenette", "160px")
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for training/validation/testing"""
        if self.dataset_name == "CIFAR":
            self.train_data, self.val_data = self._get_cifar_ds()
        elif self.dataset_name == "IMAGENETTE":
            self.train_data, self.val_data = self._get_imagenette_ds()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _get_cifar_ds(self):
        """
        Load CIFAR-10 dataset
        
        EXACT preservation from original get_cifar_ds function
        """
        dataset = load_dataset("cifar10")

        train = dataset["train"]
        test = dataset["test"]

        train_tensor = []
        for image in train:
            image_tensor = self.train_transform(image["img"])
            train_tensor.append(image_tensor)

        test_tensor = []
        for image in test:
            image_tensor = self.base_transform(image["img"])
            test_tensor.append(image_tensor)

        # Split train data for validation if needed
        val_split = self.data_config.get('val_split', 0.1)
        if val_split > 0:
            train_size = int((1 - val_split) * len(train_tensor))
            val_size = len(train_tensor) - train_size

            train_split, val_split = random_split(
                train_tensor, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            train_data = list(train_split)
            val_data = list(val_split) + test_tensor
        else:
            train_data = train_tensor
            val_data = test_tensor

        return train_data, val_data
    
    def _get_imagenette_ds(self):
        """
        Load Imagenette dataset
        
        EXACT preservation from original get_imagenette_ds function
        """
        dataset = load_dataset("frgfm/imagenette", "160px")

        train = dataset["train"]
        test = dataset["validation"]

        # Resize transform for Imagenette
        resize_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        train_tensor = []
        for image in train:
            image_tensor = resize_transform(image["image"])
            if image_tensor.shape[0] == 3:  # Only RGB images
                # Apply normalization
                image_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)
                train_tensor.append(image_tensor)

        test_tensor = []
        for image in test:
            image_tensor = resize_transform(image["image"])
            if image_tensor.shape[0] == 3:  # Only RGB images
                # Apply normalization
                image_tensor = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image_tensor)
                test_tensor.append(image_tensor)

        return train_tensor, test_tensor
    
    def train_dataloader(self):
        """Training dataloader"""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size_test,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Test dataloader (same as validation)"""
        return self.val_dataloader()
