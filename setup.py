"""
DiffiT-LoRA: Diffusion Vision Transformers with LoRA Fine-tuning

A professional implementation of DiffiT (Diffusion Vision Transformers) with 
Low-Rank Adaptation (LoRA) for efficient fine-tuning.
"""

from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diffit-lora",
    version="0.1.0",
    author="DiffiT Team",
    author_email="contact@diffit.ai",
    description="Diffusion Vision Transformers with LoRA Fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "diffit-train=diffit.cli.train:main",
            "diffit-finetune=diffit.cli.finetune:main",
            "diffit-generate=diffit.cli.generate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
