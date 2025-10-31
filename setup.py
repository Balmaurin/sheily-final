#!/usr/bin/env python3
"""
Sheily AI - Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="sheily-ai",
    version="1.0.0",
    author="Sheily AI Research Team",
    author_email="contact@sheily-ai.dev",
    description="Sistema enterprise de entrenamiento de IA con 50+ dominios especializados",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sheily-ai/sheily-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "bitsandbytes>=0.41.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sheily-train=sheily_train.train_branch:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
