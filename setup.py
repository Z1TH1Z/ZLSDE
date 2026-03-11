"""Setup configuration for ZLSDE pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zlsde",
    version="0.1.0",
    author="ZLSDE Team",
    description="Zero-Label Self-Discovering Dataset Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/zlsde",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
        "hdbscan>=0.8.29",
        "scikit-learn>=1.3.0",
        "umap-learn>=0.5.3",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "hypothesis>=6.75.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "optional": [
            "faiss-cpu>=1.7.4",
            "datasets>=2.12.0",
            "pillow>=9.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "zlsde=zlsde.cli:main",
        ],
    },
)
