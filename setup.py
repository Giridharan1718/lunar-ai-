from setuptools import setup, find_packages

setup(
    name="lunar_core",
    version="1.0.0",
    description="ISRO Chandrayaan-2 Lunar Image Registration System",
    author="ISRO Computer Vision & Remote Sensing Architecture Group",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "opencv-python-headless>=4.8.0",
        "torch>=2.0.0",
        "scikit-image>=0.21.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
        "tqdm>=4.66.0",
    ],
)
