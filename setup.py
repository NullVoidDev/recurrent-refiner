from setuptools import setup, find_packages

setup(
    name="recurrent-refiner",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.43.0",
        "huggingface_hub>=0.24.0",
    ],
)
