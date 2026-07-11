from setuptools import setup, find_packages

setup(
    name="pink-elephant-1b",
    version="0.1.0",
    description="Pink Elephant:1B - A 1B parameter language model from scratch",
    author="Pink Elephant AI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "train": [
            "datasets>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
        ],
    },
)
