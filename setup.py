from setuptools import setup, find_packages


setup(
    name="ANNforEpi",
    version="0.1",
    author="sibyl-team",
    packages=find_packages(),
    description="Epidemic inference with autoregressive neural networks",
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "torch"
    ]
)