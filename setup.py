from setuptools import setup, find_packages


setup(
    name="ANNforE",
    version="0.1",
    author="Indaco Biazzo, Fabio Mazza",
    packages=find_packages(),
    description="Epidemic inference with autoregressive neural networks",
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "torch"
    ]
)
