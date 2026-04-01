from setuptools import setup, find_packages

# Basic setup.py of the project to install the package
setup(
    name="torch-pol",
    version="0.0.2",
    description="Pytorch implementation of a Laplacian pyramid optimization layer",
    url="https://github.netflix.net/jphilip-nc/torch_PoL",
    author="Julien Philip",
    author_email="jphilip@netflixcontractors.com",
    packages=find_packages(),
    install_requires=["torch"],
)
