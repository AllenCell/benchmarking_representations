from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    install_requires=requirements,
    description="benchmarking different methods for extracting unsupervised representations from images",
    author="Ritvik Vasan",
    license="",
)
