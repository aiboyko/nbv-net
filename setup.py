from setuptools import setup, find_packages
from os.path import join, dirname

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cnbv',
    version='0.1',
    url='https://github.com/aiboyko/nbv-net',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages('cnbv'),
    platforms='any',
    install_requires=['numpy>=1.18.3',
                      'matplotlib >= 3',
                      'torch >= 1.4',
                      'torchvision'],
)
