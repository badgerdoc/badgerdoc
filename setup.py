#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='badgerdoc',
    description='Pdf parsing and objects extraction library',
    version='0.1',
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        'gdown==3.12.2',
        'torch==1.7.0',
        'torchvision==0.8.1',
        'pillow==7.2.0',
        'click==7.1.2',
        'scipy==1.5.4',
        'pdf2image==1.14.0',
        'python-poppler==0.2.2',
        'tesserocr==2.5.1',
        'paddleocr==2.0.2',
        'paddlepaddle==2.0rc1',
    ],
)
