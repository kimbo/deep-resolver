from setuptools import setup, find_packages

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

with open('README.md', "r") as readme:
    long_description = readme.read()

setup(
    name="deep-resolver",
    version="1.0",
    author="Kimball Leavitt",
    author_email="kimballleavitt@gmail.com",
    description="Fabricated response to DNS queries using deep learning",
    long_description=long_description,
    url="https://github.com/kimbo/deep-resolver",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'generate-dataset = deepresolver.scan:main',
            'train-deep-resolver = deepresolver.train:main',
        ],
    },
    install_requires=[
        'torch',
        'dnspython',
        'tqdm',
        'matplotlib',
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)