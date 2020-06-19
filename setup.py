from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='leptonic_tensor',
    version='0.0.1',
    description='Program to generate leptonic tensor for any physics model',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jxi24/LeptonicTensor',
    author='Diego Lopez Gutierrez, Joshua Isaacson',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.5, <4',
    install_requires=['numpy'],
    extras_require={
        'test': ['pytest', 'coverage']
    },
)
