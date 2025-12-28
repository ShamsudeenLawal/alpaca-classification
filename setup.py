from setuptools import setup, find_packages

def read_requirements(filepath):
    with open(filepath) as f:
        requirements = f.read().splitlines()
    # Filter out any lines that are comments or empty
    requirements = [line for line in requirements if line and not line.startswith('#')]
    # remove -e . if present
    requirements = [line for line in requirements if line != '-e .']
    return requirements


setup(
    name='alpaca-classification',
    version='0.1.0',
    packages=find_packages(),
    install_requires=read_requirements("requirements.txt"),
    author='Shamsudeen Lawal',
    author_email='sendtolawal@gmail.com',
    description='A package for Alpaca Classification tasks'
)