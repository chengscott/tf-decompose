from setuptools import setup, find_packages

setup(
    name='tf_decompose',
    version='0.1',
    packages=find_packages(exclude=['examples', 'tests']),
    install_requires=[
        'tensorflow',
        'numpy',
        'scipy',
    ],
)
