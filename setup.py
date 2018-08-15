from setuptools import setup
import tf_decompose

with open('README.md', 'r', encoding='utf-8') as fd:
    setup(
        name='tf_decompose',
        version=tf_decompose.__version__,
        author='ebigelow, chengscott',
        maintainer='chengscott',
        description='Tensor decomposition with TensorFlow',
        long_description=fd.read(),
        long_description_content_type='text/markdown',
        url='https://github.com/chengscott/tf-decompose',
        packages=['tf_decompose'],
        install_requires=[
            'tensorflow-gpu',
            'numpy',
            'scipy',
        ],
    )
