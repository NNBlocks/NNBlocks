from setuptools import setup, find_packages

setup(
    name="NNBlocks",
    version="0.1.a.dev",
    install_requires = ['theano>=0.7.0', 'matplotlib'],
    packages=find_packages(),
    author="Frederico Tommasi Caroli",
    author_email="ftcaroli@gmail.com",
    description="NNBlocks is a Deep Learning framework made to build linguistics neural models",
    license="GPL v3",
    keywords="deep learning linguistics machine learning theano",
    url="https://github.com/NNBlocks/NNBlocks"
)
