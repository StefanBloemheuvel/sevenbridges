from setuptools import setup

setup(
    name='sevenbridges',
    version='0.0.11',
    description='',
    long_description='',
    author='',
    author_email='',
    url='https://pypi.org/project/sevenbridges',
    license='',
    packages=['sevenbridges'],
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'matplotlib',
        'networkx',
        'numpy<1.23.0',
        'scikit-learn',
        'scipy',
        'dtaidistance',
        'libpysal',
    ],
)
