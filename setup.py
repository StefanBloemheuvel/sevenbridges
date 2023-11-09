from setuptools import setup

setup(
    name='sevenbridges',
    version='0.0.7',
    description='',
    long_description='',
    author='',
    author_email='',
    url='https://pypi.org/project/sevenbridges',
    license='',
    # packages=['sevenbridges'],
    py_modules=['sevenbridges'],
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'matplotlib',
        'networkx',
        'numpy',
        'scikit-learn',
        'scipy',
        'dtaidistance',
        'libpysal',
    ],
)
