from setuptools import setup, find_packages

setup(
    name='sevenbridges',
    version='0.0.1',
    description='',
    long_description='',
    author='',
    author_email='',
    url='https://pypi.org/project/sevenbridges',
    license='',
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
    entry_points={
        'console_scripts': [
            'sevenbridges = sevenbridges:main'
        ]
    }
)
