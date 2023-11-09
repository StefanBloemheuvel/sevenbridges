from setuptools import setup, find_packages

setup(
    name='7bridges',
    version='0.0.4',
    description='',
    long_description='',
    author='',
    author_email='',
    url='https://pypi.org/project/7bridges',
    license='',
    py_modules=['7bridges'],
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'matplotlib',
        'networkx',
        'numpy',
        'scikit-learn',
        'scipy',
        'dtaidistance',
        'minepy',
        'libpysal',
    ],
    entry_points={
        'console_scripts': [
            '7bridges = 7bridges:main'
        ]
    }
)
