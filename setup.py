"""
A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='chajda',
    version='1.0.0',
    description='Postgresql extension for advanced multilingual full text search',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mikeizbicki/chajda',
    author='Mike Izbicki',
    author_email='mike@izbicki.me',
    keywords='postgresql, postgres, spacy',
    packages=find_packages(),
    python_requires='>=3.6, <4',
)
