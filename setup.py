# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='systematic-metafeatures',
    version='0.0.1',
    description='Systematic Generation of Metafeatures for Metalearning',
    long_description=readme,
    author='Fábio Pinto and Matthias Feurer',
    author_email='fhpinto@inesctec.pt, feurerm@informatik.uni-freiburg.de',
    url='https://github.com/fhpinto/systematic-metafeatures',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)