#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', ]

test_requirements = [ ]

setup(
    author="Nare Abgaryan",
    author_email='nare_abgaryan@edu.aua.am',
    python_requires='>=3.6',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="This package is useful for recommendation systems, and contains different models and data manipulation tools.",
    entry_points={
        'console_scripts': [
            'RecSystem=RecSystem.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme ,
    include_package_data=True,
    keywords='RecSystem',
    name='RecSystem',
    packages=find_packages(include=['RecSystem', 'RecSystem.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/anare/RecSystem',
    version='0.1.0',
    zip_safe=False,
)
