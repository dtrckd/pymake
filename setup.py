# -*- coding: utf-8 -*-
import setuptools
from codecs import open
from os import path
import sys, io

__version__ = '0.1'

try:
    from Cython.Build import cythonize
except ImportError:
    CYTHON = False
else:
    CYTHON = 'bdist_wheel' not in sys.argv

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if not (x.strip().startswith(('#', '//')) or x.strip().endswith('?'))]
install_requires = list(filter(None, install_requires))

setuptools.setup(
    name='pymake',
    version=__version__,
    author='dtrckd',
    author_email='dtrckd@gmail.com',
    description='A platform for managing I/O of ML experiments.',
    url='https://github.com/dtrckd/pymake',
    license='GPL',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    entry_points = {
        'console_scripts': ['pymake=pymake.zymake:main'],
    },
    package_data = {'pymake' : ['pymake.cfg']},
    include_package_data=True,
)
