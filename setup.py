# -*- coding: utf-8 -*-
import sys
import io
import setuptools

try:
    from Cython.Build import cythonize
except ImportError:
    CYTHON = False
else:
    CYTHON = 'bdist_wheel' not in sys.argv

setuptools.setup(
    name='pymake',
    version='0.1',
    author='Adrien Dulac',
    packages=setuptools.find_packages(),
    entry_points = {
        'console_scripts': ['pymake=pymake.zymake'],
    }
    #include_package_data=True,
)
