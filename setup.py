from __future__ import absolute_import, division, print_function

from distutils.core import setup
import os

import numpy as np
import setuptools

from Cython.Build import cythonize


extensions = [setuptools.Extension('stratify._vinterp',
                                   ['stratify/_vinterp.pyx'],
                                   include_dirs=[np.get_include()])]

setup(
    name='stratify',
    description='Vectorized interpolators that are especially useful for Nd vertical interpolation/stratification of atmospheric and oceanographic datasets',
    version='0.2.0',
    ext_modules=cythonize(extensions),
    packages=['stratify', 'stratify.tests'],
    classifiers=[
            'Development Status :: 3 - Alpha',
            ('License :: OSI Approved :: '
             'License :: OSI Approved :: BSD License'),
             'Operating System :: MacOS :: MacOS X',
             'Operating System :: POSIX',
             'Operating System :: POSIX :: AIX',
             'Operating System :: POSIX :: Linux',
             'Operating System :: Microsoft :: Windows',
             'Programming Language :: Python',
             'Programming Language :: Python :: 2',
             'Programming Language :: Python :: 2.7',
             'Programming Language :: Python :: 3',
             'Programming Language :: Python :: 3.4',
             'Programming Language :: Python :: 3.5',
             'Topic :: Scientific/Engineering',
             'Topic :: Scientific/Engineering :: GIS',
    ],
)
