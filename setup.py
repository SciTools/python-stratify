from __future__ import absolute_import, division, print_function

from setuptools import setup, find_packages, Extension
import os

import numpy as np

from Cython.Build import cythonize


NAME = 'stratify'
DIR = os.path.abspath(os.path.dirname(__file__))

extension_kwargs = {'include_dirs': [np.get_include()]}
cython_coverage_enabled = os.environ.get('CYTHON_COVERAGE', None)
if cython_coverage_enabled:
    extension_kwargs.update({'define_macros': [('CYTHON_TRACE_NOGIL', '1')]})

extensions = [Extension('{}._vinterp'.format(NAME),
                        [os.path.join(NAME, '_vinterp.pyx')],
                        **extension_kwargs),
              Extension('{}._conservative'.format(NAME),
                        [os.path.join(NAME, '_conservative.pyx')],
                        **extension_kwargs)]

def extract_version():
    version = None
    fname = os.path.join(DIR, NAME, '__init__.py')
    with open(fname) as fd:
        for line in fd:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotations
                break
    return version


setup_args = dict(
    name=NAME,
    description=('Vectorized interpolators that are especially useful for '
                 'Nd vertical interpolation/stratification of atmospheric '
                 'and oceanographic datasets'),
    version=extract_version(),
    ext_modules=cythonize(extensions, compiler_directives={'linetrace': True,
                                                           'binding': True}),
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
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
    extras_require={'test': ['nose']},
    test_suite='{}.tests'.format(NAME),
    zip_safe=False,
)


if __name__ == '__main__':
    setup(**setup_args)
