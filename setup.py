from __future__ import absolute_import, division, print_function

import glob
import os
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext

# Python 2 has a different name for builtins.
try:
    import builtins
except ImportError:
    import __builtin__ as builtins

try:
    from Cython.Build import cythonize
except ImportError:
    print('Warning: Cython is not available - '
          'will be unable to build stratify extensions')
    cythonize = None

PACKAGE_NAME = 'stratify'
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
CMDS_NOCYTHONIZE = ['clean', 'sdist']


class NumpyBuildExt(_build_ext):
    # Delay numpy import so that setup.py can be run without numpy already
    # being installed.
    def finalize_options(self):
        _build_ext.finalize_options(self)
        builtins.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def extract_version():
    version = None
    fname = os.path.join(PACKAGE_DIR, PACKAGE_NAME, '__init__.py')
    with open(fname) as fd:
        for line in fd:
            if line.startswith('__version__'):
                _, version = line.split('=')
                version = version.strip()[1:-1]  # Remove quotations
                break
    return version


# Python 2 is not supported by numpy as of version 1.17
# but pip will attempt to install/use newer Python 3-only numpy versions.
numpy_req = 'numpy<1.17' if sys.version_info.major < 3 else 'numpy'
requirements = ['setuptools>=40.8.0', numpy_req, 'Cython']

extension_kwargs = {}
cython_directives = {'binding': True}
cython_coverage_enabled = os.environ.get('CYTHON_COVERAGE', None)
if cythonize and cython_coverage_enabled:
    extension_kwargs.update({'define_macros': [('CYTHON_TRACE_NOGIL', '1')]})
    cython_directives.update({'linetrace': True})

extensions = []
for source_file in glob.glob('{}/*.pyx'.format(PACKAGE_NAME)):
    source_file_nosuf, _ = os.path.splitext(os.path.basename(source_file))
    extensions.append(
        Extension('{}.{}'.format(PACKAGE_NAME, source_file_nosuf),
                  sources=[source_file], **extension_kwargs))

if cythonize and not any([arg in CMDS_NOCYTHONIZE for arg in sys.argv]):
    extensions = cythonize(extensions, compiler_directives=cython_directives)


setup_args = dict(
    name=PACKAGE_NAME,
    description=('Vectorized interpolators that are especially useful for '
                 'Nd vertical interpolation/stratification of atmospheric '
                 'and oceanographic datasets'),
    author='UK Met Office',
    author_email='scitools-iris-dev@googlegroups.com',
    url='https://github.com/SciTools-incubator/python-stratify',
    version=extract_version(),
    cmdclass={'build_ext': NumpyBuildExt},
    ext_modules=extensions,
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
    ],
    extras_require={'test': ['nose']},
    setup_requires=requirements,
    install_requires=requirements,
    test_suite='{}.tests'.format(PACKAGE_NAME),
    zip_safe=False,
)


if __name__ == '__main__':
    setup(**setup_args)
