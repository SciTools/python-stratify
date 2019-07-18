from __future__ import absolute_import, division, print_function

from distutils.command.sdist import sdist as _sdist
import glob
import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext

PACKAGE_NAME = 'stratify'
PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
CYTHON_DIRECTIVES = {'linetrace': True, 'binding': True}

try:
    # Detect if Cython is available. Where it is, use it, otherwise fall back
    # to C source included with source distribution.
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

source_suffix = '.pyx' if cythonize else '.c'
extension_kwargs = {}
cython_coverage_enabled = os.environ.get('CYTHON_COVERAGE', None)
if cythonize and cython_coverage_enabled:
    extension_kwargs.update({'define_macros': [('CYTHON_TRACE_NOGIL', '1')]})
if cythonize:
    # Cython requires numpy include headers
    import numpy as np
    extension_kwargs.update({'include_dirs': [np.get_include()]})

extensions = []
cmdclass = {}
for source_file in glob.glob('{}/*{}'.format(PACKAGE_NAME, source_suffix)):
    source_file_nosuf, _ = os.path.splitext(os.path.basename(source_file))
    extensions.append(
        Extension('{}.{}'.format(PACKAGE_NAME, source_file_nosuf),
                  sources=['{}/{}{}'.format(
                      PACKAGE_NAME, source_file_nosuf, source_suffix)],
                  **extension_kwargs))

if cythonize:
    extensions = cythonize(extensions, compiler_directives=CYTHON_DIRECTIVES)


class SDist(_sdist):
    # Source distribution build runs Cython so that Cython is not needed as
    # an install dependency.
    def run(self):
        cythonize(extensions, compiler_directives=CYTHON_DIRECTIVES)
        _sdist.run(self)


cmdclass['sdist'] = SDist


def numpy_build_ext(pars):
    # Delay numpy import so that setup.py can be run without numpy already
    # being installed.
    class NumpyBuildExt(_build_ext):
        def finalize_options(self):
            _build_ext.finalize_options(self)
            __builtins__.__NUMPY_SETUP__ = False
            import numpy
            self.include_dirs.append(numpy.get_include())

    return NumpyBuildExt(pars)


cmdclass['build_ext'] = numpy_build_ext


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


setup_args = dict(
    name=PACKAGE_NAME,
    description=('Vectorized interpolators that are especially useful for '
                 'Nd vertical interpolation/stratification of atmospheric '
                 'and oceanographic datasets'),
    author='UK Met Office',
    author_email='scitools-iris-dev@googlegroups.com',
    url='https://github.com/SciTools-incubator/python-stratify',
    version=extract_version(),
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
    setup_requires=['setuptools>=18.0', 'numpy'],
    install_requires=['numpy'],
    ext_modules=extensions,
    cmdclass=cmdclass,
    test_suite='{}.tests'.format(PACKAGE_NAME),
    zip_safe=False,
)


if __name__ == '__main__':
    setup(**setup_args)
