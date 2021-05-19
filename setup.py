import builtins
import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from Cython.Build import cythonize  # isort:skip
except ImportError:
    wmsg = "WARNING: Cython unavailable, unable to build stratify extensions"
    print(wmsg)
    cythonize = None


PACKAGE_NAME = "stratify"
CMDS_NOCYTHONIZE = ["clean", "sdist"]


class NumpyBuildExt(build_ext):
    # Delay numpy import so that setup.py can be run
    # without numpy already being installed.
    def finalize_options(self):
        build_ext.finalize_options(self)
        builtins.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


cython_coverage_enabled = os.environ.get("CYTHON_COVERAGE", None)
cython_directives = {}
# TODO: investigate "'PyArrayObject' has no member named 'dimensions'" cython error
#       https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#configuring-the-c-build
# define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
define_macros = []
extensions = []

if cythonize and cython_coverage_enabled:
    define_macros.append(("CYTHON_TRACE_NOGIL", "1"))
    cython_directives.update({"linetrace": True})

for fname in Path.cwd().glob(f"{PACKAGE_NAME}/*.pyx"):
    extensions.append(
        Extension(
            f"{PACKAGE_NAME}.{fname.stem}",
            sources=[str(fname)],
            define_macros=define_macros,
        )
    )

if cythonize and not any([arg in CMDS_NOCYTHONIZE for arg in sys.argv]):
    extensions = cythonize(
        extensions, compiler_directives=cython_directives, language_level=3
    )

kwargs = dict(
    cmdclass={"build_ext": NumpyBuildExt},
    ext_modules=extensions,
)

setup(**kwargs)
