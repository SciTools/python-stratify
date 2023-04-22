import os
import sys
import warnings
from pathlib import Path

# safe to import numpy here thanks to pep518
import numpy as np
from setuptools import Command, Extension, setup

try:
    from Cython.Build import cythonize  # isort:skip
except ImportError:
    wmsg = "Cython unavailable, unable to build stratify extensions!"
    warnings.warn(wmsg)
    cythonize = None


BASE_DIR = Path(__file__).resolve().parent
PACKAGE_NAME = "stratify"
SRC_DIR = BASE_DIR / "src"
STRATIFY_DIR = SRC_DIR / PACKAGE_NAME
CMDS_NOCYTHONIZE = ["clean", "sdist"]
FLAG_COVERAGE = "--cython-coverage"  # custom flag enabling Cython line tracing


class CleanCython(Command):
    description = "Purge artifacts built by Cython"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path in STRATIFY_DIR.rglob("*"):
            if path.suffix in (".pyc", ".pyo", ".c", ".so"):
                msg = f"clean: removing file {path}"
                print(msg)
                path.unlink()


cython_coverage_enabled = (
    os.environ.get("CYTHON_COVERAGE", None) or FLAG_COVERAGE in sys.argv
)
cython_directives = {}
define_macros = []
extensions = []

if cythonize and cython_coverage_enabled:
    define_macros.extend(
        [
            ("CYTHON_TRACE", "1"),
            ("CYTHON_TRACE_NOGIL", "1"),
        ]
    )
    cython_directives.update({"linetrace": True})
    if FLAG_COVERAGE in sys.argv:
        sys.argv.remove(FLAG_COVERAGE)
    print('enabled "linetrace" Cython compiler directive')

for fname in SRC_DIR.glob(f"{PACKAGE_NAME}/*.pyx"):
    # ref: https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
    extension = Extension(
        f"{PACKAGE_NAME}.{fname.stem}",
        sources=[str(fname.relative_to(BASE_DIR))],
        include_dirs=[np.get_include()],
        define_macros=define_macros,
    )
    extensions.append(extension)

if cythonize and not any([arg in CMDS_NOCYTHONIZE for arg in sys.argv]):
    extensions = cythonize(
        extensions, compiler_directives=cython_directives, language_level=3
    )

cmdclass = {"clean_cython": CleanCython}
kwargs = {"cmdclass": cmdclass, "ext_modules": extensions}
setup(**kwargs)
