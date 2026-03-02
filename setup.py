"""
TinyGNN — setup.py for building the pybind11 Python extension.

Build:
    python setup.py build_ext --inplace

Or install in development mode:
    pip install -e .

The extension compiles all C++ source files together with the pybind11
bindings into a single shared library (_tinygnn_core.pyd on Windows).
"""

import os
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# ── Work around setuptools >=82 shlex.split() bug on Windows ─────────────────
# shlex.split() treats backslashes as escape characters, mangling Windows paths
# like "C:\msys64\ucrt64\bin\gcc.exe" → "C:msys64ucrt64bingcc.exe".
# This triggers if CC / CXX env vars contain back-slashed paths.
# Fix: normalise those vars to use forward slashes before setuptools sees them.
if sys.platform == "win32":
    for _var in ("CC", "CXX"):
        _val = os.environ.get(_var, "")
        if "\\" in _val:
            os.environ[_var] = _val.replace("\\", "/")

# ── Paths (relative to setup.py — required by setuptools for sdist) ──────────
INCLUDE_DIR = "include"
SRC_DIR = "src"
PYTHON_DIR = "python"


def _is_x86_arch() -> bool:
    machine = platform.machine().lower()
    return machine in {"x86_64", "amd64", "x86", "i386", "i686"}


def _detect_libomp_prefix() -> str:
    for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if os.path.isdir(prefix):
            return prefix
    return ""


def get_pybind11_include():
    """Get pybind11 include path."""
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        raise RuntimeError(
            "pybind11 is required. Install with: pip install pybind11"
        )


# ── Source files ─────────────────────────────────────────────────────────────
cpp_sources = [
    os.path.join(SRC_DIR, "tensor.cpp"),
    os.path.join(SRC_DIR, "graph_loader.cpp"),
    os.path.join(SRC_DIR, "ops.cpp"),
    os.path.join(SRC_DIR, "layers.cpp"),
    os.path.join(SRC_DIR, "model.cpp"),
    os.path.join(PYTHON_DIR, "tinygnn_ext.cpp"),
]


class BuildExt(build_ext):
    """Custom build_ext that sets C++17 and appropriate compiler flags.
    
    On Windows, auto-detects MinGW when MSVC is not available.
    """

    def initialize_options(self):
        super().initialize_options()
        # Auto-detect MinGW on Windows when MSVC is absent
        if sys.platform == "win32" and self.compiler is None:
            import shutil
            if shutil.which("gcc") and not shutil.which("cl"):
                self.compiler = "mingw32"

    def build_extensions(self):
        # Detect compiler
        ct = self.compiler.compiler_type

        for ext in self.extensions:
            if ct == "unix" or ct == "mingw32":
                ext.extra_compile_args = [
                    "-std=c++17",
                    "-O3",
                    "-Wall",
                    "-Wextra",
                    "-fPIC",
                    "-DNDEBUG",
                ]

                if _is_x86_arch() and sys.platform != "darwin":
                    ext.extra_compile_args += ["-mavx2", "-mfma"]

                if ct == "mingw32":
                    # Static-link GCC runtime so .pyd doesn't depend on MinGW DLLs
                    ext.extra_compile_args.append("-fopenmp")
                    ext.extra_link_args = [
                        "-static-libgcc",
                        "-static-libstdc++",
                        "-Wl,-Bstatic",
                        "-lgomp",
                        "-lwinpthread",
                        "-Wl,-Bdynamic",
                    ]
                else:
                    if sys.platform == "darwin":
                        # Apple clang needs libomp from Homebrew and uses
                        # -Xpreprocessor -fopenmp instead of plain -fopenmp.
                        ext.extra_compile_args += ["-Xpreprocessor", "-fopenmp"]
                        libomp_prefix = os.environ.get("LIBOMP_PREFIX", "") or _detect_libomp_prefix()
                        if libomp_prefix:
                            ext.include_dirs.append(os.path.join(libomp_prefix, "include"))
                            ext.extra_link_args = [
                                f"-L{os.path.join(libomp_prefix, 'lib')}",
                                "-lomp",
                                f"-Wl,-rpath,{os.path.join(libomp_prefix, 'lib')}",
                            ]
                        else:
                            ext.extra_link_args = ["-lomp"]
                    else:
                        ext.extra_compile_args.append("-fopenmp")
                        ext.extra_link_args = ["-fopenmp"]
            elif ct == "msvc":
                ext.extra_compile_args = [
                    "/std:c++17",
                    "/O2",
                    "/W4",
                    "/DNDEBUG",
                    "/EHsc",
                    "/openmp",
                ]
                if _is_x86_arch():
                    ext.extra_compile_args.append("/arch:AVX2")
                ext.extra_link_args = []

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        name="_tinygnn_core",
        sources=cpp_sources,
        include_dirs=[
            INCLUDE_DIR,
            get_pybind11_include(),
        ],
        language="c++",
    ),
]


setup(
    name="tinygnn",
    packages=["tinygnn", "tinygnn.tests"],
    package_dir={
        "tinygnn":       os.path.join("python", "tinygnn"),
        "tinygnn.tests": os.path.join("python", "tinygnn", "tests"),
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    entry_points={
        "console_scripts": [
            "tinygnn-test = tinygnn.tests.smoke_tests:_main",
            "tinygnn-bench = scripts.bench_gnn:main",
        ],
    },
    zip_safe=False,
)
