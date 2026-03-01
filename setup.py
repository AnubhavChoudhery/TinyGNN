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
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-fPIC",
                    "-DNDEBUG",
                ]
                if ct == "mingw32":
                    # Static-link GCC runtime so .pyd doesn't depend on MinGW DLLs
                    ext.extra_link_args = [
                        "-static-libgcc",
                        "-static-libstdc++",
                        "-Wl,-Bstatic",
                        "-lwinpthread",
                        "-Wl,-Bdynamic",
                    ]
                else:
                    ext.extra_link_args = []
            elif ct == "msvc":
                ext.extra_compile_args = [
                    "/std:c++17",
                    "/O2",
                    "/W4",
                    "/DNDEBUG",
                    "/EHsc",
                ]
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
    version="0.1.3",
    author="Jai Ansh Singh Bindra and Anubhav Choudhery (under JBAC EdTech)",
    description="TinyGNN — Zero-dependency C++17 GNN inference engine with Python bindings",
    long_description=open("README.md", encoding="utf-8").read()
    if os.path.exists("README.md")
    else "",
    long_description_content_type="text/markdown",
    url="https://github.com/JaiAnshSB/TinyGNN",
    license="MIT",
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
    python_requires=">=3.8",
    install_requires=["pybind11>=2.11", "numpy>=1.20"],
    extras_require={
        "test": ["pytest", "scipy", "torch"],
        "pyg": ["torch_geometric"],
    },
    zip_safe=False,
)
