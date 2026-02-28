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

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(ROOT, "include")
SRC_DIR = os.path.join(ROOT, "src")
PYTHON_DIR = os.path.join(ROOT, "python")


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
    """Custom build_ext that sets C++17 and appropriate compiler flags."""

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
                # Static-link GCC runtime so .pyd doesn't depend on MinGW DLLs
                ext.extra_link_args = [
                    "-static-libgcc",
                    "-static-libstdc++",
                    "-Wl,-Bstatic",
                    "-lwinpthread",
                    "-Wl,-Bdynamic",
                ]
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
    version="0.1.0",
    author="Jai Ansh Bindra",
    description="TinyGNN — Zero-dependency GNN inference engine with Python bindings",
    long_description=open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    if os.path.exists(os.path.join(ROOT, "README.md"))
    else "",
    long_description_content_type="text/markdown",
    packages=["tinygnn"],
    package_dir={"tinygnn": os.path.join("python", "tinygnn")},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    python_requires=">=3.8",
    install_requires=["pybind11>=2.11", "numpy>=1.20"],
    extras_require={
        "test": ["pytest", "scipy", "torch"],
        "pyg": ["torch_geometric"],
    },
    zip_safe=False,
)
