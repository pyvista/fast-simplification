"""Setup for fast-simplification."""

import builtins
from io import open as io_open
import os
import platform
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext
from wheel.bdist_wheel import bdist_wheel

filepath = os.path.dirname(__file__)

# Define macros for cython
macros = []
ext_kwargs = {}
setup_kwargs = {"cmdclass": {}}
if os.name == "nt":  # windows
    extra_compile_args = ["/openmp", "/O2", "/w", "/GS"]
elif os.name == "posix":  # linux org mac os
    if sys.platform == "linux":
        extra_compile_args = ["-std=gnu++11", "-O3", "-w"]
    else:  # probably mac os
        extra_compile_args = ["-std=c++11", "-O3", "-w"]
else:
    raise OSError("Unsupported OS %s" % os.name)


# Check if 64-bit
if sys.maxsize > 2**32:
    macros.append(("IS64BITPLATFORM", None))


# https://github.com/joerick/python-abi3-package-sample/blob/main/setup.py
class bdist_wheel_abi3(bdist_wheel):  # noqa: D101
    def get_tag(self):  # noqa: D102
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            return "cp311", "abi3", plat

        return python, abi, plat


if sys.version_info.minor >= 11 and platform.python_implementation() == "CPython":
    # Can create an abi3 wheel (typed memoryviews first available in 3.11)!
    macros.append(("Py_LIMITED_API", "0x030B0000"))
    ext_kwargs["py_limited_api"] = True
    setup_kwargs["cmdclass"]["bdist_wheel"] = bdist_wheel_abi3


# Get version from version info
__version__ = None
version_file = os.path.join(filepath, "fast_simplification", "_version.py")
with io_open(version_file, mode="r") as fd:
    exec(fd.read())

# readme file
readme_file = os.path.join(filepath, "README.rst")


# for: the cc1plus: warning: command line option '-Wstrict-prototypes'
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process:
        try:
            del builtins.__NUMPY_SETUP__
        except AttributeError:
            pass
        import numpy

        self.include_dirs.append(numpy.get_include())

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        _build_ext.build_extensions(self)


setup_kwargs["cmdclass"]["build_ext"] = build_ext

setup(
    name="fast_simplification",
    packages=["fast_simplification"],
    version=__version__,
    description="Wrapper around the Fast-Quadric-Mesh-Simplification library.",
    long_description=open(readme_file).read(),
    long_description_content_type="text/x-rst",
    author="Alex Kaszynski",
    author_email="akascap@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    url="https://github.com/pyvista/fast-simplification",
    python_requires=">=3.9",
    # Build cython modules
    ext_modules=[
        Extension(
            "fast_simplification._simplify",
            ["fast_simplification/_simplify.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
            define_macros=macros,
            **ext_kwargs,
        ),
        Extension(
            "fast_simplification._replay",
            ["fast_simplification/_replay.pyx"],
            language="c++",
            extra_compile_args=extra_compile_args,
            define_macros=macros,
            **ext_kwargs,
        ),
    ],
    keywords="fast-simplification decimation",
    install_requires=["numpy"],
    **setup_kwargs,
)
