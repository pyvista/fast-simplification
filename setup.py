"""Setup for pymeshfix"""
from io import open as io_open
import sys
import os
import builtins

# workaround for *.toml https://github.com/pypa/pip/issues/7953
import site
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]


from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

filepath = os.path.dirname(__file__)

# Define macros for cython
macros = []
if os.name == 'nt':  # windows
    extra_compile_args = ['/openmp', '/O2', '/w', '/GS']
    extra_link_args = []
elif os.name == 'posix':  # linux org mac os
    if sys.platform == 'linux':
        extra_compile_args = ['-std=gnu++11', '-O3', '-w']
    else:  # probably mac os
        extra_compile_args = ['-O3', '-w']
else:
    raise OSError('Unsupported OS %s' % os.name)


# Check if 64-bit
if sys.maxsize > 2**32:
    macros.append(('IS64BITPLATFORM', None))


# Get version from version info
__version__ = None
version_file = os.path.join(filepath, 'fast_simplification', '_version.py')
with io_open(version_file, mode='r') as fd:
    exec(fd.read())

# readme file
readme_file = os.path.join(filepath, 'README.rst')


def needs_cython():
    """Check if cython source exist"""
    tgt_path = os.path.join('fast_simplification')
    has_cython_src = any(['_simplify.cpp' in fname for fname in os.listdir(tgt_path)])
    if not has_cython_src:
        try:
            import cython
        except:
            raise ImportError('Please install cython to build ``pymeshfix``')
    return not has_cython_src


def needs_numpy():
    """Check if cython source exist"""
    tgt_path = os.path.join('pymeshfix')
    has_cython_src = any(['_meshfix' in fname for fname in os.listdir(tgt_path)])
    return not has_cython_src


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


setup_requires = []
if needs_cython():
    setup_requires.extend(['cython'])
# if needs_numpy():
#     setup_requires.extend(['numpy'])


setup(
    name='fast_simplification',
    packages=['fast_simplification'],
    version=__version__,
    description='Wrapper around the Fast-Quadric-Mesh-Simplification library.',
    long_description=open(readme_file).read(),
    long_description_content_type='text/x-rst',
    author='Alex Kaszynski',
    author_email='akascap@gmail.com',
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
    ],
    url='https://github.com/pyvista/fast-simplification',

    # Build cython modules
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("fast_simplification._simplify",
                           ["fast_simplification/_simplify.pyx"],
                           language='c++',
                           extra_compile_args=extra_compile_args,
                           define_macros=macros)],

    keywords='fast-simplification decimation',
    # package_data={'pymeshfix/examples': ['StanfordBunny.ply',
                                         # 'planar_mesh.ply']},
    # install_requires=['numpy>1.11.0',
                      # 'pyvista>=0.30.0'],
    setup_requires=setup_requires,
)
