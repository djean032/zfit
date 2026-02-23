project('zfit', 'c', 'fortran', 'cython',
  version: '0.1.0',
  meson_version: '>=1.1.0',
  default_options: [
    'buildtype=release',
    'c_std=c99',
  ]
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
np_dep = dependency('numpy')

# Fortran compiler setup
ff = meson.get_compiler('fortran')

# Legacy flags for gfortran (for fixed-form .f files)
if ff.get_id() == 'gcc'
  add_project_arguments('-std=legacy', language: 'fortran')
  add_project_arguments('-fallow-argument-mismatch', language: 'fortran')
endif

# OpenMP
omp_dep = dependency('openmp', required: false)

# Fortran sources
fortran_sources = files(
  'src/chem.f90',
  'src/global.f90',
  'src/laser.f90',
  'src/zfit.f90',
  'src/minpack.f90',
  'src/opkdmain.f',
  'src/opkda1.f',
  'src/opkda2.f',
)

# Build shared library
zfit_lib = shared_library('zfit', fortran_sources,
  dependencies: omp_dep,
  install: true,
  install_dir: py.get_install_dir(),
)

# RPATH for finding libzfit at runtime
if host_machine.system() == 'linux' or host_machine.system() == 'darwin'
  rpath_args = ['-Wl,-rpath,$ORIGIN']
else
  rpath_args = []
endif

# Cython extension
cython = find_program('cython')
cython_sources = custom_target('zfit_wrapper_c',
  input: 'zfit_wrapper.pyx',
  output: 'zfit_wrapper.c',
  command: [cython, '-3', '-o', '@OUTPUT@', '@INPUT@'],
)

py.extension_module('zfit_wrapper', cython_sources,
  dependencies: [py_dep, np_dep],
  link_with: zfit_lib,
  install: true,
  link_args: rpath_args,
)

# Install Python files
py.install_sources(
  'zfit.py',
  'zfit_wrapper.pyi',
  subdir: '.',
)
