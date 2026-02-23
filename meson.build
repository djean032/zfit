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

# Compiler-specific flags
if ff.get_id() == 'gcc'
  # Legacy flags for gfortran (for fixed-form .f files)
  add_project_arguments('-std=legacy', language: 'fortran')
  add_project_arguments('-fallow-argument-mismatch', language: 'fortran')
elif ff.get_id() == 'intel-llvm' or ff.get_id() == 'intel'
  # Intel ifx/ifort: disable fast-math for correctness
  add_project_arguments('-fp-model=strict', language: 'fortran')
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
# On Windows, install to package directory for meson-python bundling
# On Linux, let meson-python handle library location for auditwheel
if host_machine.system() == 'windows'
  zfit_lib = shared_library('zfit', fortran_sources,
    dependencies: omp_dep,
    install: true,
    install_dir: py.get_install_dir(),
  )
else
  zfit_lib = shared_library('zfit', fortran_sources,
    dependencies: omp_dep,
    install: true,
  )
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
)

# Install Python files
py.install_sources(
  'zfit.py',
  'zfit_wrapper.pyi',
  subdir: '.',
)
