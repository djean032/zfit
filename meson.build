project('zfit', 'fortran', version: '0.0.1')

src = files(
  './src/chem.f90',
  './src/global.f90',
  './src/laser.f90',
  './src/zfit.f90',
)

minpack_dep = dependency('minpack', fallback: ['minpack', 'minpack_dep'])
odepack_dep = dependency('odepack', fallback: ['odepack', 'odepack_dep'])

noexecstack_args = [ '-Wl,-z,noexecstack' ]

project_target = shared_library(
  meson.project_name(),
  sources: src,
  dependencies: [minpack_dep, odepack_dep],
  link_args: noexecstack_args,
)
