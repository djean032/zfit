project('odepack', 'fortran', default_options: ['fortran_std=legacy'])
project_source_files = ['./src/opkdmain.f', './src/opkda1.f', './src/opkda2.f']
project_target = shared_library(meson.project_name(), project_source_files, install: false)
project_dep = declare_dependency(link_with: project_target)
set_variable(meson.project_name() + '_dep', project_dep)
