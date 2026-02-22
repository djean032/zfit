cmake_minimum_required(VERSION 3.15)

project(zfit VERSION 0.1.0 LANGUAGES Fortran C)

# =============================================================================
# Options
# =============================================================================
option(BUILD_PYTHON "Build Python extension" ON)
option(BUILD_TESTING "Build tests" ON)

# =============================================================================
# Find dependencies
# =============================================================================
find_package(OpenMP REQUIRED)

# =============================================================================
# Compiler-specific flags
# =============================================================================
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -fPIC -std=legacy -fallow-argument-mismatch")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -fcheck=all -fbacktrace")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -fPIC")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -check all -traceback")
endif()

# =============================================================================
# Fortran sources
# =============================================================================
set(FORTRAN_SOURCES
    src/chem.f90
    src/global.f90
    src/laser.f90
    src/zfit.f90
    src/minpack.f90
    src/opkdmain.f
    src/opkda1.f
    src/opkda2.f
)

# =============================================================================
# Build shared library
# =============================================================================
add_library(zfit SHARED ${FORTRAN_SOURCES})
target_link_libraries(zfit OpenMP::OpenMP_Fortran)

# Platform-specific configuration
if(UNIX AND NOT APPLE)
    target_link_options(zfit PRIVATE -z noexecstack)
elseif(WIN32)
    # Windows: Create DLL with exports
    set_target_properties(zfit PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
elseif(APPLE)
    # macOS: Handle RPATH
    set_target_properties(zfit PROPERTIES
        INSTALL_RPATH "@loader_path"
    )
endif()

# Set output directory
set_target_properties(zfit PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# =============================================================================
# Python extension
# =============================================================================
if(BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
    
    # Find Cython
    find_program(CYTHON_EXECUTABLE 
        NAMES cython cython3 cython.bat
        HINTS ${Python_SITELIB} ${Python_SITELIB}/bin
        DOC "Cython compiler"
    )
    
    if(NOT CYTHON_EXECUTABLE)
        message(FATAL_ERROR "Cython not found. Install with: pip install cython")
    endif()
    
    message(STATUS "Found Cython: ${CYTHON_EXECUTABLE}")
    
    # Generate C from Cython
    set(CYTHON_INPUT ${CMAKE_SOURCE_DIR}/zfit_wrapper.pyx)
    set(CYTHON_OUTPUT ${CMAKE_BINARY_DIR}/zfit_wrapper.c)
    
    add_custom_command(
        OUTPUT ${CYTHON_OUTPUT}
        COMMAND ${CYTHON_EXECUTABLE}
                -3
                -o ${CYTHON_OUTPUT}
                ${CYTHON_INPUT}
        DEPENDS ${CYTHON_INPUT}
        COMMENT "Cythonizing zfit_wrapper.pyx"
        VERBATIM
    )
    
    # Build Python extension
    add_library(zfit_wrapper MODULE ${CYTHON_OUTPUT})
    
    target_include_directories(zfit_wrapper PRIVATE 
        ${Python_INCLUDE_DIRS}
        ${Python_NumPy_INCLUDE_DIRS}
    )
    
    target_link_libraries(zfit_wrapper 
        zfit
        Python::Python
    )
    
    # Set module properties
    set_target_properties(zfit_wrapper PROPERTIES
        PREFIX ""
        POSITION_INDEPENDENT_CODE ON
    )
    
    if(UNIX AND NOT APPLE)
        set_target_properties(zfit_wrapper PROPERTIES
            INSTALL_RPATH "$ORIGIN/lib"
            BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
        )
    elseif(APPLE)
        set_target_properties(zfit_wrapper PROPERTIES
            INSTALL_RPATH "@loader_path/lib"
            BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
        )
    endif()
    
    # Platform-specific extension suffix
    if(WIN32)
        set_target_properties(zfit_wrapper PROPERTIES SUFFIX ".pyd")
    else()
        # Get Python extension suffix
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
            OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set_target_properties(zfit_wrapper PROPERTIES SUFFIX "${PYTHON_EXT_SUFFIX}")
    endif()
    
    # Output to root for development
    set_target_properties(zfit_wrapper PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    
    # Also copy libzfit to root for testing
    add_custom_command(TARGET zfit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:zfit>
            ${CMAKE_SOURCE_DIR}
        COMMENT "Copying libzfit to source directory"
    )
endif()

# =============================================================================
# Testing
# =============================================================================
if(BUILD_TESTING)
    enable_testing()
    
    # Find pytest
    find_program(PYTEST_EXECUTABLE 
        NAMES pytest pytest3
        HINTS ${Python_SITELIB}/bin
        DOC "pytest test runner"
    )
    
    if(PYTEST_EXECUTABLE)
        add_test(
            NAME pytest
            COMMAND ${PYTEST_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test -v
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        set_tests_properties(pytest PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}"
        )
    endif()
endif()

# =============================================================================
# Installation
# =============================================================================
install(TARGETS zfit
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

if(BUILD_PYTHON)
    install(TARGETS zfit_wrapper
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .
    )
    
    install(FILES 
        ${CMAKE_SOURCE_DIR}/zfit.py
        ${CMAKE_SOURCE_DIR}/zfit_wrapper.pyi
        DESTINATION .
    )
endif()
