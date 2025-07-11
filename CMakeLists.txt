cmake_minimum_required(VERSION 3.21)

project(ezp C CXX Fortran)

if (NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    message(FATAL_ERROR "Only Linux is supported.")
endif ()

include_directories(${PROJECT_SOURCE_DIR})
if (EXISTS ${PROJECT_SOURCE_DIR}/include)
    include_directories(include)
endif ()
include_directories(mpl)

add_compile_definitions(MPICH_SKIP_MPICXX)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Werror -Wno-empty-body -Wno-format")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(-Rno-debug-disables-optimization)
    add_link_options(-Rno-debug-disables-optimization)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-Wall)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-fprofile-arcs -ftest-coverage)
        # add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fconcepts-diagnostics-depth=4>)
        link_libraries(gcov)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
        link_libraries(clang_rt.profile)
    endif ()
    option(EZP_ASAN "Enable address sanitizer (-fsanitize=address,leak,undefined -fno-omit-frame-pointer)." OFF)
    if (EZP_ASAN)
        add_compile_options(-fsanitize=address,leak,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,leak,undefined -fno-omit-frame-pointer)
    endif ()
endif ()

set(CMAKE_CXX_STANDARD 20)

option(EZP_ADD_UNDERSCORE "Use lowercase with appended underscore subroutine names." ON)
if (EZP_ADD_UNDERSCORE)
    add_compile_definitions(EZP_UNDERSCORE)
endif ()

option(EZP_ENABLE_OPENMP "Use OpenMP whenever possible." ON)
if (EZP_ENABLE_OPENMP)
    option(EZP_ENABLE_OPENMP_MUMPS "Use OpenMP for MUMPS." ON)
    option(EZP_ENABLE_OPENMP_LIS "Use OpenMP for LIS." ON)
    option(EZP_ENABLE_OPENMP_GK "Use OpenMP for GKlib." ON)
endif ()

find_package(MPI REQUIRED COMPONENTS C Fortran)
message(STATUS ${MPIEXEC_EXECUTABLE})
get_filename_component(MPI_ROOT_DIR ${MPIEXEC_EXECUTABLE} DIRECTORY)
file(GLOB MPI_VERSION_FILE
        "${MPI_ROOT_DIR}/mpichversion"
        "${MPI_ROOT_DIR}/ompi_info"
        "${MPI_ROOT_DIR}/impi_info"
)
message(STATUS "MPI_VERSION_FILE: ${MPI_VERSION_FILE}")

option(EZP_ENABLE_TBB "Use TBB." OFF)
set(MKL_LINK static)
if (NOT MKL_THREADING)
    if (EZP_ENABLE_TBB AND 0) # somehow linking tbb_thread causes segfault
        set(MKL_THREADING tbb_thread)
        message(STATUS "MKL_THREADING: ${MKL_THREADING}")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(MKL_THREADING gnu_thread)
        message(STATUS "MKL_THREADING: ${MKL_THREADING}")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        set(MKL_THREADING intel_thread)
        message(STATUS "MKL_THREADING: ${MKL_THREADING}")
    endif ()
endif ()
if (NOT MKL_MPI)
    if (MPI_VERSION_FILE MATCHES "ompi_info")
        set(MKL_MPI openmpi)
        message(STATUS "MKL_MPI: ${MKL_MPI}")
    elseif (MPI_VERSION_FILE MATCHES "mpichversion")
        set(MKL_MPI mpich)
        message(STATUS "MKL_MPI: ${MKL_MPI}")
    endif ()
endif ()

option(EZP_USE_64BIT_INT "Use 64-bit integer." OFF)
if (EZP_USE_64BIT_INT)
    add_compile_definitions(EZP_INT64)
    set(MKL_INTERFACE ilp64)
else ()
    set(MKL_INTERFACE lp64)
endif ()

option(EZP_USE_SYSTEM_LIBS "Use system libraries." OFF)

option(EZP_ENABLE_LIS "Use Lis iterative sparse solver." ON)
option(EZP_ENABLE_MUMPS "Use MUMPS direct sparse solver." ON)
if (EZP_ENABLE_LIS)
    add_subdirectory(external/lis)
endif ()
if (EZP_ENABLE_MUMPS)
    add_subdirectory(external/gk)
    add_subdirectory(external/mumps)
endif ()

find_package(MKL QUIET PATHS "/opt/intel/oneapi/mkl/latest")
if (MKL_FOUND)
    message(STATUS "Using MKL.")
    link_libraries(MKL::MKL_SCALAPACK)
    add_compile_definitions(EZP_MKL)
elseif (EZP_USE_SYSTEM_LIBS)
    find_library(SCALAPACK_LIB scalapack)
    if (SCALAPACK_LIB)
        message(STATUS "Using system libraries.")
        link_libraries(${SCALAPACK_LIB})
    else ()
        message(FATAL_ERROR "ScaLAPACK not found.")
    endif ()
    find_library(FLEX_LIB flexiblas)
    if (FLEX_LIB)
        link_libraries(${FLEX_LIB})
    endif ()
    find_library(OPENBLAS_LIB openblas)
    if (OPENBLAS_LIB)
        message(STATUS "Using OpenBLAS.")
        link_libraries(${OPENBLAS_LIB})
    endif ()
else ()
    link_directories(${PROJECT_SOURCE_DIR}/libs)
    if (MPI_VERSION_FILE MATCHES "mpichversion")
        link_libraries(scalapack-mpich)
    elseif (MPI_VERSION_FILE MATCHES "ompi_info")
        link_libraries(scalapack-openmpi)
    endif ()
    link_libraries(lapack blas gfortran)
endif ()

if (EZP_ENABLE_TBB)
    if (MKL_FOUND AND MKL_THREADING STREQUAL "tbb_thread")
        add_compile_definitions(EZP_TBB)
        message(STATUS "Using TBB.")
    else ()
        find_library(TBB_LIB tbb)
        if (TBB_LIB)
            add_compile_definitions(EZP_TBB)
            message(STATUS "Using TBB.")
            link_libraries(${TBB_LIB})
        endif ()
    endif ()
endif ()

link_libraries(MPI::MPI_C)

if (PROJECT_IS_TOP_LEVEL)
    file(GLOB EXAMPLES "*.cpp" "examples/*.cpp")
    foreach (EXAMPLE ${EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WLE)
        if (EXAMPLE_NAME MATCHES "mumps" AND NOT EZP_ENABLE_MUMPS)
            continue()
        endif ()
        if (EXAMPLE_NAME MATCHES "lis" AND NOT EZP_ENABLE_LIS)
            continue()
        endif ()
        add_executable(${EXAMPLE_NAME} ${EXAMPLE})
        if (EXAMPLE_NAME MATCHES "mumps")
            target_link_libraries(${EXAMPLE_NAME} mumps)
        elseif (EXAMPLE_NAME MATCHES "lis")
            target_link_libraries(${EXAMPLE_NAME} lis)
        endif ()
    endforeach ()

    file(GLOB EXAMPLES "tests/*.cpp")
    foreach (EXAMPLE ${EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WLE)
        if (EXAMPLE_NAME MATCHES "mumps" AND NOT EZP_ENABLE_MUMPS)
            continue()
        endif ()
        if (EXAMPLE_NAME MATCHES "lis" AND NOT EZP_ENABLE_LIS)
            continue()
        endif ()
        add_executable(${EXAMPLE_NAME} ${EXAMPLE})
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            target_compile_options(${EXAMPLE_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>)
        endif ()
        if (EXAMPLE_NAME MATCHES "mumps")
            target_link_libraries(${EXAMPLE_NAME} mumps)
        elseif (EXAMPLE_NAME MATCHES "lis")
            target_link_libraries(${EXAMPLE_NAME} lis)
        endif ()
    endforeach ()

    option(EZP_TEST "Enable testing." OFF)
    if (EZP_TEST)
        add_executable(catch2 ${EXAMPLES} include/catch2/catchy.cpp)
        target_compile_definitions(catch2 PRIVATE EZP_ENABLE_TEST)
        if (EZP_ENABLE_MUMPS)
            target_link_libraries(catch2 mumps)
        endif ()
        if (EZP_ENABLE_LIS)
            target_link_libraries(catch2 lis)
        endif ()
    endif ()
endif ()

option(EZP_STANDALONE "Enable standalone solver." OFF)
if (EZP_STANDALONE)
    file(GLOB EXAMPLES "standalone/*.cpp")
    foreach (EXAMPLE ${EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WLE)
        if (EXAMPLE_NAME MATCHES "mumps" AND NOT EZP_ENABLE_MUMPS)
            continue()
        endif ()
        if (EXAMPLE_NAME MATCHES "lis" AND NOT EZP_ENABLE_LIS)
            continue()
        endif ()
        add_executable(${EXAMPLE_NAME} ${EXAMPLE})
        if (EXAMPLE_NAME MATCHES "mumps")
            target_link_libraries(${EXAMPLE_NAME} mumps)
        elseif (EXAMPLE_NAME MATCHES "lis")
            target_link_libraries(${EXAMPLE_NAME} lis)
        endif ()
        install(TARGETS ${EXAMPLE_NAME} DESTINATION bin)
    endforeach ()
endif ()
