cmake_minimum_required(VERSION 3.18)

project(lis C)

set(CMAKE_C_STANDARD 11)

file(GLOB_RECURSE lis_src **/*.c)

include_directories(include)

add_compile_definitions(USE_COMPLEX USE_VEC_COMP USE_MPI)

if (EZP_USE_64BIT_INT)
    add_compile_definitions(_LONG__LONG)
endif ()

include(CheckIncludeFile)
check_include_file(complex.h HAVE_COMPLEX_H)
if (HAVE_COMPLEX_H)
    add_compile_definitions(HAVE_COMPLEX_H)
endif ()
check_include_file(malloc.h HAVE_MALLOC_H)
if (HAVE_MALLOC_H)
    add_compile_definitions(HAVE_MALLOC_H)
endif ()
check_include_file(quadmath.h HAVE_QUADMATH_H)
if (HAVE_QUADMATH_H)
    add_compile_definitions(HAVE_QUADMATH_H)
endif ()

add_compile_options(-w)

add_library(${PROJECT_NAME} STATIC ${lis_src})

if (EZP_ENABLE_OPENMP_LIS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        target_compile_options(${PROJECT_NAME} PRIVATE -qopenmp)
    endif ()
endif ()
