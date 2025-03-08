cmake_minimum_required(VERSION 3.12)

project(ezp)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_CXX_STANDARD 20)

set(MKL_LINK static)
set(MKL_INTERFACE lp64)
#set(MKL_MPI openmpi)
find_package(MKL QUIET)
if (MKL_FOUND)
    message(STATUS "Using MKL")
    link_libraries(MKL::MKL_SCALAPACK)
endif ()

find_package(MPI REQUIRED)

if (NOT MKL_FOUND)
    link_directories(${PROJECT_SOURCE_DIR}/libs)
    message(STATUS ${MPIEXEC_EXECUTABLE})
    string(TOLOWER ${MPI_C_INCLUDE_DIRS} MPI_PATH)
    if (MPI_PATH MATCHES "openmpi")
        message(STATUS "Using OpenMPI")
        link_libraries(scalapack-openmpi)
    else ()
        message(STATUS "Using MPICH")
        link_libraries(scalapack-mpich)
    endif ()
    link_libraries(openblas gfortran gomp)
endif ()

link_libraries(MPI::MPI_C MPI::MPI_CXX)

add_compile_options(-fconcepts-diagnostics-depth=4)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fprofile-arcs -ftest-coverage)
    link_libraries(gcov)
endif ()

option(EZP_ASAN "Enable asan." OFF)

if (EZP_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

file(GLOB EXAMPLES "*.cpp" "examples/*.cpp")

foreach (EXAMPLE ${EXAMPLES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WLE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE})
endforeach ()

option(EZP_TEST "Enable testing." OFF)

if (EZP_TEST)
    add_compile_definitions(EZP_ENABLE_TEST)
endif ()

file(GLOB EXAMPLES "tests/*.cpp")

if (EZP_TEST)
    add_executable(catch2 ${EXAMPLES} include/catch2/catchy.cpp)
else ()
    foreach (EXAMPLE ${EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WLE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE})
    endforeach ()
endif ()

option(EZP_STANDALONE "Enable standalone solver." OFF)

if (EZP_STANDALONE)
    include_directories(mpl)
    file(GLOB EXAMPLES "standalone/*.cpp")
    foreach (EXAMPLE ${EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WLE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE})
    endforeach ()
endif ()
