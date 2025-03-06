cmake_minimum_required(VERSION 3.12)
project(playground)

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 20)

set(MKL_LINK static)
set(MKL_INTERFACE lp64)
set(MKL_MPI openmpi)
find_package(MKL QUIET)
if (MKL_FOUND)
    message(STATUS "Using MKL")
    link_libraries(MKL::MKL_SCALAPACK)
else ()
    link_directories(${PROJECT_SOURCE_DIR}/libs)
    link_libraries(scalapack-openmpi openblas gfortran gomp)
endif ()

find_package(MPI REQUIRED)

link_libraries(MPI::MPI_C MPI::MPI_CXX)

file(GLOB EXAMPLES "*.cpp" "examples/*.cpp")

foreach (EXAMPLE ${EXAMPLES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME_WLE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE})
endforeach ()
