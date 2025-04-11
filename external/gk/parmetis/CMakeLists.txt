cmake_minimum_required(VERSION 3.18)

project(parmetis C)

include_directories(.)
include_directories(../gklib)
include_directories(../include)

find_package(MPI REQUIRED COMPONENTS C)
link_libraries(MPI::MPI_C)

add_compile_options(-w)

file(GLOB_RECURSE SRC_PARMETIS *.c)

add_library(${PROJECT_NAME} ${SRC_PARMETIS})

target_link_libraries(${PROJECT_NAME} metis)
