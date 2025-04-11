cmake_minimum_required(VERSION 3.18)

project(metis C)

include_directories(.)
include_directories(../gklib)
include_directories(../include)

add_compile_options(-w)

file(GLOB_RECURSE SRC_METIS *.c)

add_library(${PROJECT_NAME} ${SRC_METIS})

target_link_libraries(${PROJECT_NAME} gklib)
