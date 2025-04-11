cmake_minimum_required(VERSION 3.18)

project(gk C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(gklib)
add_subdirectory(metis)
add_subdirectory(parmetis)
