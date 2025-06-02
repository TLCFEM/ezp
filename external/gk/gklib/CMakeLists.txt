cmake_minimum_required(VERSION 3.21)

project(gklib C)

include_directories(.)

if (EZP_ENABLE_OPENMP_GK)
    add_compile_definitions(__OPENMP__)
    find_package(OpenMP REQUIRED COMPONENTS C)
endif ()

include(CheckIncludeFile)
check_include_file(execinfo.h HAVE_EXECINFO_H)
if (HAVE_EXECINFO_H)
    add_compile_definitions(HAVE_EXECINFO_H)
endif ()

include(CheckFunctionExists)
check_function_exists(getline HAVE_GETLINE)
if (HAVE_GETLINE)
    add_compile_definitions(HAVE_GETLINE)
endif ()

if (MSVC)
    try_compile(HAVE_TLS ${CMAKE_BINARY_DIR} check_thread_storage.c)
    if (HAVE_TLS)
        add_compile_definitions("__thread=__declspec(thread)")
    else ()
        add_compile_definitions("__thread=")
    endif ()
endif ()

add_compile_options(-w)

file(GLOB_RECURSE SRC_GK *.c)

list(REMOVE_ITEM SRC_GK check_thread_storage.c)

add_library(${PROJECT_NAME} ${SRC_GK})

if (EZP_ENABLE_OPENMP_GK)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_C)
endif ()
