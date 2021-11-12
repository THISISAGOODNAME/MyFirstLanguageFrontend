cmake_minimum_required(VERSION 3.20)
project(MyFirstLanguageFrontend)

set(CMAKE_CXX_STANDARD 14)

set(LLVM_DIR "E:/theforge/llvm-project/my_build/lib/cmake/llvm")

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

set(LLVM_LINK_COMPONENTS
        Support
        Analysis)

#add_executable(MyFirstLanguageFrontend main.cpp)
add_llvm_executable(MyFirstLanguageFrontend
        main.cpp)
set_property(TARGET MyFirstLanguageFrontend PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")