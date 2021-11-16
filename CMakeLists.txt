cmake_minimum_required(VERSION 3.20)
project(MyFirstLanguageFrontend)

set(CMAKE_CXX_STANDARD 14)

#set(LLVM_DIR "E:/theforge/llvm-project/my_build/lib/cmake/llvm")
find_path(LLVM_DIR
    NAMES AddLLVM.cmake
    PATHS
    "${LLVM_ROOT}"
    "E:/theforge/llvm-project/my_build/lib/cmake/llvm"
    "E:/workspace/llvm-project-llvmorg-13.0.0/my_build/lib/cmake/llvm"
    NO_DEFAULT_PATH
    )

find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

set(LLVM_LINK_COMPONENTS
    Core
    Support
    OrcJIT
    native
    all)

#add_executable(MyFirstLanguageFrontend main.cpp)
add_llvm_executable(MyFirstLanguageFrontend
        MyFirstLanguageFrontend/main.cpp)
set_property(TARGET MyFirstLanguageFrontend PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

add_llvm_executable(klc
        klc/klc.cpp)
set_property(TARGET klc PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

add_executable(klctest
        tests/NativeCompile/testfib.cpp tests/NativeCompile/output.o)
