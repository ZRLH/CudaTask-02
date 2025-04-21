#cmake_minimum_required(VERSION 3.30)
#project(cuda_test CUDA)
#
#set(CMAKE_CUDA_STANDARD 14)
#
#add_executable(cuda_test main.cu)
#
#set_target_properties(cuda_test PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

#cmake_minimum_required(VERSION 3.18)
#set(CMAKE_VERBOSE_MAKEFILE ON)
#
#project(cuda_test LANGUAGES CXX CUDA)
#
#set(CMAKE_CUDA_STANDARD 14)
#set(CMAKE_CUDA_ARCHITECTURES 86)
#
#include_directories("E:/cuda_manual/include")
#link_directories("E:/cuda_manual/lib/x64")
#
#add_executable(cuda_test main.cu)
#
#target_link_libraries(cuda_test cudart)


cmake_minimum_required(VERSION 3.18)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(cuda_test LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories("E:/cuda_manual/include")
link_directories("E:/cuda_manual/lib/x64")

add_executable(cuda_test main.cu
        main_public.cu
        main_public_private.cu)

target_link_libraries(cuda_test cudart)

set_target_properties(cuda_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(cuda_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(cuda_test cudart)

set_target_properties(cuda_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)