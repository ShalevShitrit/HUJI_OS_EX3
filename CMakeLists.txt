#cmake_minimum_required(VERSION 3.16.3)
#project(HUJI_OS_EX3)
#
#set(CMAKE_CXX_STANDARD 11)
#
#include_directories(.)
#SET(GCC_COVERAGE_COMPILE_FLAGS)
#add_executable(HUJI_OS_EX3
#        MapReduceClient.h
#        MapReduceFramework.cpp
#        MapReduceFramework.h
#        "Barrier/Barrier.cpp"
#        "Barrier/Barrier.h"
#        "Test/test1-1_thread_1_process.cpp"
##        Test/test4-1_thread_4_process.cpp
##        "ref.cpp"
#
#)
## Find and link the pthread library
#find_package(Threads REQUIRED)
#target_link_libraries(HUJI_OS_EX3 ${CMAKE_THREAD_LIBS_INIT})
#
cmake_minimum_required(VERSION 3.1)


# NOTE: You can't have both ThreadSanitizer and AddressSanitizer enabled at the same time.

## Uncomment the following to enable ThreadSanitizer.
#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")

# Uncomment the following to enable AddressSanitizer.
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")


# Project configuration
project(ex3 VERSION 1.0 LANGUAGES C CXX)


# Ensure system has pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(MapReduceFramework
        MapReduceClient.h
        "Barrier/barrier.cpp"
        "Barrier/barrier.h"
        MapReduceFramework.cpp MapReduceFramework.h

        # ------------- Add your own .h/.cpp files here -------------------
)


set_property(TARGET MapReduceFramework PROPERTY CXX_STANDARD 11)
# to include debugging information, add the flags `-ggdb -g3`, to compile an optimized build add `-O3`
target_compile_options(MapReduceFramework PUBLIC -Wall -Wextra)
target_include_directories(MapReduceFramework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# link pthreads to your framework
target_link_libraries(MapReduceFramework PUBLIC Threads::Threads)

# Add tests
add_subdirectory(mattanTests)