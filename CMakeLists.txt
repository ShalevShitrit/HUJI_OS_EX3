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
##        "Test/test1-1_thread_1_process.cpp"
#        Test/test4-1_thread_4_process.cpp
##        "ref.cpp"
#
#)
## Find and link the pthread library
#find_package(Threads REQUIRED)
#target_link_libraries(HUJI_OS_EX3 ${CMAKE_THREAD_LIBS_INIT})

cmake_minimum_required(VERSION 3.16.3)

project(HUJI_OS_EX3 VERSION 1.0 LANGUAGES C CXX)

# Set C++ standard to 11
set(CMAKE_CXX_STANDARD 11)

# Enable ThreadSanitizer for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=thread")

# Ensure system has pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# FetchContent added in CMake 3.11
include(FetchContent)

# Download GoogleTest
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)

# Make GoogleTest available
FetchContent_MakeAvailable(googletest)

# Add your library
add_library(MapReduceFramework
        MapReduceClient.h
        MapReduceFramework.cpp
        MapReduceFramework.h
        Barrier/Barrier.cpp
        Barrier/Barrier.h

)
include_directories(googletest)
# Set properties for the library
set_property(TARGET MapReduceFramework PROPERTY CXX_STANDARD 11)
target_compile_options(MapReduceFramework PUBLIC -Wall -Wextra)
target_include_directories(MapReduceFramework PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link pthreads to your framework
target_link_libraries(MapReduceFramework PUBLIC Threads::Threads)

# Add your executable
add_executable(HUJI_OS_EX3
        mattanTests/SampleClient.cpp
)

# Link the framework and pthreads to your executable
target_link_libraries(HUJI_OS_EX3 PRIVATE MapReduceFramework Threads::Threads)

# Link GoogleTest to your executable (if needed for tests)
target_link_libraries(HUJI_OS_EX3 PRIVATE gtest gtest_main)

# Add subdirectory for additional tests (if any)
