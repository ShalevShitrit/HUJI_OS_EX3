cmake_minimum_required(VERSION 3.24)
project(HUJI_OS_EX3)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(HUJI_OS_EX3
        "Sample Client/SampleClient.cpp"
        MapReduceClient.h
        "Sample Client/MapReduceFramework.h"
        MapReduceFramework.cpp
        "Barrier/Barrier.cpp"
        "Barrier/Barrier.h"
        Tests/test1-1_thread_1_process.cpp

)
