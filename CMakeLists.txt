cmake_minimum_required(VERSION 3.24)
project(HUJI_OS_EX3)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(HUJI_OS_EX3
        "Sample Client/SampleClient.cpp"
        MapReduceClient.h
        MapReduceFramework.h
        MapReduceFramework.cpp)
