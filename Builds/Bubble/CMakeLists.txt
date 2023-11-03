cmake_minimum_required(VERSION 3.12)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

add_executable(bubble_seq bubble_seq.cpp)
target_link_libraries(bubble_seq PRIVATE caliper)

message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})