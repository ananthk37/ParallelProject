cmake_minimum_required(VERSION 3.12)

find_package(caliper REQUIRED)
find_package(adiak REQUIRED)
find_package(MPI REQUIRED)

add_executable(selection_seq selection_seq.cpp)

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(selection_seq PRIVATE caliper)
target_link_libraries(selection_seq PRIVATE MPI::MPI_CXX)
