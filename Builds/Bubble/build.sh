#!/bin/bash

module load GCCcore/8.3.0
module load CMake/3.12.1

cmake \
    -Dcaliper_DIR=/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper \
    -Dadiak_DIR=/scratch/group/csce435-f23/Adiak/adiak/lib/cmake/adiak \
    .

make