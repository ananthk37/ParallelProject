#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
const char* correctness_check = "correctness_check";

__global__ void fill_array(float* nums, const char* input_type, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(input_type == "random") {
        nums[index] = rand() % size;
    }
    if(input_type == "sorted") {
        nums[index] = index;
    }
    if(input_type == "reverse") {
        nums[index] = size - index - 1;
    }
}

int main(int argc, char *argv[]) {
    const char* input_type = argv[1];
    THREADS = atoi(argv[2]);
    NUM_VALS = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    float* nums;
    cudaMalloc((void**)&nums, sizeof(float) * NUM_VALS);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    fill_array<<<blocks, threads>>>(nums, input_type, NUM_VALS);
    // test
    for(float f : nums) {
        cout << f << " ";
    }
    cout << endl;

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", NUM_VALS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    //adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
}
  