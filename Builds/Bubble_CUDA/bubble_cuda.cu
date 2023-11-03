#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <adiak.hpp>

using namespace std;

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
const char* correctness_check = "correctness_check";

__global__ void random_fill(float* nums, const char* input_type, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (float)curand_uniform(&state) * size;
}

__global__ void sorted_fill(float* nums, const char* input_type, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (float)index;
}

__global__ void reverse_fill(float* nums, const char* input_type, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (float)(size - index - 1);
}

__global__ void nearly_fill(float* nums, const char* input_type, int size) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (float)curand_uniform(&state) * blockIdx.x;
}

void fill_array(float* nums, const char* input_type) {
    float *dev_nums;
    size_t size = NUM_VALS * sizeof(float);
    
    cudaMalloc((void**) &dev_nums, size);

    cudaMemcpy(dev_nums, nums, size, cudaMemcpyHostToDevice);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    if(strcmp(input_type, "random") == 0) {
        random_fill<<<blocks, threads>>>(dev_nums, input_type, NUM_VALS);
    }
    if(strcmp(input_type, "sorted") == 0) {
        sorted_fill<<<blocks, threads>>>(dev_nums, input_type, NUM_VALS);
    }
    if(strcmp(input_type, "reverse") == 0) {
        reverse_fill<<<blocks, threads>>>(dev_nums, input_type, NUM_VALS);
    }
    if(strcmp(input_type, "nearly") == 0) {
        nearly_fill<<<blocks, threads>>>(dev_nums, input_type, NUM_VALS);
    }

    cudaMemcpy(nums, dev_nums, size, cudaMemcpyDeviceToHost);

    cudaFree(dev_nums);
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

    float *nums = (float*) malloc(NUM_VALS * sizeof(float));

    CALI_MARK_BEGIN(data_init);
    fill_array(nums, input_type);
    CALI_MARK_END(data_init);
    cout << "Data Initialized" << endl;
    // test
    for(int i = 0; i < NUM_VALS; i++) {
        cout << nums[i] << " ";
    }
    cout << endl;

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Odd Even Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", NUM_VALS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    //adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    free(nums);
}
  