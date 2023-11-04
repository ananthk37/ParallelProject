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
const char* data_gen_h2d = "data_gen_h2d";
const char* data_gen_d2h = "data_gen_d2h";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";
const char* correctness_check = "correctness_check";
const char* correctness_h2d = "correctness_h2d";
const char* correctness_d2h = "correctness_d2h";


__global__ void random_fill(float* nums, int size, const char* input_type) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (float)curand_uniform(&state) * size;
}

__global__ void sorted_fill(float* nums, int size, const char* input_type) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (float)index;
}

__global__ void reverse_fill(float* nums, int size, const char* input_type) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (float)(size - index - 1);
}

__global__ void nearly_fill(float* nums, int size, const char* input_type) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (float)curand_uniform(&state) * blockIdx.x;
}

__global__ void confirm_sorted_step(float* nums, int size, bool* sorted) {
    // __shared__ bool sorted;
    // sorted = true;
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    if(index < size - 1) {
        if(nums[index] > nums[index + 1]) {
            *sorted = false;
        }
    }

    // __syncthreads();

    // // parallel reduction to check if whole array is sorted
    // for(int i = 1; i < blockDim.x; i *= 2) {
    //     if(index % (2 * i) == 0) {
    //         sorted = sorted && __shfl_down_sync(0xFFFFFFFF, sorted, i);
    //     }
    //     __syncthreads();
    // }

    // if(index == 0) {
    //     *isSorted = sorted;
    // }
}

void fill_array(float* nums, const char* input_type) {
    float *dev_nums;
    size_t size = NUM_VALS * sizeof(float);
    
    cudaMalloc((void**) &dev_nums, size);

    //MEM COPY FROM HOST TO DEVICE
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(data_gen_h2d);
    cudaMemcpy(dev_nums, nums, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(data_gen_h2d);
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    // FILLING ARRAY
    CALI_MARK_BEGIN(data_init);
    if(strcmp(input_type, "random") == 0) {
        random_fill<<<blocks, threads>>>(dev_nums, NUM_VALS, input_type);
    }
    if(strcmp(input_type, "sorted") == 0) {
        sorted_fill<<<blocks, threads>>>(dev_nums, NUM_VALS, input_type);
    }
    if(strcmp(input_type, "reverse") == 0) {
        reverse_fill<<<blocks, threads>>>(dev_nums, NUM_VALS, input_type);
    }
    if(strcmp(input_type, "nearly") == 0) {
        nearly_fill<<<blocks, threads>>>(dev_nums, NUM_VALS, input_type);
    }
    CALI_MARK_END(data_init);

    //MEM COPY FROM DEVICE TO HOST
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(data_gen_d2h);
    cudaMemcpy(nums, dev_nums, size, cudaMemcpyDeviceToHost);
    CALI_MARK_BEGIN(data_gen_d2h);
    CALI_MARK_END(comm);

    cudaFree(dev_nums);
}

bool confirm_sorted(float* nums) {
    float *dev_nums;
    bool *dev_sorted;
    bool sorted = true;
    size_t size = NUM_VALS * sizeof(float);

    cudaMalloc((void**) &dev_nums, size);
    cudaMalloc((void**) &dev_sorted, sizeof(bool));

    //MEM COPY FROM HOST TO DEVICE
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(correctness_h2d);
    cudaMemcpy(dev_nums, nums, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sorted, &sorted, sizeof(bool), cudaMemcpyHostToDevice);
    CALI_MARK_END(correctness_h2d);
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);

    // CHECKING CORRECTNESS
    CALI_MARK_BEGIN(correctness_check);
    confirm_sorted_step<<<blocks, threads>>>(dev_nums, NUM_VALS, dev_sorted);
    cudaDeviceSynchronize();
    CALI_MARK_END(correctness_check);

    //MEM COPY FROM DEVICE TO HOST
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(correctness_d2h);
    cudaMemcpy(&sorted, dev_sorted, sizeof(bool), cudaMemcpyDeviceToHost);
    CALI_MARK_END(correctness_d2h);
    CALI_MARK_END(comm);

    cudaFree(dev_nums);
    cudaFree(dev_sorted);
    return sorted;
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    // retrieve user input
    const char* input_type = argv[1];
    THREADS = atoi(argv[2]);
    NUM_VALS = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n", BLOCKS);

    // initialize array
    float *nums = (float*) malloc(NUM_VALS * sizeof(float));
    
    // fill array
    fill_array(nums, input_type);
    cout << "Data Initialized" << endl;

    // test print array
    // for(int i = 0; i < NUM_VALS; i++) {
    //     cout << nums[i] << " ";
    // }
    // cout << endl;

    // check correctness
    if(confirm_sorted(nums)) {
        cout << "Correctness Check Passed!" << endl;
    }
    else {
        cout << "Correctness Check Failed..." << endl;
    }

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
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    free(nums);
}
  