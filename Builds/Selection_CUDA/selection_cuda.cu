#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <limits>

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
const char* data_init_h2d = "data_init_h2d";
const char* data_init_d2h = "data_init_d2h";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp_h2d = "comp_h2d";
const char* comp_d2h = "comp_d2h";
const char* correctness_check = "correctness_check";
const char* correctness_h2d = "correctness_h2d";
const char* correctness_d2h = "correctness_d2h";


__global__ void random_fill(unsigned int* nums, int size, const char* input_type) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (unsigned int)curand(&state) * size;
}

__global__ void sorted_fill(unsigned int* nums, int size, const char* input_type) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (unsigned int)index;
}

__global__ void reverse_fill(unsigned int* nums, int size, const char* input_type) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (unsigned int)(size - index - 1);
}

__global__ void nearly_fill(unsigned int* nums, int size, const char* input_type) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (unsigned int)(curand(&state) % blockDim.x) + blockIdx.x;
}


void fill_array(unsigned int* nums, const char* input_type) {
    unsigned int *dev_nums;
    size_t size = NUM_VALS * sizeof(unsigned int);
    
    cudaMalloc((void**) &dev_nums, size);

    //MEM COPY FROM HOST TO DEVICE
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(data_init_h2d);
    cudaMemcpy(dev_nums, nums, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(data_init_h2d);
    CALI_MARK_END(comm_large);
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
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(data_init_d2h);
    cudaMemcpy(nums, dev_nums, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(data_init_d2h);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    cudaFree(dev_nums);
}



__global__ void find_min_step(unsigned int* nums, unsigned int* min_num, int size, int index) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id >= index && id < size) {
        atomicMin(min_num, nums[id]);
    }
}

__global__ void find_index_step(unsigned int*nums, unsigned int* min_num, int* min_index, int size, int index) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if(id >= index && id < size) {
        if(*min_num == nums[id]) {
            *min_index = id;
        }
    }
}
__global__ void swap_step(unsigned int*nums, int* min_index, int index) {
    unsigned int temp = nums[index];
    nums[index] = nums[*min_index];
    nums[*min_index] = temp;
}
void print_list(unsigned int* nums) {
    for(unsigned int i = 0; i < NUM_VALS; i++) {
	cout << nums[i] << " ";
    }
    cout << endl;
}
void selection_sort(unsigned int* nums) {
    unsigned int *dev_nums, *min_num;
    int* min_index;
    size_t size = NUM_VALS * sizeof(unsigned int);
    unsigned int max_int = numeric_limits<unsigned int>::max();
    cudaMalloc((void**)& min_num, sizeof(unsigned int));
    cudaMalloc((void**)&min_index, sizeof(int));
    cudaMalloc((void**) &dev_nums, size);



    //MEM COPY FROM HOST TO DEVICE
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(comp_h2d);
    cudaMemcpy(dev_nums, nums, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(comp_h2d);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);
    dim3 single(1, 1);

    // SORT
    for(int i = 0; i < NUM_VALS; i++) {
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(comm_h2d);
        cudaMemcpy(min_num, &max_int, sizeof(unsigned int), cudaMemcpyHostToDevice);
        CALI_MARK_END(comm_h2d);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        find_min_step<<<blocks, threads>>>(dev_nums, min_num, NUM_VALS, i);
        find_index_step<<<blocks, threads>>>(dev_nums, min_num, min_index, NUM_VALS, i);
	    swap_step<<<single, single>>>(dev_nums, min_index, i);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
    }

    //MEM COPY FROM DEVICE TO HOST
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(comp_d2h);
    cudaMemcpy(nums, dev_nums, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comp_d2h);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    cudaFree(dev_nums);
}

__global__ void confirm_sorted_step(unsigned int* nums, int size, bool* sorted) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < size - 1) {
        if(nums[index] > nums[index + 1]) {
            *sorted = false;
        }
    }
}

bool confirm_sorted(unsigned int* nums) {
    unsigned int *dev_nums;
    bool *dev_sorted;
    bool sorted = true;
    size_t size = NUM_VALS * sizeof(unsigned int);

    cudaMalloc((void**) &dev_nums, size);
    cudaMalloc((void**) &dev_sorted, sizeof(bool));

    //MEM COPY FROM HOST TO DEVICE
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(correctness_h2d);
    cudaMemcpy(dev_nums, nums, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sorted, &sorted, sizeof(bool), cudaMemcpyHostToDevice);
    CALI_MARK_END(correctness_h2d);
    CALI_MARK_END(comm_large);
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
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(correctness_d2h);
    cudaMemcpy(&sorted, dev_sorted, sizeof(bool), cudaMemcpyDeviceToHost);
    CALI_MARK_END(correctness_d2h);
    CALI_MARK_END(comm_large);
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
    printf("Number of blocks: %d\n\n", BLOCKS);

    // initialize array
    unsigned int *nums = (unsigned int*) malloc(NUM_VALS * sizeof(unsigned int));
    
    // fill array
    fill_array(nums, input_type);
    cout << "Data Initialized" << endl;

    // sort array
    selection_sort(nums);
    cout << "Odd-Even Sort Completed" << endl;

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
    adiak::value("Algorithm", "Odd-Even selection Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "unsigned int"); // The datatype of input elements (e.g., double, int, unsigned int)
    adiak::value("SizeOfDatatype", sizeof(unsigned int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", NUM_VALS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 3); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    free(nums);
}
  