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

const char *data_init = "data_init";
const char *data_init_h2d = "data_init_h2d";
const char *data_init_d2h = "data_init_d2h";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comp_cpu = "comp_cpu";
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comm_h2d = "comm_h2d";
const char *comm_d2h = "comm_d2h";
const char *correctness_check = "correctness_check";
const char *correctness_h2d = "correctness_h2d";
const char *correctness_d2h = "correctness_d2h";

void printArray(float *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

__global__ void random_fill(float *nums, int size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (float)curand_uniform(&state) * size;
}

__global__ void sorted_fill(float *nums)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (float)index;
}

__global__ void reverse_fill(float *nums, int size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (float)(size - index - 1);
}

__global__ void nearly_fill(float *nums, int size)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);
    if ((int)truncf(curand_uniform(&state) * (100 - 1 + 0.999999) + 1) == 1)
    {
        int swap_index = (int)truncf(curand_uniform(&state) * (size - 1 + 0.999999));
        float temp = nums[index];
        nums[index] = nums[swap_index];
        nums[swap_index] = temp;
    }
}

void fill_array(float *nums, const char *input_type)
{
    float *dev_nums;
    size_t size = NUM_VALS * sizeof(float);

    cudaMalloc((void **)&dev_nums, size);

    // MEM COPY FROM HOST TO DEVICE
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
    if (strcmp(input_type, "random") == 0)
    {
        random_fill<<<blocks, threads>>>(dev_nums, NUM_VALS);
    }
    if (strcmp(input_type, "sorted") == 0)
    {
        sorted_fill<<<blocks, threads>>>(dev_nums);
    }
    if (strcmp(input_type, "reverse") == 0)
    {
        reverse_fill<<<blocks, threads>>>(dev_nums, NUM_VALS);
    }
    if (strcmp(input_type, "nearly") == 0)
    {
        sorted_fill<<<blocks, threads>>>(dev_nums);
        nearly_fill<<<blocks, threads>>>(dev_nums, NUM_VALS);
    }
    CALI_MARK_END(data_init);

    // MEM COPY FROM DEVICE TO HOST
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(data_init_d2h);
    cudaMemcpy(nums, dev_nums, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(data_init_d2h);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    cudaFree(dev_nums);
}

__global__ void confirm_sorted_step(float *nums, int size, bool *sorted)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size - 1)
    {
        if (nums[index] > nums[index + 1])
        {
            *sorted = false;
        }
    }
}

bool confirm_sorted(float *nums)
{
    float *dev_nums;
    bool *dev_sorted;
    bool sorted = true;
    size_t size = NUM_VALS * sizeof(float);

    cudaMalloc((void **)&dev_nums, size);
    cudaMalloc((void **)&dev_sorted, sizeof(bool));

    // MEM COPY FROM HOST TO DEVICE
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

    // MEM COPY FROM DEVICE TO HOST
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

void mergeHost(float *array, int const left, int const mid,
               int const right)
{
    int const subArrayOne = mid - left + 1;
    int const subArrayTwo = right - mid;

    // Create temp arrays
    auto *leftArray = new float[subArrayOne],
         *rightArray = new float[subArrayTwo];

    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];

    auto indexOfSubArrayOne = 0, indexOfSubArrayTwo = 0;
    int indexOfMergedArray = left;

    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo)
    {
        if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo])
        {
            array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        }
        else
        {
            array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }

    // Copy the remaining elements of
    // left[], if there are any
    while (indexOfSubArrayOne < subArrayOne)
    {
        array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }

    // Copy the remaining elements of
    // right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo)
    {
        array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
    delete[] leftArray;
    delete[] rightArray;
}


__global__ void selectionSort(float *data, float *temp, int n, int chunkSize)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate the range of elements to be sorted in this launch
    int start = tid * chunkSize;
    int end = min(start + chunkSize - 1, n - 1);

    for(int i = start; i < end; i++) {
        float current_min = data[i];
        int current_min_index = i;
        for(int j = i; j <= end; j++) {
            if(data[j] < current_min) {
                current_min_index = j;
                current_min = data[j];
            }
        }
        float temp = current_min;
        data[current_min_index] = data[i];
        data[i] = temp;
    }
}

int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    // retrieve user input
    const char *input_type = argv[1];
    THREADS = atoi(argv[2]);
    NUM_VALS = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;

    printf("Number of threads: %d\n", THREADS);
    printf("Number of values: %d\n", NUM_VALS);
    printf("Number of blocks: %d\n\n", BLOCKS);

    // initialize arrays
    float *h_data = (float *)malloc(sizeof(float) * NUM_VALS); // Host array
    float *d_data, *d_temp;                                    // Device arrays

    // fill array
    fill_array(h_data, input_type);
    cout << "Data Initialized" << endl;

    // Allocate memory on the device
    cudaMalloc((void **)&d_data, sizeof(float) * NUM_VALS);
    cudaMalloc((void **)&d_temp, sizeof(float) * NUM_VALS);

    // Copy data from the host to the device
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(comm_h2d);
    cudaMemcpy(d_data, h_data, sizeof(float) * NUM_VALS, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_h2d);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    int chunkSize = 2048;
    dim3 blocks(BLOCKS, 1);
    dim3 threads(THREADS, 1);
    // Launch the CUDA kernel to perform merge sort
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    for (int i = 0; i < NUM_VALS / chunkSize; i++)
    {
        selectionSort<<<blocks, threads>>>(d_data, d_temp, NUM_VALS, chunkSize);
    }
    cudaDeviceSynchronize();

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // Copy the sorted data back to the host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(comm_d2h);
    cudaMemcpy(h_data, d_data, sizeof(float) * NUM_VALS, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_d2h);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Clean up and free memory
    cudaFree(d_data);
    cudaFree(d_temp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    CALI_MARK_BEGIN(comp_cpu);
    for (int i = 1; i < NUM_VALS / chunkSize; ++i)
    {
        mergeHost(h_data, 0, i * chunkSize - 1, std::min(((i + 1) * chunkSize - 1), NUM_VALS - 1));
    }
    CALI_MARK_END(comp_cpu);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // check correctness
    if (confirm_sorted(h_data))
    {
        cout << "Correctness Check Passed!" << endl;
    }
    else
    {
        cout << "Correctness Check Failed..." << endl;
    }
    
    for (int i = 0; i < NUM_VALS; i++)
    {
        cout << h_data[i] << " ";
    }

    adiak::init(NULL);
    adiak::launchdate();                           // launch date of the job
    adiak::libraries();                            // Libraries used
    adiak::cmdline();                              // Command line used to launch the job
    adiak::clustername();                          // Name of the cluster
    adiak::value("Algorithm", "Merge Sort");       // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA");      // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float");             // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS);           // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type);         // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_threads", NUM_VALS);         // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS);            // The number of CUDA blocks
    adiak::value("group_num", 3);                  // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI");   // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    free(h_data);
}
