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
const char *comm = "comm";
const char *comm_large = "comm_large";
const char *comp_h2d = "comp_h2d";
const char *comp_d2h = "comp_d2h";
const char *correctness_check = "correctness_check";
const char *correctness_h2d = "correctness_h2d";
const char *correctness_d2h = "correctness_d2h";

__global__ void random_fill(float *nums, int size, const char *input_type)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (float)curand_uniform(&state) * size;
}

__global__ void sorted_fill(float *nums, int size, const char *input_type)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (float)index;
}

__global__ void reverse_fill(float *nums, int size, const char *input_type)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    nums[index] = (float)(size - index - 1);
}

__global__ void nearly_fill(float *nums, int size, const char *input_type)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    curandState state;
    curand_init(1, index, 0, &state);

    nums[index] = (float)curand_uniform(&state) * blockIdx.x;
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
        random_fill<<<blocks, threads>>>(dev_nums, NUM_VALS, input_type);
    }
    if (strcmp(input_type, "sorted") == 0)
    {
        sorted_fill<<<blocks, threads>>>(dev_nums, NUM_VALS, input_type);
    }
    if (strcmp(input_type, "reverse") == 0)
    {
        reverse_fill<<<blocks, threads>>>(dev_nums, NUM_VALS, input_type);
    }
    if (strcmp(input_type, "nearly") == 0)
    {
        nearly_fill<<<blocks, threads>>>(dev_nums, NUM_VALS, input_type);
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

__device__ void merge(float *data, float *temp, int left, int mid, int right)
{
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right)
    {
        if (data[i] <= data[j])
        {
            temp[k] = data[i];
            i++;
        }
        else
        {
            temp[k] = data[j];
            j++;
        }
        k++;
    }

    while (i <= mid)
    {
        temp[k] = data[i];
        i++;
        k++;
    }

    while (j <= right)
    {
        temp[k] = data[j];
        j++;
        k++;
    }

    for (int x = left; x <= right; x++)
    {
        data[x] = temp[x];
    }
}

__global__ void mergeSort(float *data, float *temp, int n)
{
    for (int currSize = 1; currSize < n; currSize *= 2)
    {
        for (int left = 0; left < n - 1; left += 2 * currSize)
        {
            int mid = min(left + currSize - 1, n - 1);
            int right = min(left + 2 * currSize - 1, n - 1);
            merge(data, temp, left, mid, right);
        }
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
    cudaMemcpy(d_data, h_data, sizeof(float) * NUM_VALS, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to perform merge sort
    mergeSort<<<1, 1>>>(d_data, d_temp, NUM_VALS);
    cudaDeviceSynchronize();

    // Copy the sorted data back to the host
    cudaMemcpy(h_data, d_data, sizeof(float) * NUM_VALS, cudaMemcpyDeviceToHost);

    // Clean up and free memory
    cudaFree(d_data);
    cudaFree(d_temp);

    // Perform any other necessary operations with the sorted data

    // check correctness
    if (confirm_sorted(h_data))
    {
        cout << "Correctness Check Passed!" << endl;
    }
    else
    {
        cout << "Correctness Check Failed..." << endl;
    }

    free(h_data);
    return 0;
}
