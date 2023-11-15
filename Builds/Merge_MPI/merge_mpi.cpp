#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <limits.h>
#include <vector>
#include <cstring>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

// master worker implementation of merge sort

int num_procs,          /* number of processes in partition */
    rank,               /* a process identifier */
    source,             /* task id of message source */
    dest,               /* task id of message destination */
    mtype,              /* message type */
    local_size,         /* entries of array sent to each worker */
    avg, extra, offset; /* used to determine rows sent to each worker */

const char *data_init = "data_init";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *local_sort = "local_sort";
const char *merge_and_partition = "merge_and_partition";
const char *comm = "comm";
const char *comm_small = "comm_small";
const char *comm_large = "comm_large";
const char *send = "MPI_Send";
const char *recieve = "MPI_Recv";
const char* reduce = "MPI_Reduce";
const char *correctness_check = "correctness_check";

void random_fill(float *local_nums, int size)
{
    for (int i = 0; i < local_size; i++)
    {
        local_nums[i] = rand() % size;
    }
}

void sorted_fill(float *local_nums)
{
    for (int i = 0; i < local_size; i++)
    {
        local_nums[i] = offset + i;
    }
}

void reverse_fill(float *local_nums, int size)
{
    for (int i = 0; i < local_size; i++)
    {
        local_nums[i] = size - offset - i - 1;
    }
}

void nearly_fill(float *local_nums, int size)
{
    for (int i = 0; i < local_size; i++)
    {
        local_nums[i] = (rand() % size) / (size - offset - i);
    }
}

void fill_array(float *local_nums, int size, const char *input_type)
{
    // fill array
    CALI_MARK_BEGIN(data_init);
    if (strcmp(input_type, "random") == 0)
    {
        random_fill(local_nums, size);
    }
    if (strcmp(input_type, "sorted") == 0)
    {
        sorted_fill(local_nums);
    }
    if (strcmp(input_type, "reverse") == 0)
    {
        reverse_fill(local_nums, size);
    }
    if (strcmp(input_type, "nearly") == 0)
    {
        nearly_fill(local_nums, size);
    }
    CALI_MARK_END(data_init);
}

int confirm_sorted(float* nums, int size) {
    CALI_MARK_BEGIN(correctness_check);
    for(int i = 0; i < local_size; i++) {
        int index = i + offset;
        if(index < size - 1 && nums[index] > nums[index + 1]) {
            CALI_MARK_END(correctness_check);
            return 0;
        }
    }
    CALI_MARK_END(correctness_check);
    return 1;
}

void merge(float *arr, int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;

    std::vector<float> L(n1);
    std::vector<float> R(n2);

    for (int i = 0; i < n1; i++)
    {
        L[i] = arr[l + i];
    }
    for (int j = 0; j < n2; j++)
    {
        R[j] = arr[m + 1 + j];
    }

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(float *arr, int l, int r)
{
    if (l < r)
    {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

int main(int argc, char **argv)
{
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    MPI_Init(&argc, &argv);
    int rank, num_procs; // Process rank and total num of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // retrieve user input
    const char *input_type = argv[1];
    int size = atoi(argv[2]);
    float *nums = new float[size];

    fill_array(nums, size, input_type);

    // create local values
    avg = floor(size / num_procs);
    extra = size % num_procs;
    local_size = (rank < extra) ? (avg + 1) : avg;
    offset = (rank < extra) ? (rank * avg + rank) : (rank * avg + extra);
    float *local_nums = new float[local_size];
    int local_sorted = 0;
    // fill array
    fill_array(local_nums, local_size, input_type);
    if (rank == 0)
    {
        cout << "Data Initialized" << endl;
    }

    // start sort
    if (rank == 0)
    {
        // Master's own chunk
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        CALI_MARK_BEGIN(local_sort);
        mergeSort(local_nums, 0, local_size - 1);
        CALI_MARK_END(local_sort);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        // Merge sorted chunks received from workers
        std::vector<float> data(local_nums, local_nums + local_size);
        for (int i = 1; i < num_procs; i++)
        {

            std::vector<float> received_data(local_size);
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_large);
            CALI_MARK_BEGIN(recieve);
            MPI_Recv(received_data.data(), local_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            CALI_MARK_END(recieve);
            CALI_MARK_END(comm_large);
            CALI_MARK_END(comm);
            for (int j = 0; j < local_size; j++)
            {
                data.push_back(received_data[j]);
            }
            CALI_MARK_BEGIN(comp);
            CALI_MARK_BEGIN(comp_large);
            merge(data.data(), 0, local_size * i - 1, local_size * (i + 1) - 1);
            CALI_MARK_END(comp_large);
            CALI_MARK_END(comp);
        }

        delete[] nums;
        nums = data.data();
        local_sorted = confirm_sorted(nums, size);
    }
    else
    {
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        CALI_MARK_BEGIN(local_sort);
        mergeSort(local_nums, 0, local_size - 1);
        CALI_MARK_END(local_sort);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);

        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(send);
        MPI_Send(local_nums, local_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END(send);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    }
    
    

    
    if (rank == 0)
    {
        if (local_sorted == 1)
        {
            cout << "Correctness Check Passed!" << endl;
        }
        else
        {
            cout << "Correctness Check Failed..." << endl;
        }
    }

    adiak::init(NULL);
    adiak::launchdate();                           // launch date of the job
    adiak::libraries();                            // Libraries used
    adiak::cmdline();                              // Command line used to launch the job
    adiak::clustername();                          // Name of the cluster
    adiak::value("Algorithm", "Merge Sort");       // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI");       // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float");             // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size);               // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type);         // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs);          // The number of processors (MPI ranks)
    adiak::value("group_num", 3);                  // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI");   // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
    MPI_Finalize();
    delete[] local_nums;
    return 0;
}
