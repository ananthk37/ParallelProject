#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm> 
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define ODDSTEP 0
#define EVENSTEP 1

using namespace std;

int	num_procs,             /* number of processes in partition */
	proc_id,               /* a process identifier */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	local_size,            /* entries of array sent to each worker */
	avg, extra, offset;    /* used to determine rows sent to each worker */

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comm_bcast = "comm_MPI_BCAST";
const char* comm_scatter = "comm_MPI_SCATTER";
const char* comm_gather = "comm_MPI_GATHER";
const char* comm_send = "comm_MPI_SEND";
const char* comm_receive = "comm_MPI_RECEIVE";
const char* data_init_MPI_GATHER = "data_init_MPI_GATHER";
const char* correctness_check = "correctness_check";
void random_fill(int* local_nums, int size) {
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = rand() % size;
    }
}

void sorted_fill(int* local_nums) {
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = offset + i;
    }
}

void reverse_fill(int* local_nums, int size) {
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = size - offset - i - 1;
    }
}

void nearly_fill(int* local_nums, int size) {
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = (rand() % size) / (size - offset - i);
    }
}


void fill_array(int* local_nums, int size, const char* input_type) {
    // fill array
    CALI_MARK_BEGIN(data_init);
    if(strcmp(input_type, "random") == 0) {
        random_fill(local_nums, size);
    }
    if(strcmp(input_type, "sorted") == 0) {
        sorted_fill(local_nums);
    }
    if(strcmp(input_type, "reverse") == 0) {
        reverse_fill(local_nums, size);
    }
    if(strcmp(input_type, "nearly") == 0) {
        nearly_fill(local_nums, size);
    }
    CALI_MARK_END(data_init);
}

void selection_sort(int* arr, int start, int end) {
    if (end <= 1)
        return;
 
 
    for (int i = start; i < start + end; i++) {
        int min_num = arr[i];
        int min_index  = i;
        for(int j = i; j < start + end; j++) {
            if(arr[j] < min_num) {
                min_num = arr[j];
                min_index = j;
            }
        }
        int temp = arr[i];
        arr[i] = arr[min_index];
        arr[min_index] = temp;
    }
}
int* merge(int* arr1, int n1, int* arr2, int n2)
{
    int* result = (int*)malloc((n1 + n2) * sizeof(int));
    int i = 0;
    int j = 0;
    int k;
 
    for (k = 0; k < n1 + n2; k++) {
        if (i >= n1) {
            result[k] = arr2[j];
            j++;
        }
        else if (j >= n2) {
            result[k] = arr1[i];
            i++;
        }
 
        else if (arr1[i] < arr2[j]) {
            result[k] = arr1[i];
            i++;
        }
        else {
            result[k] = arr2[j];
            j++;
        }
    }
    return result;
}


int confirm_sorted(float* nums, int size) {
    CALI_MARK_BEGIN(correctness_check);
    for(int i = 0; i < local_size; i++) {
        int index = i + offset;
        if(index < size - 1 && nums[index] > nums[index + 1]) {
            return 0;
        }
    }
    return 1;
    CALI_MARK_END(correctness_check);
}
int main (int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    MPI_Status status;

    // get MPI info
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // retrieve user input
    const char* input_type = argv[1];
    int size = atoi(argv[2]);
    int* nums = (int*)malloc(size * sizeof(int));

    // fill array
    fill_array(nums, size, input_type);


    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(comm_bcast);
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_bcast);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);


    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    int chunk_size = (size % num_procs == 0) ? (size / num_procs) : size / (num_procs - 1);

    int* chunk = (int*)malloc(chunk_size * sizeof(int));
    CALI_MARK_END(comp_small);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(comm_scatter);
    MPI_Scatter(nums, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_scatter);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    free(nums);
    nums = NULL;

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int own_chunk_size = (size >= chunk_size * (proc_id + 1)) ? chunk_size : (size - chunk_size * proc_id);
 
    selection_sort(chunk, 0, own_chunk_size);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

     for (int step = 1; step < num_procs; step = 2 * step) {
        if (proc_id % (2 * step) != 0) {
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_large);
            CALI_MARK_BEGIN(comm_send);
            MPI_Send(chunk, own_chunk_size, MPI_INT, proc_id - step, 0, MPI_COMM_WORLD);
            CALI_MARK_END(comm_send);
            CALI_MARK_END(comm_large);
            CALI_MARK_END(comm);
            break;
        }
        if (proc_id + step < num_procs) {
            int received_chunk_size = (size >= chunk_size * (proc_id + 2 * step)) ? (chunk_size * step) : (size - chunk_size * (proc_id + step));
            int* chunk_received;
            chunk_received = (int*)malloc( received_chunk_size * sizeof(int));
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_large);
            CALI_MARK_BEGIN(comm_receive);
            MPI_Recv(chunk_received, received_chunk_size, MPI_INT, proc_id + step, 0, MPI_COMM_WORLD, &status);
            CALI_MARK_END(comm_receive);
            CALI_MARK_END(comm_large);
            CALI_MARK_END(comm);


            CALI_MARK_BEGIN(comp);
            CALI_MARK_BEGIN(comp_large);
            nums = merge(chunk, own_chunk_size, chunk_received, received_chunk_size);
            free(chunk);
            free(chunk_received);
            chunk = nums;
            own_chunk_size = own_chunk_size + received_chunk_size;
            CALI_MARK_END(comp_large);
            CALI_MARK_END(comp);
        }
    }
 

    if(proc_id == 0) {
        cout << "MASTER RANK" << endl;
        for(int i = 0; i < size; i++) {
            cout << nums[i] << " ";
        }
        cout << endl;
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SelectionSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("group_num", 3); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    free(nums);
}