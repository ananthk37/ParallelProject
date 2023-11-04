#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <limits.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* correctness_check = "correctness_check";

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int	num_procs,             /* number of processes in partition */
	proc_id,               /* a process identifier */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	local_size,            /* entries of array sent to each worker */
	avg, extra, offset;    /* used to determine rows sent to each worker */

void random_fill(float* nums, int size) {
    float* local_nums = new float[local_size];
    
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = rand() % size;
    }

    MPI_Gather(local_nums, local_size, MPI_FLOAT, nums, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    delete[] local_nums;
}

void sorted_fill(float* nums) {
    float* local_nums = new float[local_size];

    for(int i = 0; i < local_size; i++) {
        local_nums[i] = offset + i;
    }

    MPI_Gather(local_nums, local_size, MPI_FLOAT, nums, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    delete[] local_nums;
}

void reverse_fill(float* nums, int size) {
    float* local_nums = new float[local_size];

    for(int i = 0; i < local_size; i++) {
        local_nums[i] = size - offset - i - 1;
    }

    MPI_Gather(local_nums, local_size, MPI_FLOAT, nums, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    delete[] local_nums;
}

void nearly_fill(float* nums, int size) {
    float* local_nums = new float[local_size];

    for(int i = 0; i < local_size; i++) {
        local_nums[i] = rand() % size / (size - offset - i);
    }

    MPI_Gather(local_nums, local_size, MPI_FLOAT, nums, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    delete[] local_nums;
}

void fill_array(float* nums, int size, const char* input_type) {
    // calculate helper values for array fill
    avg = floor(size / num_procs);
    extra = size % num_procs;
    local_size = (proc_id < extra) ? (avg + 1) : avg;
    offset = (proc_id < extra) ? (proc_id * avg + proc_id) : (proc_id * avg + extra);

    // fill array
    if(strcmp(input_type, "random") == 0) {
        random_fill(nums, size);
    }
    if(strcmp(input_type, "sorted") == 0) {
        sorted_fill(nums);
    }
    if(strcmp(input_type, "reverse") == 0) {
        reverse_fill(nums, size);
    }
    if(strcmp(input_type, "nearly") == 0) {
        nearly_fill(nums, size);
    }
}

int main (int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    // get MPI info
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // retrieve user input
    const char* input_type = argv[1];
    int size = atoi(argv[2]);
    float* nums = new float[size];

    // fill array
    fill_array(nums, size, input_type);

    // test print
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
    adiak::value("Algorithm", "Odd-Even Bubble Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("group_num", 3); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    delete[] nums;
}