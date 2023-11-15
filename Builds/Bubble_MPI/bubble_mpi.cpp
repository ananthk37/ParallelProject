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
	local_size,            /* number of entries sent to each worker*/
	avg, extra, offset;    /* used to determine rows sent to each worker */

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* sendrecv = "MPI_Sendrecv";
const char* bcast = "MPI_Bcast";
const char* gather = "MPI_Gather";
const char* reduce = "MPI_Reduce";
const char* barrier = "MPI_Barrier";
const char* correctness_check = "correctness_check";

void random_fill(float* local_nums, int size) {
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = rand() % size;
    }
}

void sorted_fill(float* local_nums) {
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = offset + i;
    }
}

void reverse_fill(float* local_nums, int size) {
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = size - offset - i - 1;
    }
}

void nearly_fill(float* local_nums, int size) {
    sorted_fill(local_nums);
    for(int i = 0; i < local_size; i++) {
        if(rand() % 100 == 0) {
            swap(local_nums[i], local_nums[rand() % local_size]);
        }
    }
}

void fill_array(float* local_nums, int size, const char* input_type) {
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

void merge_high(float* local_nums, float* partner_nums) {
    int local_index = local_size - 1;
    int partner_index = local_size - 1;
    float* temp_nums = new float[local_size];

    for(int i = local_size - 1; i >= 0; i--) {
        if(local_nums[local_index] > partner_nums[partner_index]) {
            temp_nums[i] = local_nums[local_index--];
        }
        else {
            temp_nums[i] = partner_nums[partner_index--];
        }
    }

    for(int i = 0; i < local_size; i++) {
        local_nums[i] = temp_nums[i];
    }

    delete[] temp_nums;
}

void merge_low(float* local_nums, float* partner_nums) {
    int local_index = 0;
    int partner_index = 0;
    float* temp_nums = new float[local_size];

    for(int i = 0; i < local_size; i++) {
        if(local_nums[local_index] < partner_nums[partner_index]) {
            temp_nums[i] = local_nums[local_index++];
        }
        else {
            temp_nums[i] = partner_nums[partner_index++];
        }
    }
    
    for(int i = 0; i < local_size; i++) {
        local_nums[i] = temp_nums[i];
    }

    delete[] temp_nums;
}

void bubble_sort(float* local_nums) {
    // sort the local data to begin
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    sort(local_nums, local_nums + local_size);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    float* partner_nums = new float[local_size];

    for(int i = 0; i < num_procs; i++) {
        // even step
        if(i % 2 == 0) {
            int partner = (proc_id % 2 == 0) ? (proc_id + 1) : (proc_id - 1);
            if(partner >= 0 && partner < num_procs) {
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_large);
                CALI_MARK_BEGIN(sendrecv);
                MPI_Sendrecv(local_nums, local_size, MPI_FLOAT, partner, EVENSTEP, partner_nums, local_size, MPI_FLOAT, partner, EVENSTEP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(sendrecv);
                CALI_MARK_END(comm_large);
                CALI_MARK_END(comm);

                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_large);
                // even rank, gets low nums
                if(proc_id % 2 == 0) {
                    merge_low(local_nums, partner_nums);
                }
                // odd rank, gets high nums
                else {
                    merge_high(local_nums, partner_nums);
                }
                CALI_MARK_END(comp_large);
                CALI_MARK_END(comp);
            }    
        }

        // odd step
        else {
            int partner = (proc_id % 2 == 0) ? (proc_id - 1) : (proc_id + 1);
            if(partner >= 0 && partner < num_procs) {
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_large);
                CALI_MARK_BEGIN(sendrecv);
                MPI_Sendrecv(local_nums, local_size, MPI_FLOAT, partner, ODDSTEP, partner_nums, local_size, MPI_FLOAT, partner, ODDSTEP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                CALI_MARK_END(sendrecv);
                CALI_MARK_END(comm_large);
                CALI_MARK_END(comm);
                
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_large);
                // even rank, gets high nums
                if(proc_id % 2 == 0) {
                    merge_high(local_nums, partner_nums);
                }
                // odd rank, gets low nums
                else {
                    merge_low(local_nums, partner_nums);
                }
                CALI_MARK_END(comp_large);
                CALI_MARK_END(comp);      
            }           
        }
    }

    delete[] partner_nums;
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

    if(proc_id == 0) {
        printf("Input Type: %s\n", input_type);
        printf("Number of processes: %d\n", num_procs);
        printf("Number of values: %d\n\n", size);
    }

    // create local values
    avg = floor(size / num_procs);
    extra = size % num_procs;
    local_size = (proc_id < extra) ? (avg + 1) : avg;
    offset = (proc_id < extra) ? (proc_id * avg + proc_id) : (proc_id * avg + extra);
    float* local_nums = new float[local_size];

    // fill array
    fill_array(local_nums, size, input_type);
    if(proc_id == 0) {
        cout << "Data Initialized" << endl;
    }

    // perform sort
    bubble_sort(local_nums);
    if(proc_id == 0) {
        cout << "Odd-Even Sort Completed" << endl;
    }

    // gather to process 0
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(gather);
    MPI_Gather(local_nums, local_size, MPI_FLOAT, nums, local_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(gather);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // broadcast to all processes
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(bcast);
    MPI_Bcast(nums, size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(bcast);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // test print
    // if(proc_id == 0) {
    //     cout << "MASTER RANK" << endl;
    //     for(int i = 0; i < size; i++) {
    //         cout << nums[i] << " ";
    //     }
    //     cout << endl;
    // }

    // correctness check
    int sorted = 1;
    int local_sorted = confirm_sorted(nums, size);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(reduce);
    MPI_Reduce(&local_sorted, &sorted, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    CALI_MARK_END(reduce);
    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    if(proc_id == 0) {
        if(sorted == 1) {
            cout << "Correctness Check Passed!" << endl;
        }
        else {
            cout << "Correctness Check Failed..." << endl;
        }
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
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    delete[] nums;
    delete[] local_nums;
}