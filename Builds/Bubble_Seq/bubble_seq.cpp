#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* correctness_check = "correctness_check";

void random_fill(float* nums, int n) {
    for(int i = 0; i < n; i++) {
        nums[i] = rand() % n;
    }
}

void sorted_fill(float* nums, int n) {
    for(int i = 0; i < n; i++) {
        nums[i] = i;
    }
}

void reverse_fill(float* nums, int n) {
    for(int i = 0; i < n; i++) {
        nums[i] = n - i - 1;
    }
}

void nearly_fill(float* nums, int n) {

}

void bubble_sort(float* nums, int n) {
    bool sorted = false;
    for(int i = 0; i < n - 1; i++) {
        sorted = true;
        for(int j = 1; j < n - i - 1; j++) {
            if(nums[j] > nums[j + 1]) {
                swap(nums[j], nums[j + 1]);
                sorted = false;
            }
        }
        if(sorted) {
            break;
        }
    }
}

bool confirm_sorted(float* nums, int n) {
    for(int i = 0; i < n - 1; i++) {
        if(nums[i] > nums[i + 1]) {
            cout << "Correctness Check Failed..." << endl;
            return false;
        }
    }
    cout << "Correctness Check Passed!" << endl;
    return true;
}

int main (int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    // retrieve user input
    const char* input_type = argv[1];
    int size = atoi(argv[2]);
    float* nums = new float [size];

    // initialize data in array
    CALI_MARK_BEGIN(data_init);
    if(input_type == "random") {
        random_fill(nums, size);
    }
    if(input_type == "sorted") {
        sorted_fill(nums, size);
    }
    if(input_type == "reverse") {
        reverse_fill(nums, size);
    }
    if(input_type == "nearly") {
        cout << "Need to implement" << endl;
    }
    CALI_MARK_END(data_init);
    cout << "Data Initialized" << endl;

    // perform sort
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    bubble_sort(nums, size);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    cout << "Bubble Sort Completed" << endl;

    // check for correctedness
    CALI_MARK_BEGIN(correctness_check);
    confirm_sorted(nums, size);
    CALI_MARK_END(correctness_check);

    // Metadata
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Bubble Sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "Sequential"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    //adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
}