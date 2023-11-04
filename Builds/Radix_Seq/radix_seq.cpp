#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <assert.h>
#include <string.h>

#include <caliper/cali.h>
#include <adiak.hpp>


//number of bits to shift for each cut
#define SHIFT_NUMBER 4
//number of buckets created
#define NUM_BUCKETS 1 << SHIFT_NUMBER
//max distance to shift
#define MAX_SHIFT sizeof(unsigned int) * 8
//constant to & with to get right number of bits
#define LAST_DIGITS 0xF
using namespace std;

const char* data_init = "data_init";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* correctness_check = "correctness_check";

void radix_sort(unsigned int* nums, unsigned int n);
void random_fill(unsigned int* nums, unsigned int n);
void counting_sort(unsigned int* nums, unsigned int n, int shift_offset);
bool verify_sorted(unsigned int* nums, unsigned int n);


void random_fill(unsigned int* nums, unsigned int n) {
    for(unsigned int i = 0; i < n; i++) {
        nums[i] = rand() % n;
    }
}
void sorted_fill(unsigned int* nums, unsigned int n) {
    for(unsigned int i = 0; i < n; i++) {
        nums[i] =  i;
    }
}
void reverse_fill(unsigned int* nums, unsigned int n) {
    for(unsigned int i = 0; i < n; i++) {
        nums[i] = n - 1 - i;
    }
}
void nearly_fill(unsigned int* nums, unsigned int n) {
    for(unsigned int i = 0; i < n; i++) {
        nums[i] = rand() % 5 + i;
    }
}

void radix_sort(unsigned int* nums, unsigned int n) {
    for(int i = 0; i < MAX_SHIFT; i += SHIFT_NUMBER) {
        counting_sort(nums, n, i);
    }

}
void counting_sort(unsigned int* nums, unsigned int n, int shift_offset) {
    queue<unsigned int> buckets[NUM_BUCKETS];
    for(unsigned int i = 0; i < n; i++) {
        //find bucket -- shift by offset, take last digits
        unsigned int which_bucket = (nums[i] >> shift_offset) & LAST_DIGITS;
        buckets[which_bucket].push(nums[i]);
    }
    unsigned int index = 0;
    for(auto bucket: buckets) {
        while(!bucket.empty()) {
            nums[index] = bucket.front();
            index++;
            bucket.pop();
        }
    }
}

bool verify_sorted(unsigned int* nums, unsigned int n) {
    for(unsigned int i = 1; i < n; i++){
        if(nums[i] < nums[i-1]) {
            cout << nums[i] << ',' << nums[i-1] << endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();

    const char* input_type = argv[1];
    unsigned int size = atoi(argv[2]);
    CALI_MARK_BEGIN(data_init);
    unsigned int* nums = new unsigned int[size];


    if(strcmp(input_type, "random") == 0) {
        random_fill(nums, size);
    }
    else if (strcmp(input_type, "sorted") == 0) {
        sorted_fill(nums, size);
    }
    else if (strcmp(input_type, "reverse") == 0) {
        reverse_fill(nums, size);
    }
    else if (strcmp(input_type, "nearly") == 0) {
        nearly_fill(nums, size);
    }
    else {
        cerr << "Invalid type" << endl;
        delete[] nums;
        exit(0);
    }
    CALI_MARK_END(data_init);
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    radix_sort(nums, size);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(correctness_check);
    bool is_sorted = verify_sorted(nums, size);
    CALI_MARK_END(correctness_check);
    delete[] nums;
    assert(is_sorted);

    // Metadata
	adiak::init(NULL);
	adiak::launchdate();
	adiak::libraries();
	adiak::cmdline();
	adiak::clustername();
    adiak::value("Algorithm", "Radix Sort");
    adiak::value("ProgrammingModel", "Sequential");
    adiak::value("Datatype", "unsigned int");
    adiak::value("SizeOfDatatype", sizeof(unsigned int));
    adiak::value("InputSize", size);
    adiak::value("InputType", input_type);
    adiak::value("group_num", 3);
    adiak::value("implementation_source", "Handwritten");

    mgr.stop();
    mgr.flush();
}