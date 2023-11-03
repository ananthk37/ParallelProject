#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <queue>
#include <assert.h>

// #include <caliper/cali.h>
// #include <adiak.hpp>

//number of bits to shift for each cut
#define SHIFT_NUMBER 4
//number of buckets created
#define NUM_BUCKETS 1 << SHIFT_NUMBER
//max distance to shift
#define MAX_SHIFT sizeof(unsigned int) * 8
//constant to & with to get right number of bits
#define LAST_DIGITS 0xF
using namespace std;

void radix_sort(unsigned int* nums, unsigned int n);
void random_fill(unsigned int* nums, unsigned int n);
void counting_sort(unsigned int* nums, unsigned int n, int shift_offset);
bool verify_sorted(unsigned int* nums, unsigned int n);


void random_fill(unsigned int* nums, unsigned int n) {
    for(unsigned int i = 0; i < n; i++) {
        nums[i] = rand() % n;
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
    unsigned int size = atoi(argv[1]);
    unsigned int* nums = new unsigned int[size];

    // Metadata
	// adiak::init(NULL);
	// adiak::user();
	// adiak::launchdate();
	// adiak::libraries();
	// adiak::cmdline();
	// adiak::clustername();
	// adiak::value("threads", num_threads);
	// adiak::value("array_size", size);
    cout << MAX_SHIFT << endl;
    random_fill(nums, size);
    radix_sort(nums, size);
    assert(verify_sorted(nums, size));
    cout << "sorted succesfully" << endl;
    delete[] nums;
}