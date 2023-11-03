#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// #include <caliper/cali.h>
// #include <adiak.hpp>
#define SHIFT_NUMBER 4
#define MAX_SHIFT sizeof(unsigned int)
using namespace std;

void random_fill(unsigned int* nums, unsigned int n) {
    for(unsigned int i = 0; i < n; i++) {
        nums[i] = rand() % n;
    }
}

void radix_sort(unsigned int* nums, int n) {
    for(int i = 0; i < MAX_SHIFT; i += SHIFT_NUMBER) {

    }

}


int main(int argc, char** argv) {
    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
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

    random_fill(nums, size);
    radix_sort(nums, size);
}