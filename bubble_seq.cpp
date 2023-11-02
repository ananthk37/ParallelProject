#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// #include <caliper/cali.h>
// #include <adiak.hpp>

using namespace std;

void random_fill(float* nums, int n) {
    for(int i = 0; i < n; i++) {
        nums[i] = rand() % n;
    }
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

int main(int argc, char** argv) {
    int size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    float* nums = new float [size];

    // Metadata
	adiak::init(NULL);
	adiak::user();
	adiak::launchdate();
	adiak::libraries();
	adiak::cmdline();
	adiak::clustername();
	adiak::value("threads", num_threads);
	adiak::value("array_size", size);

    random_fill(nums, size);
    bubble_sort(nums, size);
}