#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

using namespace std;

const char *data_init = "data_init";
const char *comp = "comp";
const char *comp_large = "comp_large";
const char *comm = "comm";
const char *correctness_check = "correctness_check";



// data fill algos
void random_fill(float *nums, int n)
{
    for (int i = 0; i < n; i++)
    {
        nums[i] = rand() % n;
    }
}

void sorted_fill(float *nums, int n)
{
    for (int i = 0; i < n; i++)
    {
        nums[i] = i;
    }
}

void reverse_fill(float *nums, int n)
{
    for (int i = 0; i < n; i++)
    {
        nums[i] = n - i - 1;
    }
}

void nearly_fill(float *nums, int n)
{
    for (int i = 0; i < n; i++)
    {
        nums[i] = (rand() % n) / (n - i);
    }
}

void fill_array(float *nums, int size, const char* input_type)
{
    CALI_MARK_BEGIN(data_init);
    if (strcmp(input_type, "random") == 0)
    {
        random_fill(nums, size);
    }
    if (strcmp(input_type, "sorted") == 0)
    {
        sorted_fill(nums, size);
    }
    if (strcmp(input_type, "reverse") == 0)
    {
        reverse_fill(nums, size);
    }
    if (strcmp(input_type, "nearly") == 0)
    {
        nearly_fill(nums, size);
    }
    CALI_MARK_END(data_init);
}

/*
def merge_sort(array):
    total_length = array.length
    if total_length < 2:
        return array

    midpoint = total_length / 2
    left = merge_sort[0:midpoint]
    right = merge_sort[midpoint:total_length]
    l_index = 0
    r_index = 0
    final_array = []

    while l_index < left.length and r_index < r.length:
        if left[l_index] < r[r_index]:
            final_array.append(left[l_index])
            l_index += 1
    else:
        final_array.append(right[r_index])
        r_index += 1

    #catch extraneous values
    while l_index < left.length:
    final_array.append(left[l_index])
    l_index += 1
    while r_index < right.length:
    final_array.append(right[r_index])
    r_index += 1
*/

// https://www.geeksforgeeks.org/merge-sort/#
//  Merges two subarrays of array[].
//  First subarray is arr[begin..mid]
//  Second subarray is arr[mid+1..end]
void merge(float array[], int const left, int const mid,
           int const right)
{
    int const subArrayOne = mid - left + 1;
    int const subArrayTwo = right - mid;

    // Create temp arrays
    auto *leftArray = new float[subArrayOne],
         *rightArray = new float[subArrayTwo];

    // Copy data to temp arrays leftArray[] and rightArray[]
    for (auto i = 0; i < subArrayOne; i++)
        leftArray[i] = array[left + i];
    for (auto j = 0; j < subArrayTwo; j++)
        rightArray[j] = array[mid + 1 + j];

    auto indexOfSubArrayOne = 0, indexOfSubArrayTwo = 0;
    int indexOfMergedArray = left;

    // Merge the temp arrays back into array[left..right]
    while (indexOfSubArrayOne < subArrayOne && indexOfSubArrayTwo < subArrayTwo)
    {
        if (leftArray[indexOfSubArrayOne] <= rightArray[indexOfSubArrayTwo])
        {
            array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
            indexOfSubArrayOne++;
        }
        else
        {
            array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
            indexOfSubArrayTwo++;
        }
        indexOfMergedArray++;
    }

    // Copy the remaining elements of
    // left[], if there are any
    while (indexOfSubArrayOne < subArrayOne)
    {
        array[indexOfMergedArray] = leftArray[indexOfSubArrayOne];
        indexOfSubArrayOne++;
        indexOfMergedArray++;
    }

    // Copy the remaining elements of
    // right[], if there are any
    while (indexOfSubArrayTwo < subArrayTwo)
    {
        array[indexOfMergedArray] = rightArray[indexOfSubArrayTwo];
        indexOfSubArrayTwo++;
        indexOfMergedArray++;
    }
    delete[] leftArray;
    delete[] rightArray;
}

// begin is for left index and end is right index
// of the sub-array of arr to be sorted
void mergeSort(float array[], int const begin, int const end)
{
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    if (begin >= end)
        return;

    int mid = begin + (end - begin) / 2;
    mergeSort(array, begin, mid);
    mergeSort(array, mid + 1, end);
    merge(array, begin, mid, end);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
}

// UTILITY FUNCTIONS
// Function to print an array
void printArray(float A[], int size)
{
    for (int i = 0; i < size; i++)
        cout << A[i] << " ";
    cout << endl;
}

bool confirm_sorted(float *nums, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        if (nums[i] > nums[i + 1])
        {
            cout << "Correctness Check Failed..." << endl;
            return false;
        }
    }
    cout << "Correctness Check Passed!" << endl;
    return true;
}

// Driver code
int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    mgr.start();
    // retrieve user input
    const char *input_type = argv[1];
    int size = atoi(argv[2]);
    float *nums = new float[size];

    fill_array(nums, size, input_type);
    cout << "Data Initialized" << endl;
    mergeSort(nums, 0, size - 1);
    cout << "Merge Sort Completed" << endl;
    CALI_MARK_BEGIN(correctness_check);
    confirm_sorted(nums, size);
    CALI_MARK_END(correctness_check);

    // Metadata
    adiak::init(NULL);
    adiak::launchdate();                                  // launch date of the job
    adiak::libraries();                                   // Libraries used
    adiak::cmdline();                                     // Command line used to launch the job
    adiak::clustername();                                 // Name of the cluster
    adiak::value("Algorithm", "Merge Sort");             // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "Sequential");       // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float");                    // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float));        // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", size);                      // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type);                // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("group_num", 3);                         // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    delete[] nums;
}