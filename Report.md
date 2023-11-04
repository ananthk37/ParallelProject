# CSCE 435 Group project

## 0. Group number: 

## 1. Group members:
We will communicate using discord during the duration of the project.
1. Robbie Clark
2. Eric Lee
3. Rushil Aggarwal
4. Ananth Kumar 

## 2. Project topic (e.g., parallel sorting algorithms)
Sorting Algorithms

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)
1. Bubble Sort (Sequential) / Odd-Even Sort (Parallel)
    - Sequential
    - Parallel using hardware threads (MPI)
    - Parallel using CUDA (GPU)
2. Merge Sort
    - Sequential
    - Parallel using hardware threads (MPI)
    - Parallel using CUDA (GPU)
3. Radix Sort
    - Sequential
    - Parallel using hardware threads (MPI)
    - Parallel using CUDA (GPU)

### 2b. Pseudocode for each parallel algorithm
- For MPI programs, include MPI calls you will use to coordinate between processes
- For CUDA programs, indicate which computation will be performed in a CUDA kernel,
  and where you will transfer data to/from GPU

1. Bubble Sort (Sequential)
    ```python
    def bubble_sort(array):
        i = array.length - 1
        sorted = False
        while i > 0 and not sorted:
            sorted = True
            for j=1 to i-1:
                if a[j-1] > a[j]:
                    swap a[j-1] and a[j]
                    sorted = false
            i -= 1
    ```
2. Odd-Even Sort (Parallel)
    ```c++
    def OddEvenSortStep(float* nums, int size, int i) {
        index = // get either MPI rank or index using CUDA
        // Odd step
        if (i == 0 && (index * 2 + 1) < size) {
            if(nums[index * 2] > nums[index * 2 + 1]) {
                swap(nums[index * 2], nums[index * 2 + 1]);
            }
        }
 
        // Even step
        if (i == 0 && (index * 2 + 2) < size) {
            if(nums[index * 2 + 1] > nums[index * 2 + 2]) {
                swap(nums[index * 2 + 1], nums[index * 2 + 2]);
            }
        }
    }

    def OddEvenSort(float* nums, int size) {
        //memcpy host to device if using cuda
        for (i = 1; i <= size; i++) {
            OddEvenSortStep(nums, size, i%2);
        }
        //memcpy device to host if using cuda
    }
    ```
3. Merge Sort (Sequential)
    ```python
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
    ```
4. Merge Sort (Parallel)
    - Perform merge sort but allocate each recursive call to a different thread
5. Radix Sort (Sequential)
    ```python
    def radix_sort(arr):
        maxNum = max(arr)
        magnitude = 1
        while maxNum/magnitude >= 1: #while we are within the number of digits of the max number
            n = length(arr)
            count = [0] * 10 #initialize count array of 0s of size 10 each index represents a counter for each digit 0-9
            output = [0] * n
            for i from 0 to n:
                temp = arr[i] #reads value at ith place in array
                count[temp % 10] += 1 #counts the digit at the current index we are looking att
            for i from 1 to 10:
                count[i] += count[i-1]
            
            i = n - 1
            while i >= 0:
                index = arr[i]
                output[count[index%10] - 1] = arr[i]
                count[index%10] -= 1
                i -= 1
            i = 0
            for i in range(0,len(arr)):
                arr[i] = output[i]
                magnitude *= 10
        ```
6. Radix Sort (Parallel)
    ```
    parallel_for part in 0..K-1
    for i in indexes(part):
        bucket = compute_bucket(a[i])
        cnt[part][bucket]++

    base = 0
    for bucket in 0..R-1
        for part in 0..K-1
            Cnt[part][bucket] += base
            base = Cnt[part][bucket]

    parallel_for part in 0..K-1
    for i in indexes(part)
        bucket = compute_bucket(a[i])
        out[Cnt[part][bucket]++] = a[i]
    ```

#### Sources Used
1. https://www.geeksforgeeks.org/odd-even-transposition-sort-brick-sort-using-pthreads/ (Odd-Even Sort)
2. https://rachitvasudeva.medium.com/parallel-merge-sort-algorithm-e8175ab60e7 (Merge Sort Parallel)
3. https://www.geeksforgeeks.org/radix-sort/ (Radix Sequential)
4. https://cs.stackexchange.com/questions/6871/how-does-the-parallel-radix-sort-work (Radix Parallel)

### 2c. Evaluation plan - what and how will you measure and compare
- Input sizes, Input types
- Strong scaling (same problem size, increase number of processors/nodes)
- Weak scaling (increase problem size, increase number of processors)
- Number of threads in a block on the GPU 


## 3. Project implementation
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.

### 3a. Caliper instrumentation
Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` 
(same as lab1 build.sh) to collect caliper files for each experiment you run.

Your Caliper regions should resemble the following calltree
(use `Thicket.tree()` to see the calltree collected on your runs):
```
main
|_ data_init
|_ comm
|    |_ MPI_Barrier
|    |_ comm_small  // When you broadcast just a few elements, such as splitters in Sample sort
|    |   |_ MPI_Bcast
|    |   |_ MPI_Send
|    |   |_ cudaMemcpy
|    |_ comm_large  // When you send all of the data the process has
|        |_ MPI_Send
|        |_ MPI_Bcast
|        |_ cudaMemcpy
|_ comp
|    |_ comp_small  // When you perform the computation on a small number of elements, such as sorting the splitters in Sample sort
|    |_ comp_large  // When you perform the computation on all of the data the process has, such as sorting all local elements
|_ correctness_check
```

Required code regions:
- `main` - top-level main function.
    - `data_init` - the function where input data is generated or read in from file.
    - `correctness_check` - function for checking the correctness of the algorithm output (e.g., checking if the resulting data is sorted).
    - `comm` - All communication-related functions in your algorithm should be nested under the `comm` region.
      - Inside the `comm` region, you should create regions to indicate how much data you are communicating (i.e., `comm_small` if you are sending or broadcasting a few values, `comm_large` if you are sending all of your local values).
      - Notice that auxillary functions like MPI_init are not under here.
    - `comp` - All computation functions within your algorithm should be nested under the `comp` region.
      - Inside the `comp` region, you should create regions to indicate how much data you are computing on (i.e., `comp_small` if you are sorting a few values like the splitters, `comp_large` if you are sorting values in the array).
      - Notice that auxillary functions like data_init are not under here.

All functions will be called from `main` and most will be grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

**Nesting Code Regions** - all computation code regions should be nested in the "comp" parent code region as following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
mergesort();
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Looped GPU kernels** - to time GPU kernels in a loop:
```
### Bitonic sort example.
int count = 1;
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
int j, k;
/* Major step */
for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
        bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        count++;
    }
}
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Calltree Examples**:

```
# Bitonic sort tree - CUDA looped kernel
1.000 main
├─ 1.000 comm
│  └─ 1.000 comm_large
│     └─ 1.000 cudaMemcpy
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Matrix multiplication example - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  ├─ 1.000 comm_large
│  │  ├─ 1.000 MPI_Recv
│  │  └─ 1.000 MPI_Send
│  └─ 1.000 comm_small
│     ├─ 1.000 MPI_Recv
│     └─ 1.000 MPI_Send
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Mergesort - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  └─ 1.000 comm_large
│     ├─ 1.000 MPI_Gather
│     └─ 1.000 MPI_Scatter
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

#### 3b. Collect Metadata

Have the following `adiak` code in your programs to collect metadata:
```
adiak::init(NULL);
adiak::launchdate();    // launch date of the job
adiak::libraries();     // Libraries used
adiak::cmdline();       // Command line used to launch the job
adiak::clustername();   // Name of the cluster
adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

**See the `Builds/` directory to find the correct Caliper configurations to get the above metrics for CUDA, MPI, or OpenMP programs.** They will show up in the `Thicket.dataframe` when the Caliper file is read into Thicket.
