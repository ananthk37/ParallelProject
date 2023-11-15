# CSCE 435 Group project

## 0. Group number: 
Group #3

## 1. Group members:
We will communicate using discord during the duration of the project.
1. Robbie Clark
2. Eric Lee
3. Rushil Aggarwal
4. Ananth Kumar 

## 2. Project topic (e.g., parallel sorting algorithms)
Parallel Sorting Algorithms

### 2a. Brief project description (what algorithms will you be comparing and on what architectures)
1. Bubble Sort (Sequential) / Odd-Even Sort (Parallel)
    - Sequential
    - Parallel using MPI
    - Parallel using CUDA
2. Merge Sort
    - Sequential
    - Parallel using MPI
    - Parallel using CUDA
3. Selection Sort
    - Sequential
    - Parallel using MPI
    - Parallel using CUDA
4. Quick Sort
    - Sequential
    - Parallel using MPI
    - Parallel using CUDA

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
2. Odd-Even Sort (MPI)
    ```
    rank = rank of process
    num_procs = total number of processes
    n = local_array.length
    sort(local_array)

    for i to n:
        // even step
        if i % 2 == 0:
            partner = rank-1 if odd rank and rank+1 if even rank
            if partner >= 0 and partner < num_procs:
                //MPI_SendRecv local_array with partner rank
                if rank % 2 == 0:
                    local_array = //min half of combined data (local & partner array)
                else:
                    local_array = //max half of combined data (local & partner array)

        // odd step
        else:
            partner = rank+1 if odd rank and rank-1 if even rank
            if partner >= 0 and partner < num_procs:
                //MPI_SendRecv local_array with partner rank
                if rank % 2 == 0:
                    local_array = //max half of combined data (local & partner array)
                else:
                    local_array = //min half of combined data (local & partner array)
    ```

3. Odd-Even Sort (CUDA)
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
        //memcpy host to device
        for (i = 1; i <= size; i++) {
            // performed in a CUDA Kernel
            OddEvenSortStep(nums, size, i%2);
        }
        //memcpy device to host
    }
    ```
4. Merge Sort (Sequential)
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
5. Merge Sort (CUDA)
    ```
    __device__ void merge()
        serial merge implementation
    __global__ void mergeSort(data, size of data 'n')
        tid = threadindex
        for i to n stepping by *2 //this is for going from size 1 arrays then merge to 2 then 4 etc.
            for left = tid * 2 * currsize; left < n-1; left += gridDim * 2 * currSize
                mid = min(left + currsize-1, n-1)
                right = min(left + 2 * currsize - 1, n - 1);
                merge(data, left, mid right)
    main()
        numBlocks = size of data/numthreads per block
        mergeSort<<numBlocks, numThreads>> (inputdata, size of data)
    ```

6. Merge Sort (MPI)
    ```
    void merge()
        serial merge implementation
    void mergesort()
        serial mergeSort implementation
    void main()
        Do MPI initializiation (init, comm, etc)

        if root:
            distribute work to workers
            work on your own chunk of data
            receive data from other threads and do final merge
        else:
            get work from master process
            do mergeSort on your data
            send sorted data back to master
    ```
7. Selection Sort (Sequential)
    ```python
    def sort(arr):
        for start_index in arr:
            min_value = arr[start_index]
            min_index = start_index
            for index from start to arr.size:
                if arr[index] < min_value:
                    min_index = index
                    min_value = arr[index]
            swap arr[start_index] and arr[min_index]
    ```
8. Selection Sort (CUDA)
    ```python
    def host_sort(arr):
        load arr into gpu
        load min_val into gpu
        for start_index in arr:
            set min_val to max_int  
            min_val = gpu_min(arr from start_index to end)
            min_index = gpu_index(min_val, arr from start_index to end)
            gpu_swap(arr[min_index] and arr[start_index])
    
    def gpu_min(value): #gpu function on each value
        min_val = atomic_min(min_val, arr[value])
    
    def gpu_index(value): #gpu function on each value
        if arr[value] == min_val:
            min_index = value
    
    def gpu_swap(val1, val2) #gpu function called once
        atomic_swap(arr[val1], arr[val2])

    ```
9. Selection Sort (MPI)
    ```python
    def mpi_sort(array):
        scatter array to p procs in local_array
        selection_sort(local_array) for each proc
        gather local_arrays into array
        for i from p/2 to 0, dividing by 2:
            for j from 0 to i, adding by 1:
                send 2 sorted arrays to proc j
                merge(arr1, arr2)
                receive merged array
                put back into array

    def selection_sort(array):
        same as sequential selection_sort

    def merge(arr1, arr2):
        same as sequential merge_sort_step
        send to master process

    ```
10. Quick Sort (Sequential)
    ```c++
    int partition(int* nums, int low, int high){
    int pivot = nums[high];

    int i = (low - 1);
    
    for(int j = low; j < high; j++){
        if(nums[j] < pivot){
            i++;
            swap(nums[i], nums[j]);
        }
    }

    swap(nums[i + 1], nums[high]);

    return i + 1;
    }

    void quick_sort(int* nums, int low, int high) {
    if(low < high){
        int pi = partition(nums, low, high);

        quick_sort(nums, low, pi - 1);
        quick_sort(nums, pi + 1, high);
    }
    }
    ```
11. Quick Sort (CUDA)
    ```c++
    void quick_sort_step(int* dev_nums, int left, int right){
        int stack[64];
        int top = -1;
    
        stack[++top] = left;
        stack[++top] = right;
    
        while (top >= 0) {
            right = stack[top--];
            left = stack[top--];
    
            int pivotIndex = partition(dev_nums, left, right);
    
            if (pivotIndex - 1 > left) {
                stack[++top] = left;
                stack[++top] = pivotIndex - 1;
            }
    
            if (pivotIndex + 1 < right) {
                stack[++top] = pivotIndex + 1;
                stack[++top] = right;
            }
        }
    }
    void quick_sort(int* nums){
        for(int i = 0; i < NUM_VALS; i++){
            //perform cuda kernal call
            quick_sort_step(dev_nums, 0, i);
        }
    }
    ```
12. Quick Sort (MPI)
    ```c++
    void quicksort(int* arr, int start, int end){
        int pivot, index;
        if (end <= 1)
            return;
        pivot = arr[start + end / 2];
        swap(arr, start, start + end / 2);
        index = start;
        for (int i = start + 1; i < start + end; i++) {
            if (arr[i] < pivot) {
                index++;
                swap(arr, i, index);
            }
        }
        swap(arr, start, index);
        quicksort(arr, start, index - start);
        quicksort(arr, index + 1, start + end - index - 1);
    }

    int* merge(int* arr1, int n1, int* arr2, int n2){
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

    int chunk_size = (size % num_procs == 0) ? (size / num_procs) : size / (num_procs - 1);
    int* chunk = (int*)malloc(chunk_size * sizeof(int));
    // DO MPI_Scatter
    int own_chunk_size = (size >= chunk_size * (proc_id + 1)) ? chunk_size : (size - chunk_size * proc_id);
    quicksort(chunk, 0, own_chunk_size);
    if (rank_of_process % (2 * step) != 0) {
        //Do MPI_Send
        }
    if (rank_of_process + step < number_of_process)
        // DO MPI_Recv
        nums = merge(chunk, own_chunk_size, chunk_received, received_chunk_size);
    ```

#### Sources Used
1. https://selkie-macalester.org/csinparallel/modules/MPIProgramming/build/html/oddEvenSort/oddEven.html (Odd-Even MPI)
2. https://rachitvasudeva.medium.com/parallel-merge-sort-algorithm-e8175ab60e7 (Merge Sort Parallel)
3. https://www.geeksforgeeks.org/implementation-of-quick-sort-using-mpi-omp-and-posix-thread/ (Quick Sort MPI)


### 2c. Evaluation plan - what and how will you measure and compare
- Input Type: For each algorithm, the data type being sorted will be floats. There will be using 4 different types of data generation which include, sorted, reverse sorted, nearly sorted, and random. 
- Input sizes: The array will have 2<sup>16</sup>, 2<sup>20</sup>, or 2<sup>24</sup> values.
- Threads per GPU Block: Each block will have 64, 128, 256, 512, or 1024 threads.
- MPI Processes: The number of processes will be 2, 4, 8, 16, 32, 64, or 128.
- Scaling: We will be performing strong scaling for each given problem size. By scaling in this fashion we can also look at weak scaling by comparing performance across different problem sizes.


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

### 3b. Collect Metadata

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

## 4. Performance evaluation

Include detailed analysis of computation performance, communication performance. 
Include figures and explanation of your analysis.

### 4a. Vary the following parameters
For inputSizes:
- 2^16, 2^18, 2^20, 2^22, 2^24, 2^26, 2^28

For inputTypes:
- Sorted, Random, Reverse sorted, 1%perturbed

num_procs, num_threads:
- MPI: num_procs:
    - 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- CUDA: num_threads:
    - 64, 128, 256, 512, 1024, 2048, 4096

This should result in 4x7x10=280 Caliper files for your MPI experiments.

### 4b. Hints for performance analysis

To automate running a set of experiments, parameterize your program.

- inputType: If you are sorting, "Sorted" could generate a sorted input to pass into your algorithms
- algorithm: You can have a switch statement that calls the different algorithms and sets the Adiak variables accordingly
- num_procs:   How many MPI ranks you are using
- num_threads: Number of CUDA or OpenMP threads

When your program works with these parameters, you can write a shell script 
that will run a for loop over the parameters above (e.g., on 64 processors, 
perform runs that invoke algorithm2 for Sorted, ReverseSorted, and Random data).  

### 4c. You should measure the following performance metrics
- `Time`
    - Min time/rank
    - Max time/rank
    - Avg time/rank
    - Total time
    - Variance time/rank
    - `If GPU`:
        - Avg GPU time/rank
        - Min GPU time/rank
        - Max GPU time/rank
        - Total GPU time

`Intel top-down`: For your CPU-only experiments on the scale of a single node, you should
generate additional performance data, measuring the hardware counters on the CPU. This can be done by adding `topdown.all` to the `spot()` options in the `CALI_CONFIG` in your jobfile.

## 5. Presentation

## 6. Final Report
Submit a zip named `TeamX.zip` where `X` is your team number. The zip should contain the following files:
- Algorithms: Directory of source code of your algorithms.
- Data: All `.cali` files used to generate the plots seperated by algorithm/implementation.
- Jupyter notebook: The Jupyter notebook(s) used to generate the plots for the report.
- Report.md