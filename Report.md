# CSCE 435 Group project

## 1. Group members:
1. Robbie Clark
2. Eric Lee
3. Rushil Aggarwal
4. Ananth Kumar

---

## 2. Project topic
Sorting Algorithms

## 3. Brief project description (what algorithms will you be comparing and on what architectures)

1. Bubble Sort (Sequential) / Odd-Even Sort (Parallel)
    - Sequential
    - Parallel using hardware threads (CPU)
    - Parallel using CUDA (GPU)
2. Merge Sort
    - Sequential
    - Parallel using hardware threads (CPU)
    - Parallel using CUDA (GPU)
3. Radix Sort
    - Sequential
    - Parallel using hardware threads (CPU)
    - Parallel using CUDA (GPU)
    
### Psuedo-code
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
    def OddEvenSort(pthread_t threads[]){
	int i, j;
 
    for (i = 1; i <= n; i++) {
        // Odd step
        if (i % 2 == 1) {
            tmp = 0;
 
            // Creating threads
            for (j = 0; j < max_threads; j++)
                pthread_create(& threads[j], NULL, compare, NULL);
 
            // joining threads i.e. waiting
            // for all the threads to complete
            for (j = 0; j < max_threads; j++)
                pthread_join(threads[j], NULL);
        }
 
        // Even step
        else {
            tmp = 1;
 
            // Creating threads
            for (j = 0; j < max_threads - 1; j++)
                pthread_create(&threads[j], NULL, compare, NULL);
 
            // joining threads i.e. waiting
            // for all the threads to complete
            for (j = 0; j < max_threads - 1; j++)
                pthread_join(threads[j], NULL);
        }
    }

    void* compare(void* arg)
    {
    
        // Each thread compares
        // two consecutive elements of the array
        int index = tmp;
        tmp = tmp + 2;
    
        if ((index + 1 < n) && (a[index] > a[index + 1])) {
            swap(a[index], a[index + 1]);
        }
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

### Sources Used
1. https://www.geeksforgeeks.org/odd-even-transposition-sort-brick-sort-using-pthreads/ (Odd-Even Sort)
2. https://rachitvasudeva.medium.com/parallel-merge-sort-algorithm-e8175ab60e7 (Merge Sort Parallel)
3. https://www.geeksforgeeks.org/radix-sort/ (Radix Sequential)
4. https://cs.stackexchange.com/questions/6871/how-does-the-parallel-radix-sort-work (Radix Parallel)
