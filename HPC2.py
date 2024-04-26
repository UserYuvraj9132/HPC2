import numpy as np
import time
from numba import jit, prange
import multiprocessing as mp

def parallel_bubble_sort(arr):
    n = len(arr)
    num_threads = mp.cpu_count()
    chunk_size = max(n // num_threads, 1)

    for _ in range(n):
        swapped = False
        
        with mp.Pool(processes=num_threads) as pool:
            results = pool.starmap(bubble_sort_worker, [(arr, i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)])
            arr = np.concatenate(results)
        for i in range(0, n-1):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
                swapped = True
        if not swapped:
            break

    return arr


def bubble_sort_worker(arr, start, end):
    for i in range(start, end-1):
        for j in range(start, end-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr[start:end]

@jit(nopython=True, parallel=True)
def parallel_merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = parallel_merge_sort(arr[:mid])
    right = parallel_merge_sort(arr[mid:])
    return merge(left, right)

@jit(nopython=True)
def merge(left, right):
    result = np.empty(len(left) + len(right), dtype=left.dtype)
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result[k] = left[i]
            i += 1
        else:
            result[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        result[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        result[k] = right[j]
        j += 1
        k += 1
    return result


if __name__ == "__main__":
    np.random.seed(9)
    arr = np.random.randint(0, 10, 10)
    print("Original Array:", arr)

    start_time = time.time()
    sorted_arr = parallel_bubble_sort(arr.copy())
    end_time = time.time()
    print("Parallel Bubble Sort Result:", sorted_arr)
    print("Parallel Bubble Sort Time:", end_time - start_time)

    start_time = time.time()
    sorted_arr = parallel_merge_sort(arr.copy())
    end_time = time.time()
    print("Parallel Merge Sort Result:", sorted_arr)
    print("Parallel Merge Sort Time:", end_time - start_time)