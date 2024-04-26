import numpy as np
import time
from numba import jit, prange
import multiprocessing as mp

def parallel_bubble_sort(arr):
    n = len(arr)
    num_threads = mp.cpu_count()
    chunk_size = max(1, n // num_threads)  # Ensure chunk_size is at least 1

    for _ in range(n):
        swapped = False
        # Parallelize the loop
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

@jit(nopython=True)
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

def sort_time(func, arr):
    start_time = time.time()
    sorted_arr = func(arr.copy())
    end_time = time.time()
    print("Sorted Array:", sorted_arr)
    print("Sorting Time:", end_time - start_time)

if __name__ == "__main__":
    n = int(input("Enter the number of elements in the array: "))
    arr_str = input("Enter the elements of the array separated by spaces: ")
    arr = np.array(list(map(int, arr_str.split())))

    print("Original Array:", arr)

    print("Parallel Bubble Sort:")
    sort_time(parallel_bubble_sort, arr)

    print("Parallel Merge Sort:")
    sort_time(parallel_merge_sort, arr)