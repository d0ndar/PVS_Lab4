#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <climits>
#include <ctime>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__device__ void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

__global__ void bitonicSortStep(int* dev_values, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    
    if (ixj > i) {
        if ((i & k) == 0) {
            if (dev_values[i] > dev_values[ixj]) {
                swap(&dev_values[i], &dev_values[ixj]);
            }
        }
        if ((i & k) != 0) {
            if (dev_values[i] < dev_values[ixj]) {
                swap(&dev_values[i], &dev_values[ixj]);
            }
        }
    }
}

void bitonicSort(int* values, int size) {
    int* dev_values;
    CHECK_CUDA(cudaMalloc(&dev_values, size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dev_values, values, size * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortStep<<<blocks, threads>>>(dev_values, j, k);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    CHECK_CUDA(cudaMemcpy(values, dev_values, size * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dev_values));
}

int main() {
    const char* size_str = getenv("ARRAY_SIZE");
    if (!size_str) {
        printf("ARRAY_SIZE environment variable not set\n");
        return 1;
    }
    
    const int ARRAY_SIZE = atoi(size_str);
    if (ARRAY_SIZE <= 0) {
        printf("Invalid ARRAY_SIZE value\n");
        return 1;
    }

    // Размер массива должен быть степенью двойки для Bitonic Sort
    int actual_size = 1;
    while (actual_size < ARRAY_SIZE) actual_size <<= 1;
    
    int* arr = (int*)malloc(actual_size * sizeof(int));
    
    // Инициализация массива случайными числами
    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand() % 10000;
    }
    // Заполнение остатка максимальными значениями
    for (int i = ARRAY_SIZE; i < actual_size; i++) {
        arr[i] = INT_MAX;
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    bitonicSort(arr, actual_size);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("%.6f\n", milliseconds / 1000.0f);

    free(arr);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}