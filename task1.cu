#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    }

__global__ void sumReduction(int* input, int* output, int size) {
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Загрузка данных в shared memory
    shared_data[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();
    
    // Параллельное сокращение
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s && (idx + s) < size) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    // Сохранение результата блока
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

int main() {
    // Получаем размер массива из переменной окружения
    const char* array_size_str = getenv("ARRAY_SIZE");
    if (!array_size_str) {
        fprintf(stderr, "Error: ARRAY_SIZE environment variable not set\n");
        return 1;
    }
    
    const int ARRAY_SIZE = atoi(array_size_str);
    if (ARRAY_SIZE <= 0) {
        fprintf(stderr, "Error: Invalid ARRAY_SIZE value\n");
        return 1;
    }

    // Выделяем и инициализируем память на хосте
    int* h_array = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_array[i] = 1;
    }

    // Выделяем память на устройстве
    int *d_input, *d_partial_sums;
    cudaError_t err;
    err = cudaMalloc(&d_input, ARRAY_SIZE * sizeof(int));
    CHECK_CUDA_ERROR(err);
    err = cudaMalloc(&d_partial_sums, ((ARRAY_SIZE + 255)/256) * sizeof(int));
    CHECK_CUDA_ERROR(err);

    // Копируем данные на устройство
    err = cudaMemcpy(d_input, h_array, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR(err);

    // Настраиваем параметры запуска
    const int blockSize = 256;
    int gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;

    // Создаем события для измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Первый этап: сокращение в блоках
    sumReduction<<<gridSize, blockSize, blockSize*sizeof(int)>>>(d_input, d_partial_sums, ARRAY_SIZE);
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err);
    cudaDeviceSynchronize();

    // Второй этап: итеративное сокращение результатов
    int remaining = gridSize;
    while (remaining > 1) {
        gridSize = (remaining + blockSize - 1) / blockSize;
        sumReduction<<<gridSize, blockSize, blockSize*sizeof(int)>>>(d_partial_sums, d_partial_sums, remaining);
        err = cudaGetLastError();
        CHECK_CUDA_ERROR(err);
        cudaDeviceSynchronize();
        remaining = gridSize;
    }

    // Завершаем измерение времени
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Выводим время выполнения
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%.6f\n", milliseconds / 1000.0f);

    // Освобождаем ресурсы
    cudaFree(d_input);
    cudaFree(d_partial_sums);
    free(h_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}