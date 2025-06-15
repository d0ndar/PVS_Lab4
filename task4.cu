#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void matrix_operations(const float* a, const float* b, float* add, float* sub, 
                                 float* mul, float* div, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        int idx = row * size + col;
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        div[idx] = (b[idx] != 0.0f) ? a[idx] / b[idx] : 0.0f;
    }
}

int main() {
    const char* size_str = getenv("ARRAY_SIZE");
    if (!size_str) {
        fprintf(stderr, "Error: ARRAY_SIZE environment variable must be set\n");
        return EXIT_FAILURE;
    }

    const int SIZE = atoi(size_str);
    if (SIZE <= 0) {
        fprintf(stderr, "Error: Invalid ARRAY_SIZE value\n");
        return EXIT_FAILURE;
    }

    // Проверка на слишком большой размер
    if (SIZE > 8192) {
        fprintf(stderr, "Error: ARRAY_SIZE too large (max 8192)\n");
        return EXIT_FAILURE;
    }

    // Выделение памяти с проверкой
    size_t matrix_size = SIZE * SIZE * sizeof(float);
    float *h_a = (float*)malloc(matrix_size);
    float *h_b = (float*)malloc(matrix_size);
    float *h_add = (float*)malloc(matrix_size);
    float *h_sub = (float*)malloc(matrix_size);
    float *h_mul = (float*)malloc(matrix_size);
    float *h_div = (float*)malloc(matrix_size);
    
    assert(h_a && h_b && h_add && h_sub && h_mul && h_div);

    // Инициализация
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            int idx = i * SIZE + j;
            h_a[idx] = (float)(rand() % 100) / 10.0f;
            h_b[idx] = (float)(rand() % 100 + 1) / 10.0f;
        }
    }

    // Выделение памяти на GPU
    float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;
    CHECK_CUDA(cudaMalloc(&d_a, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_b, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_add, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_sub, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_mul, matrix_size));
    CHECK_CUDA(cudaMalloc(&d_div, matrix_size));

    // Копирование данных
    CHECK_CUDA(cudaMemcpy(d_a, h_a, matrix_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, matrix_size, cudaMemcpyHostToDevice));

    // Конфигурация запуска
    dim3 blockSize(16, 16);
    dim3 gridSize((SIZE + blockSize.x - 1) / blockSize.x, 
                 (SIZE + blockSize.y - 1) / blockSize.y);

    // Запуск ядра
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    matrix_operations<<<gridSize, blockSize>>>(d_a, d_b, d_add, d_sub, d_mul, d_div, SIZE);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Замер времени
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // Копирование результатов
    CHECK_CUDA(cudaMemcpy(h_add, d_add, matrix_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sub, d_sub, matrix_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mul, d_mul, matrix_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_div, d_div, matrix_size, cudaMemcpyDeviceToHost));

    // Вывод времени
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("%.6f \n", milliseconds / 1000.0f);

    // Очистка
    free(h_a); free(h_b); free(h_add); free(h_sub); free(h_mul); free(h_div);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_add));
    CHECK_CUDA(cudaFree(d_sub));
    CHECK_CUDA(cudaFree(d_mul));
    CHECK_CUDA(cudaFree(d_div));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}