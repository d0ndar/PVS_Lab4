#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void vector_operations(const float* a, const float* b, float* add, float* sub, float* mul, float* div, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Сложение
        add[idx] = a[idx] + b[idx];
        // Вычитание
        sub[idx] = a[idx] - b[idx];
        // Умножение
        mul[idx] = a[idx] * b[idx];
        // Деление (с проверкой деления на ноль)
        div[idx] = (b[idx] != 0.0f) ? a[idx] / b[idx] : 0.0f;
    }
}

int main() {
    const char* size_str = getenv("ARRAY_SIZE");
    if (!size_str) {
        fprintf(stderr, "Error: ARRAY_SIZE environment variable not set\n");
        return EXIT_FAILURE;
    }

    const int ARRAY_SIZE = atoi(size_str);
    if (ARRAY_SIZE <= 0) {
        fprintf(stderr, "Error: Invalid ARRAY_SIZE value\n");
        return EXIT_FAILURE;
    }

    // Выделение памяти и инициализация массивов на хосте
    float *h_a = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_b = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_add = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_sub = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_mul = (float*)malloc(ARRAY_SIZE * sizeof(float));
    float *h_div = (float*)malloc(ARRAY_SIZE * sizeof(float));

    // Инициализация массивов случайными значениями
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = (float)(rand() % 100) / 10.0f;
        h_b[i] = (float)(rand() % 100 + 1) / 10.0f; // +1 чтобы избежать деления на 0
    }

    // Выделение памяти на устройстве
    float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;
    CHECK_CUDA(cudaMalloc(&d_a, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_add, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sub, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mul, ARRAY_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_div, ARRAY_SIZE * sizeof(float)));

    // Копирование данных на устройство
    CHECK_CUDA(cudaMemcpy(d_a, h_a, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Настройка параметров запуска ядра
    int threadsPerBlock = 256;
    int blocksPerGrid = (ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    // Создание событий для измерения времени
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Запуск ядра
    vector_operations<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_add, d_sub, d_mul, d_div, ARRAY_SIZE);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Завершение измерения времени
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Копирование результатов обратно на хост
    CHECK_CUDA(cudaMemcpy(h_add, d_add, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sub, d_sub, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_mul, d_mul, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_div, d_div, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    // Вычисление и вывод времени выполнения
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("%.6f\n", milliseconds / 1000.0f);

    // Освобождение ресурсов
    free(h_a); free(h_b); free(h_add); free(h_sub); free(h_mul); free(h_div);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}