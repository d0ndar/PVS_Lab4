#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
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

    int* arr = (int*)malloc(ARRAY_SIZE * sizeof(int));
    
    // Инициализация массива случайными числами
    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = rand() % 10000;
    }

    clock_t start = clock();
    mergeSort(arr, 0, ARRAY_SIZE - 1);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%.6f\n", time_spent);

    free(arr);
    return 0;
}