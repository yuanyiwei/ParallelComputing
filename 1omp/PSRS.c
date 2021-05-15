#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define num_threads 3
#define DBL_MAX 65535

void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}
int partition(int array[], int low, int high)
{
    int pivot = array[high];
    int i = (low - 1);
    for (int j = low; j < high; j++)
    {
        if (array[j] <= pivot)
        {
            i++;
            swap(&array[i], &array[j]);
        }
    }
    swap(&array[i + 1], &array[high]);
    return (i + 1);
}

void quickSort(int array[], int low, int high)
{
    if (low < high)
    {
        int pi = partition(array, low, high);
        quickSort(array, low, pi - 1);
        quickSort(array, pi + 1, high);
    }
}

void copy_seq(int *from, int *to, int len)
{
    for (int i = 0; i < len; i++)
        to[i] = from[i];
}

void PSRS(int *seq, int n)
{
    int k, m, step, i, p, j = 0, position[num_threads][num_threads + 1], len[num_threads];
    int sample[num_threads * num_threads], sample_main[num_threads], copy[n + num_threads - 1];
    if (n <= 1)
        return;
    p = num_threads;
    while (p * p > n)
    {
        p--;
    }
    k = int(ceil((double)n / p));
    m = k * p - n;
    copy_seq(seq, copy, n);
    for (i = 0; i < m; i++)
        copy[n + i] = DBL_MAX;
    step = int(floor((double)k / p));
    omp_set_num_threads(p);
#pragma omp parallel private(i)
    {
        int id;
        id = omp_get_thread_num();
        for (i = 0; i < p; i++)
        {
            quickSort(copy, id * k, id * k + k - 1);
            sample[id * p + i] = copy[id * k + step * i];
        }
#pragma omp critical
        {
            printf("\n%d:", id);
            for (i = 0; i < k; i++)
                printf("%d  ", (int)copy[id * k + i]);
        }
    }
    quickSort(sample, 0, p * p - 1);
    printf("\n");
    for (i = 0; i < p * p; i++)
        printf("%d  ", (int)sample[i]);
    for (i = 1; i < p; i++)
        sample_main[i - 1] = sample[i * p];
    sample_main[p - 1] = DBL_MAX - 1;
    printf("\n");
    for (i = 0; i < p - 1; i++)
        printf("%d  ", (int)sample_main[i]);
    omp_set_num_threads(p);
#pragma omp parallel private(i, j) shared(position)
    {
        int id, start = 0, t;
        id = omp_get_thread_num();
        for (i = 0, j = 0, position[id][0] = 0; i < k; i++)
        {
            if (copy[id * k + i] > sample_main[j])
            {
                j++;
                position[id][j] = i;
            }
        }
        position[id][p] = k;
//position[p-1][p]=k-m;
#pragma omp barrier
        for (i = 0; i < p; i++)
            len[id] += position[i][id + 1] - position[i][id];
#pragma omp barrier
        for (i = 0; i < id; i++)
            start += len[i];
        for (i = 0, j = 0; i < p; i++)
        {
            printf("\nthread:%d,j=%d,start=%d,n=%d", omp_get_thread_num(), j, start, position[i][id + 1] - position[i][id]);
            copy_seq(copy + i * k + position[i][id], seq + start + j, position[i][id + 1] - position[i][id]);
            for (t = 0; t < position[i][id + 1] - position[i][id]; t++)
                printf("  %d  ", (int)copy[i * k + position[i][id] + t]);
            j = j + position[i][id + 1] - position[i][id];
        }
        //#pragma omp barrier
        //position[p-1][p]=k-m;
        quickSort(seq + start, 0, len[id] - 1);
    }
}

int main()
{
    int array[27] = {15, 46, 48, 93, 39, 6, 72, 91, 14, 36, 69, 40, 89, 61, 97, 12, 21, 54, 53, 97, 84, 58, 32, 27, 33, 72, 20};
    double begin, end, time;

    begin = omp_get_wtime();
    PSRS(array, 27);
    // quickSort(array, 0, 26);
    end = omp_get_wtime();

    time = end - begin;
    printf("\nFinal: ");
    for (int i = 0; i < 27; i++)
        printf("%d ", array[i]);
    printf("\nRunning time: %lfs\n", time);
    return 0;
}