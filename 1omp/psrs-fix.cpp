#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define NUM_THREADS 3
#define MAXNUM 65535

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

void qSort(int array[], int low, int high)
{
    if (low < high)
    {
        int mid = partition(array, low, high);
        qSort(array, low, mid - 1);
        qSort(array, mid + 1, high);
    }
}

void veccopy(int *from, int *to, int len)
{
    for (int i = 0; i < len; i++)
        to[i] = from[i];
}

void PSRS(int *seq, int n)
{
    int step, len[NUM_THREADS], sample[NUM_THREADS * NUM_THREADS], sample_main[NUM_THREADS], seq_copy[n + NUM_THREADS - 1], position[NUM_THREADS][NUM_THREADS + 1];
    if (n <= 1)
        return;
    int k = int(ceil((double)n / NUM_THREADS));
    int m = k * NUM_THREADS - n;
    veccopy(seq, seq_copy, n);
    for (int i = 0; i < m; i++)
        seq_copy[n + i] = MAXNUM;
    step = int(floor((double)k / NUM_THREADS));
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (int i = 0; i < NUM_THREADS; i++)
        {
            qSort(seq_copy, id * k, id * k + k - 1);
            sample[id * NUM_THREADS + i] = seq_copy[id * k + step * i];
        }
    }
    qSort(sample, 0, NUM_THREADS * NUM_THREADS - 1);
    for (int i = 1; i < NUM_THREADS; i++)
    {
        sample_main[i - 1] = sample[i * NUM_THREADS];
    }
    sample_main[NUM_THREADS - 1] = MAXNUM - 1;
    omp_set_num_threads(NUM_THREADS);
#pragma omp parallel shared(position)
    {
        int start = 0, id = omp_get_thread_num(), i = 0, j = 0;
        position[id][0] = 0;
        while (i < k)
        {
            if (seq_copy[id * k + i] > sample_main[j])
            {
                j++;
                position[id][j] = i;
            }
            i++;
        }
        position[id][NUM_THREADS] = k;
#pragma omp barrier
        for (int i = 0; i < NUM_THREADS; i++)
        {
            len[id] += position[i][id + 1] - position[i][id];
        }
#pragma omp barrier
        for (int i = 0; i < id; i++)
        {
            start += len[i];
        }
        for (int i = 0, j = 0; i < NUM_THREADS; i++)
        {
            veccopy(seq_copy + i * k + position[i][id], seq + start + j, position[i][id + 1] - position[i][id]);
            j += position[i][id + 1] - position[i][id];
        }
        qSort(seq + start, 0, len[id] - 1);
    }
}

int main()
{
    int array[27] = {15, 46, 48, 93, 39, 6, 72, 91, 14, 36, 69, 40, 89, 61, 97, 12, 21, 54, 53, 97, 84, 58, 32, 27, 33, 72, 20};
    double begin, end, time;

    begin = omp_get_wtime();
    PSRS(array, 27);
    // qSort(array, 0, 26);
    end = omp_get_wtime();

    time = end - begin;
    printf("Final: ");
    for (int i = 0; i < 27; i++)
        printf("%d ", array[i]);
    printf("\nRunning time: %lfs\n", time);
    return 0;
}