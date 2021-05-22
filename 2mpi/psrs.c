#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#define INF 65535

// A[l : m] + A[m + 1 : r] => A[l : r]
void Merge(int *A, int l, int m, int r)
{
    int i, j, k, n1 = m - l + 1, n2 = r - m;
    int *L = (int *)malloc((n1 + 1) * sizeof(int));
    int *R = (int *)malloc((n2 + 1) * sizeof(int));
    for (i = 0; i < n1; i++)
        L[i] = A[l + i];
    for (j = 0; j < n2; j++)
        R[j] = A[m + 1 + j];
    L[i] = R[j] = INF;
    i = j = 0;
    for (k = l; k <= r; k++)
        if (L[i] <= R[j])
            A[k] = L[i++];
        else
            A[k] = R[j++];
    free(L);
    free(R);
}

void mSort(int *A, int l, int r)
{
    if (l < r)
    {
        int m = (l + r) / 2;
        mSort(A, l, m);
        mSort(A, m + 1, r);
        Merge(A, l, m, r);
    }
}

void psrs(int *A, int n, int id, int num_processes)
{
    //每个进程都会执行这个函数，这里面的变量每个进程都有一份，因此都是局部的
    //global表示这个变量是0进程会使用的，但每个进程都声明了
    int *samples, *global_samples, per, *pivots, *sizes, *newsizes, *offsets, *newoffsets, *newdatas, newdatassize, *global_sizes, *global_offsets;

    per = n / num_processes;
    samples = (int *)malloc(num_processes * sizeof(int));
    pivots = (int *)malloc(num_processes * sizeof(int));
    //num_processes - 1 个主元，最后一个为哨兵
    if (id == 0)
    {
        global_samples = (int *)malloc(num_processes * num_processes * sizeof(int)); //主进程申请全局采样数组, 正则采样数为 num_processes * num_processes
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //mergesort in per
    mSort(A, id * per, (id + 1) * per - 1);
    //当前进程选出 num_processes 个样本放在local_sample中
    for (int k = 0; k < num_processes; k++)
        samples[k] = A[id * per + k * per / num_processes];
    //主进程的sample收集各进程的local_sample，共 num_processes * num_processes 个
    MPI_Gather(samples, num_processes, MPI_INT, global_samples, num_processes, MPI_INT, 0, MPI_COMM_WORLD);
    // sample sort
    if (id == 0)
    {
        mSort(global_samples, 0, num_processes * num_processes - 1); //对采样的num_processes * num_processes个样本进行排序
        for (int k = 0; k < num_processes - 1; k++)                  //num_processes - 1 主元
            pivots[k] = global_samples[(k + 1) * num_processes];
        pivots[num_processes - 1] = INF;
    }
    //0进程向各个进程广播，所有进程拥有一样的pivots数组
    MPI_Bcast(pivots, num_processes, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    sizes = (int *)calloc(num_processes, sizeof(int));      //当前进程的per个元素根据主元划分之后的每段的长度
    offsets = (int *)calloc(num_processes, sizeof(int));    //当前进程的per个元素根据主元划分之后的每段的起始位置
    newsizes = (int *)calloc(num_processes, sizeof(int));   //当前进程在进行全局交换之后的每段的长度
    newoffsets = (int *)calloc(num_processes, sizeof(int)); //当前进程在进行全局交换之后的每段的起始位置

    for (int k = 0, j = id * per; j < id * per + per; j++)
    {                         //当前进程对自己的per个元素按主元划分为num_processes段，此处不处理数据，只计算每段大小
        if (A[j] < pivots[k]) //如果之前不用哨兵，最后一段要单独考虑
            sizes[k]++;
        else
            sizes[++k]++;
    }
    //每个进程向每个接收者发送接收者对应的sizes
    MPI_Alltoall(sizes, 1, MPI_INT, newsizes, 1, MPI_INT, MPI_COMM_WORLD); //要发送的数据都是一样的,为sizes
    //计算原来的段偏移数组，新的段偏移数组，新的数据大小
    newdatassize = newsizes[0];
    for (int k = 1; k < num_processes; k++)
    {
        offsets[k] = offsets[k - 1] + sizes[k - 1];
        newoffsets[k] = newoffsets[k - 1] + newsizes[k - 1];
        newdatassize += newsizes[k];
    }
    newdatas = (int *)malloc(newdatassize * sizeof(int)); //当前进程在进行全局交换之后的数据，是由交换后的各段组合而成的
    MPI_Alltoallv(&(A[id * per]), sizes, offsets, MPI_INT, newdatas, newsizes, newoffsets, MPI_INT, MPI_COMM_WORLD);
    //每个进程向每个接收者发送接收者对应的A数据,即每一段之中的内容

    MPI_Barrier(MPI_COMM_WORLD);

    mSort(newdatas, 0, newdatassize - 1);

    MPI_Barrier(MPI_COMM_WORLD);
    //收集各进程新数据的大小
    if (id == 0)
        global_sizes = (int *)calloc(num_processes, sizeof(int));
    MPI_Gather(&newdatassize, 1, MPI_INT, global_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //计算即将搜集的各进程数据的起始位置
    if (id == 0)
    {
        global_offsets = (int *)calloc(num_processes, sizeof(int));
        for (int k = 1; k < num_processes; k++)
            global_offsets[k] = global_offsets[k - 1] + global_sizes[k - 1];
    }
    //收集各个进程的数据,写回A
    MPI_Gatherv(newdatas, newdatassize, MPI_INT, A, global_sizes, global_offsets, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    free(samples);
    free(pivots);
    free(sizes);
    free(offsets);
    free(newdatas);
    free(newsizes);
    free(newoffsets);
    samples = NULL;
    pivots = NULL;
    sizes = NULL;
    offsets = NULL;
    newdatas = NULL;
    newsizes = NULL;
    newoffsets = NULL;
    if (id == 0)
    {
        free(global_samples);
        free(global_sizes);
        free(global_offsets);
        global_samples = NULL;
        global_sizes = NULL;
        global_offsets = NULL;
    }
}

int main(int argc, char *argv[])
{
    int A[27] = {15, 46, 48, 93, 39, 6, 72, 91, 14, 36, 69, 40, 89, 61, 97, 12, 21, 54, 53, 97, 84, 58, 32, 27, 33, 72, 20};
    double time_start, time_end;
    int id, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if (id == 0)
        time_start = MPI_Wtime();
    psrs(A, 27, id, num_processes);
    if (id == 0)
    {
        time_end = MPI_Wtime();
        printf("Final: ");
        for (int i = 0; i < 27; i++)
            printf("%d ", A[i]);
        printf("\nTime: %lfs\n", time_end - time_start);
    }
    MPI_Finalize();
    return 0;
}
