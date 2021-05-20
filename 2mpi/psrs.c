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

void MergeSort(int *A, int l, int r)
{
    if (l < r)
    {
        int m = (l + r) / 2;
        MergeSort(A, l, m);
        MergeSort(A, m + 1, r);
        Merge(A, l, m, r);
    }
}

/**************************************************
* 对A[0 : n - 1]进行PSRS排序
***************************************************/
void PSRS(int *A, int n, int id, int num_processes)
{
     //每个进程都会执行这个函数，这里面的变量每个进程都有一份，因此都是局部的（对于当前进程而言）
    int per;
    int *samples, *global_samples; //global表示这个变量是主进程会使用的，但事实上每个进程都声明了
    int *pivots;
    int *sizes, *newsizes;
    int *offsets, *newoffsets;
    int *newdatas;
    int newdatassize;
    int *global_sizes;
    int *global_offsets;
    //-------------------------------------------------------------------------------------------------------------------

    per = n / num_processes;                              //A的n个元素划分为num_processes段，每个进程处理per个元素
    samples = (int *)malloc(num_processes * sizeof(int)); //当前进程的采样数组
    pivots = (int *)malloc(num_processes * sizeof(int));  //num_processes - 1 个主元，最后一个设为INF，作为哨兵
    if (id == 0)
    {                                                                                //主进程申请全局采样数组
        global_samples = (int *)malloc(num_processes * num_processes * sizeof(int)); //正则采样数为 num_processes * num_processes
    }
    MPI_Barrier(MPI_COMM_WORLD); //设置路障，同步所有进程
    //-------------------------------------------------------------------------------------------------------------------
    //（1）均匀划分，当前进程对A中属于自己的部分进行串行归并排序
    MergeSort(A, id * per, (id + 1) * per - 1); //这里没有把A中对应当前进程的数据复制到当前进程，而是直接对A部分排序
    //（2）正则采样，当前进程选出 num_processes 个样本放在local_sample中
    for (int k = 0; k < num_processes; k++)
        samples[k] = A[id * per + k * per / num_processes];
    //主进程的sample收集各进程的local_sample，共 num_processes * num_processes 个
    MPI_Gather(samples, num_processes, MPI_INT, global_samples, num_processes, MPI_INT, 0, MPI_COMM_WORLD);
    //-------------------------------------------------------------------------------------------------------------------
    //（3）采样排序 （4）选择主元
    if (id == 0)
    {                                                                    //主进程
        MergeSort(global_samples, 0, num_processes * num_processes - 1); //对采样的num_processes * num_processes个样本进行排序
        for (int k = 0; k < num_processes - 1; k++)                      //选出num_processes - 1个主元
            pivots[k] = global_samples[(k + 1) * num_processes];
        pivots[num_processes - 1] = INF; //哨兵
    }
    //主进程向各个进程广播，所有进程拥有一样的pivots数组
    MPI_Bcast(pivots, num_processes, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //-------------------------------------------------------------------------------------------------------------------
    sizes = (int *)calloc(num_processes, sizeof(int));      //当前进程的per个元素根据主元划分之后的每段的长度，用calloc分配后自动初始化为0
    offsets = (int *)calloc(num_processes, sizeof(int));    //当前进程的per个元素根据主元划分之后的每段的起始位置，用calloc分配后自动初始化为0
    newsizes = (int *)calloc(num_processes, sizeof(int));   //当前进程在进行全局交换之后的每段的长度，用calloc分配后自动初始化为0
    newoffsets = (int *)calloc(num_processes, sizeof(int)); //当前进程在进行全局交换之后的每段的起始位置，用calloc分配后自动初始化为0
    //（5）主元划分
    for (int k = 0, j = id * per; j < id * per + per; j++)
    {                         //当前进程对自己的per个元素按主元划分为num_processes段，此处不处理数据，只计算每段大小
        if (A[j] < pivots[k]) //如果之前不用哨兵，最后一段要单独考虑
            sizes[k]++;
        else
            sizes[++k]++;
    }
    //（6）全局交换
    //多对多全局交换消息，首先每个进程向每个接收者发送接收者对应的【段的大小】
    MPI_Alltoall(sizes, 1, MPI_INT, newsizes, 1, MPI_INT, MPI_COMM_WORLD);
    //计算原来的段偏移数组，新的段偏移数组，新的数据大小
    newdatassize = newsizes[0];
    for (int k = 1; k < num_processes; k++)
    {
        offsets[k] = offsets[k - 1] + sizes[k - 1];
        newoffsets[k] = newoffsets[k - 1] + newsizes[k - 1];
        newdatassize += newsizes[k];
    }
    //申请当前进程新的数据空间
    newdatas = (int *)malloc(newdatassize * sizeof(int)); //当前进程在进行全局交换之后的数据，是由交换后的各段组合而成的
    //多对多全局交换消息，每个进程向每个接收者发送接收者对应的【段】
    MPI_Alltoallv(&(A[id * per]), sizes, offsets, MPI_INT, newdatas, newsizes, newoffsets, MPI_INT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //-------------------------------------------------------------------------------------------------------------------
    //（7）当前进程归并排序自己的新数据
    MergeSort(newdatas, 0, newdatassize - 1);
    MPI_Barrier(MPI_COMM_WORLD);
    //（8）主进程收集各个进程的数据，写回A
    //首先收集各进程新数据的大小
    if (id == 0)
        global_sizes = (int *)calloc(num_processes, sizeof(int));
    MPI_Gather(&newdatassize, 1, MPI_INT, global_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //主进程计算即将搜集的各进程数据的起始位置
    if (id == 0)
    {
        global_offsets = (int *)calloc(num_processes, sizeof(int));
        for (int k = 1; k < num_processes; k++)
            global_offsets[k] = global_offsets[k - 1] + global_sizes[k - 1];
    }
    //主进程收集各个进程的数据
    MPI_Gatherv(newdatas, newdatassize, MPI_INT, A, global_sizes, global_offsets, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    //-------------------------------------------------------------------------------------------------------------------
    //销毁动态数组
    free(samples);
    samples = NULL;
    free(pivots);
    pivots = NULL;
    free(sizes);
    sizes = NULL;
    free(offsets);
    offsets = NULL;
    free(newdatas);
    newdatas = NULL;
    free(newsizes);
    newsizes = NULL;
    free(newoffsets);
    newoffsets = NULL;
    if (id == 0)
    {
        free(global_samples);
        global_samples = NULL;
        free(global_sizes);
        global_sizes = NULL;
        free(global_offsets);
        global_offsets = NULL;
    }
}

int main(int argc, char *argv[])
{
    int A[27] = {15, 46, 48, 93, 39, 6, 72, 91, 14, 36, 69, 40, 89, 61, 97, 12, 21, 54, 53, 97, 84, 58, 32, 27, 33, 72, 20};
    double t1, t2;
    int id, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    if (id == 0)
        t1 = MPI_Wtime();
    PSRS(A, 27, id, num_processes);
    if (id == 0)
    {
        t2 = MPI_Wtime();
        printf("time: %lfs\n", t2 - t1);
        for (int i = 0; i < 27; i++)
            printf("%d ", A[i]);
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}