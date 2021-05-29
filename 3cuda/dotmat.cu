#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>

cudaEvent_t start, stop;
float elapsedTime = 0.0;
constexpr auto N = 1000;
constexpr auto BlkNum = 100;

__global__ void dot(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = 0; j < N; j++)
    {
        c[i * N + j] = 0;
        for (int k = 0; k < N; k++)
            c[i * N + j] += a[i * N + k] * b[k * N + j];
    }
}

int main()
{
    std::ios::sync_with_stdio(false);
    std::ifstream in("in.txt");
    if (!in)
    {
        std::cerr << "Err: input\n";
        return -2;
    }
    auto a = new double[N * N], b = new double[N * N], res = new double[N * N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            in >> a[i * N + j];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            in >> b[i * N + j];
    in.close();

    double *a1, *b1, *res1;
    cudaMalloc(&a1, sizeof(double) * N * N);
    cudaMemcpy(a1, a, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMalloc(&b1, sizeof(double) * N * N);
    cudaMemcpy(b1, b, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMalloc(&res1, sizeof(double) * N * N);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dot<<<BlkNum, N / BlkNum>>>(a1, b1, res1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(res, res1, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    std::cout << "Running Time: " << elapsedTime << "s" << std::endl;

    std::ofstream out("out.txt");
    if (!out)
    {
        std::cerr << "Err: output\n";
        return -1;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            out << std::setprecision(15) << res[i * N + j] << ",";
        out << "\n";
    }
    out.close();
    free(a);
    free(b);
    free(res);
    cudaFree(a1);
    cudaFree(b1);
    cudaFree(res1);
    return 0;
}
