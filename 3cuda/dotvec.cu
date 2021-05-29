#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>

cudaEvent_t start, stop;
float elapsedTime = 0.0;
constexpr int N = 1000;
constexpr int BlkNum = 100;

__global__ void dot(double *a, double *b, double *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = 0;
    for (int k = 0; k < N; k++)
        c[i] += a[i * N + k] * b[k];
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
    auto a = new double[N * N], b = new double[N], res = new double[N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            in >> a[i * N + j];
    for (int i = 0; i < N; i++)
        in >> b[i];
    in.close();

    double *a1, *b1, *res1;
    cudaMalloc(&a1, sizeof(double) * N * N);
    cudaMemcpy(a1, a, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMalloc(&b1, sizeof(double) * N);
    cudaMemcpy(b1, b, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMalloc(&res1, sizeof(double) * N);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    dot<<<BlkNum, N / BlkNum>>>(a1, b1, res1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(res, res1, sizeof(double) * N, cudaMemcpyDeviceToHost);
    std::cout << "Running Time: " << elapsedTime << "s" << std::endl;

    std::ofstream out("out.txt");
    if (!out)
    {
        std::cerr << "Err: output\n";
        return -1;
    }
    for (int i = 0; i < N; i++)
        out << std::setprecision(15) << res[i] << "\n";
    out.close();
    free(a);
    free(b);
    free(res);
    cudaFree(a1);
    cudaFree(b1);
    cudaFree(res1);
    return 0;
}
