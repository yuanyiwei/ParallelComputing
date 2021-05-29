#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstdio>

cudaEvent_t start, stop;
float elapsedTime = 0.0;
constexpr auto N = 1000;
constexpr auto BlkNum = 100;

__global__ void dot(double *a, double *b, double *res);

int main()
{
  std::ios::sync_with_stdio(false);
  std::ifstream in("in.txt");
  if (!in)
  {
    std::cerr << "Getting input failed\n";
    return -2;
  }
  auto a = new double[N * N], b = new double[N], res = new double[N];
  for (auto i = 0; i < N; i++)
  {
    for (auto j = 0; j < N; j++)
    {
      in >> a[i * N + j];
    }
  }
  for (auto i = 0; i < N; i++)
  {
    in >> b[i];
  }
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
    std::cerr << "Getting output failed\n";
    return -1;
  }
  for (auto i = 0; i < N; i++)
  {
    out << std::setprecision(15) << res[i] << "\n";
  }
  out.close();
  return 0;
}

__global__ void dot(double *a, double *b, double *res)
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  res[i] = 0;
  for (auto k = 0; k < N; k++)
  {
    res[i] += a[i * N + k] * b[k];
  }
}
