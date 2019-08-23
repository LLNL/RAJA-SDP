#include <iostream>
#include <cuda_profiler_api.h>
#include "device2.hpp"


template <typename LOOP_BODY>
__global__ void forall_kernel_gpu2(int start, int length, LOOP_BODY body)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < length) {
    body(idx);
  }
}


template <typename LOOP_BODY>
void forall(camp::devices::Cuda dev, int begin, int end, LOOP_BODY&& body)
{
  size_t blockSize = 32;
  size_t gridSize = (end - begin + blockSize - 1) / blockSize;

  forall_kernel_gpu2<<<gridSize, blockSize, 0, dev.get_stream()>>>(begin, end - begin, body);
}


// This is a kernel that does no real work but runs at least for a specified number of clocks
__global__ void clock_block_a(clock_t clock_count)
{
  unsigned int start_clock = (unsigned int) clock();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count)
  {
    unsigned int end_clock = (unsigned int) clock();
    clock_offset = (clock_t)(end_clock - start_clock);
  }
}


int main(int argc, char *argv[])
{
  float kernel_time = 20; // time the kernel should run in ms
  int cuda_device = 0;
  int N = 30000;
  
  cudaDeviceProp deviceProp;
  cudaGetDevice(&cuda_device);
  cudaGetDeviceProperties(&deviceProp, cuda_device);
  if ((deviceProp.concurrentKernels == 0))
  {
    printf("> GPU does not support concurrent kernel execution\n");
    printf("  CUDA kernel runs will be serialized\n");
  }
  printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
   deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

#if defined(__arm__) || defined(__aarch64__)
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 1000));
#else
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif


  // -----------------------------------------------------------------------

  camp::devices::Cuda cudev1;
  camp::devices::Cuda cudev2;
  float * m1 = cudev1.allocate<float>(N);
  float * m2 = cudev2.allocate<float>(N);


  auto clock_lambda_1 = [=] __device__ (int idx) {
    m1[idx] = idx * 2;
    unsigned int start_clock = (unsigned int) clock();
    clock_t clock_offset = 0;
    while (clock_offset < time_clocks)
    {
      unsigned int end_clock = (unsigned int) clock();
      clock_offset = (clock_t)(end_clock - start_clock);
    }
  };

  auto clock_lambda_2 = [=] __device__ (int idx) {
    m2[idx] = 1234;
    unsigned int start_clock = (unsigned int) clock();
    clock_t clock_offset = 0;
    while (clock_offset < time_clocks)
    {
      unsigned int end_clock = (unsigned int) clock();
      clock_offset = (clock_t)(end_clock - start_clock);
    }
  };

  auto clock_lambda_3 = [=] __device__ (int idx) {
    float val = m1[idx];
    m1[idx] = val * val;
    unsigned int start_clock = (unsigned int) clock();
    clock_t clock_offset = 0;
    while (clock_offset < time_clocks)
    {
      unsigned int end_clock = (unsigned int) clock();
      clock_offset = (clock_t)(end_clock - start_clock);
    }
  };


  forall(cudev1, 0, N, clock_lambda_1);
  forall(cudev2, 0, N, clock_lambda_2);
  forall(cudev1, 0, N, clock_lambda_3);

  cudaDeviceSynchronize();

  // -----------------------------------------------------------------------
  

  std::cout << "---------- M1 = (idx * 2) ^ 2 ----------" << std::endl;
  for (int i = 0; i < 15; i++) {
    std::cout << m1[i] << std::endl;
  }

  std::cout << "---------- M2 = 1234 ----------" << std::endl;
  for (int i = 0; i < 15; i++) {
    std::cout << m2[i] << std::endl;
  }

  cudaDeviceReset();
  return 0;
}
