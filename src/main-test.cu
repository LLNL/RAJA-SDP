#include <iostream>
#include <cuda_profiler_api.h>
#include "device2.hpp"


template <typename LOOP_BODY>
__global__ void forall_kernel_gpu2(int start, int length, LOOP_BODY body)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < length) {
    body();
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

  
  // allocate host memory
  //clock_t *a = 0;    // pointer to the array data in host memory
  //cudaMallocHost((void **)&a, nbytes);

  // allocate device memory
  //clock_t *d_a = 0;  // pointers to data and init value in the device memory
  //cudaMalloc((void **)&d_a, nbytes);


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


  camp::devices::Cuda cudev1;
  camp::devices::Cuda cudev2;


#if defined(__arm__) || defined(__aarch64__)
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 1000));
#else
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif


  auto clock_lambda = [=] __device__ () {
    unsigned int start_clock = (unsigned int) clock();
    clock_t clock_offset = 0;
    while (clock_offset < time_clocks)
    {
      unsigned int end_clock = (unsigned int) clock();
      clock_offset = (clock_t)(end_clock - start_clock);
    }
  };
  


  forall(cudev1, 0, 1, clock_lambda);
  forall(cudev2, 0, 1, clock_lambda);
  forall(cudev1, 0, 1, clock_lambda);

  return 0;
}
