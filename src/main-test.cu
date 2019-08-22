#include <iostream>
#include <cuda_profiler_api.h>


// This is a kernel that does no real work but runs at least for a specified number of clocks
__global__ void clock_block_a(clock_t *d_o, clock_t clock_count)
{
  __shared__ unsigned int smem[32768/4];

  unsigned int start_clock = (unsigned int) clock();
  smem[0] = start_clock;

  clock_t clock_offset = 0;
  while (clock_offset < clock_count)
  {
    unsigned int end_clock = (unsigned int) clock();
    clock_offset = (clock_t)(end_clock - start_clock);
  }
  d_o[0] = clock_offset;
}




int main(int argc, char *argv[])
{
  int nkernels = 8;               // number of concurrent kernels
  int nstreams = nkernels + 1;    // use one more stream than concurrent kernel
  int nbytes = nkernels * sizeof(clock_t);   // number of data bytes
  float kernel_time = 10; // time the kernel should run in ms
  float elapsed_time;   // timing variables
  int cuda_device = 0;

  
  // allocate host memory
  clock_t *a = 0;    // pointer to the array data in host memory
  cudaMallocHost((void **)&a, nbytes);

  // allocate device memory
  clock_t *d_a = 0;  // pointers to data and init value in the device memory
  cudaMalloc((void **)&d_a, nbytes);



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



  clock_t total_clocks = 0;
#if defined(__arm__) || defined(__aarch64__)
  // the kernel takes more time than the channel reset time on arm archs, so to prevent hangs reduce time_clocks.
  clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 1000));
#else
  clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif


  cudaStream_t s1;
  cudaStream_t s2;

  cudaStreamCreate(&s1);
  cudaStreamCreate(&s2);

  std::cout << "Hello 2" << std::endl;
  clock_block_a<<<1,1,0,s1>>>(&d_a[0], time_clocks);
  clock_block_a<<<1,1,0,s2>>>(&d_a[0], time_clocks);




  cudaFreeHost(a);
  cudaFree(d_a);

  return 0;
}
