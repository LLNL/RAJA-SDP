#include <iostream>
#include <typeinfo>
#include <future>
#include "forall.hpp"

#define HOST_DEVICE __host__ __device__

#define B_SIZE 32
#define G_SIZE ((1 << 20) + B_SIZE - 1) / B_SIZE

template <typename LOOP_BODY>
__global__ void forall_kernel_gpu2(int start, int length, LOOP_BODY body, float * mem)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (idx < length) {
    body(idx, mem);
  }
}

int main(int argc, char *argv[])
{
  int option = 0;
  if (argc > 1) option = atoi(argv[1]);  

  auto lambda = [] HOST_DEVICE (int tid, int grid, int block, int N, float * x) {
    for (int i = tid; i < N; i+= block * grid){
      x[i] = sqrt(pow(3.14159,i));
    }
  };

  camp::devices::Cuda cudaDevice;
  camp::devices::Cuda cudaDevice2;

  int N = 1 << 20; 

  float * m1 = cudaDevice.allocate<float>(N);
  float * m2 = cudaDevice2.allocate<float>(N);

  switch(option){ 
    case 0:
      //std::cout << "Running Sequentially" << std::endl;
      //sequential s;
      //forall(s, 0, 30000, lambda, m1);
      break;
    case 1:
      std::cout << "Running On GPU" << std::endl;
      
      forall(cudaDevice, 0, N, lambda, m1);

      forall(cudaDevice2, 0, N, lambda, m2);

      forall(cudaDevice, 0, N, lambda, m1);
      break;
    case 2:
      //std::cout << "Raw CUDA Calls" << std::endl;
      //forall_kernel_gpu2<<<G_SIZE, B_SIZE >>>(0, N, lambda, m1);
      //forall_kernel_gpu2<<<G_SIZE, B_SIZE, 0, cudaDevice2.get_stream()>>>(0, N, lambda, m1);
      //forall_kernel_gpu2<<<G_SIZE, B_SIZE, 0, cudaDevice.get_stream()>>>(0, N, lambda, m1);
      break;
  }

  return 0;
}
