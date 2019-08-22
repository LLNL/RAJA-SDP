#include <iostream>
#include <typeinfo>
#include "forall.hpp"

#define HOST_DEVICE __host__ __device__

using namespace std;

int main(int argc, char *argv[])
{
  int option = 0;
  if (argc > 1) option = atoi(argv[1]);  

  auto lambda = [] HOST_DEVICE (int tid) {
    int sum = 0;
    for (int i = 0; i < tid; i++){
      sum += sqrt(pow(3.14159,i));
    }
  };

  
  switch(option){ 
    case 0:
      std::cout << "Running Sequentially" << std::endl;
      sequential s;
      forall(s, 0, 30000, lambda);
      break;
    case 1:
      std::cout << "Running On GPU" << std::endl;
      
      camp::devices::Cuda cudaDevice;
      forall(cudaDevice, 0, 30000, lambda);

      camp::devices::Cuda cudaDevice2;
      forall(cudaDevice2, 0, 30000, lambda);

      forall(cudaDevice, 0, 30000, lambda);
      break;
  }

  return 0;
}
