#include<stdio.h>

__global__ void Hello (){
    int id =blockIdx.x*blockDim.x+threadIdx.x;
    printf("hello from thread %d",id);
}


int main(){   
    Hello<<<1,256>>>();
    cudaDeviceSynchronize();
    return 0;

}
