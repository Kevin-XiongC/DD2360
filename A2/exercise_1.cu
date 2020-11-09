#include<stdio.h>

__global__ void Hello (){
    int id = blockIdx.y;
    printf("hello from thread %d",id);
}


int main(){   
    Hello<<<1,256>>>();
    return 0;

}
