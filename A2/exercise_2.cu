#include<stdio.h>
#include <sys/time.h>
#define N 1<<17
#define margin 1e-6



double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void SAXPY (float* x,float* y,float* a){
//AX+Y
    int id =blockIdx.x*blockDim.x+threadIdx.x;
    if(id<N)
        y[id]+=*a+x[id];
}

int main(){   
    float* x,*y,a,*r;
    uint gridsize=(N+256-1)/256;
    uint nBytes=sizeof(float)*N;
    x=(float*)malloc(nBytes);
    y=(float*)malloc(nBytes);
    r=(float*)malloc(nBytes);
    for(int i=0;i<N;i++)
    {
        x[i]=i;
        y[i]=2*i;
    }
    a=1.1;

    float *d_x, *d_y, *d_a;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_a, sizeof(float));
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_a, (void*)&a, sizeof(float), cudaMemcpyHostToDevice);

    double start = cpuSecond();
    for(int i=0;i<N;i++)
    {
        y[i]+=a*x[i];
        r[i]=y[i];
    }
    printf("CPU costs %lf\n",cpuSecond()-start);

    start=cpuSecond();
    SAXPY<<<gridsize,256>>>(d_x, d_y, d_a);
    cudaDeviceSynchronize();
    printf("GPU costs %lf\n",cpuSecond()-start);
    cudaMemcpy((void*)y, (void*)d_y, nBytes, cudaMemcpyDeviceToHost);

    int c=0;
    for(int i=0;i<N;i++)
        c+=abs(y[i]-r[i])>margin?1:0;
    printf("%d mismatches\n");
      
    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
