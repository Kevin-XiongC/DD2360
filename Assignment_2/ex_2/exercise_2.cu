#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

#define MARGIN 1e-6
#define ARRAY_SIZE 100000

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp, NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void SAXPY (double *x,double *y, double a) {
    // Y = AX + Y
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < ARRAY_SIZE; i += stride)
        y[i] += a * x[i];
}

int main(int argc, char *argv[]) {   
    double *x, *y, a = 2.2, *r;
    unsigned int gridsize = (ARRAY_SIZE + 256 - 1) / 256;
    unsigned int nBytes = sizeof(double) * ARRAY_SIZE;
    x = (double*)malloc(nBytes);
    y = (double*)malloc(nBytes);
    r = (double*)malloc(nBytes);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        x[i] = i;
        y[i] = 2 * i;
    }

    double *d_x, *d_y;
    cudaMalloc(&d_x, nBytes);
    cudaMalloc(&d_y, nBytes);
    cudaMemcpy(d_x, x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, nBytes, cudaMemcpyHostToDevice);

    printf("Computing SAXPY on the CPU...\n");
    double start = cpuSecond();
    for (int i = 0; i < ARRAY_SIZE; i++) {
        y[i] += a * x[i];
        r[i] = y[i];
    }
    printf("Done! CPU costs %lf\n", cpuSecond() - start);

    printf("Computing SAXPY on the GPU...\n");
    start = cpuSecond();
    SAXPY<<<gridsize,256>>>(d_x, d_y, a);
    cudaDeviceSynchronize();
    printf("Done! GPU costs %lf\n", cpuSecond() - start);

    printf("Comparing the output for each implementation...\n");
    cudaMemcpy(y, d_y, nBytes, cudaMemcpyDeviceToHost);
    int c = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        double diff = abs(y[i] - r[i]);
        if (diff > MARGIN) {
            c += 1;
            printf("The %d-th element doesn't match. The difference is %lf\n", i, diff);
        }
    }
    printf("Totally %d mismatch(es)\n", c);
    if (c == 0)
        printf("Correct!\n");

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    free(r);
    return 0;
}