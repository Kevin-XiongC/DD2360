#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <curand_kernel.h>
#include <curand.h>

#define SEED     921
#define NUM_ITER 25600000
#define TRIALS_PER_THREAD 100000

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void pi_seq(int *g_count, curandState *states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize the random state
    int seed = id;
    curand_init(seed, id, 0, &states[id]);

    double x, y, z;
    for (int i = 0; i < TRIALS_PER_THREAD; i++) {
        x = curand_uniform(&states[id]);
        y = curand_uniform(&states[id]);
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0) g_count[id]++;
    }
}

int main(int argc, char* argv[]) {
    int count = 0;
    double x, y, z, pi;
    
    srand(SEED); // Important: Multiply SEED by "rank" when you introduce MPI!
    
    double cpu_start = cpuSecond();
    // Calculate PI following a Monte Carlo method
    for (int iter = 0; iter < NUM_ITER; iter++) {
        // Generate random (X,Y) points
        x = (double)random() / (double)RAND_MAX;
        y = (double)random() / (double)RAND_MAX;
        z = sqrt((x*x) + (y*y));
        
        // Check if point is in unit circle
        if (z <= 1.0) count++;
    }
    
    // Estimate Pi and display the result
    pi = ((double)count / (double)NUM_ITER) * 4.0;
    double cpu_end = cpuSecond();
    printf("The result estimated by CPU is %lf in %lfs\n", pi, cpu_end - cpu_start);

    int num_block = 1, num_threads = 256;
    // Allocate curandState for every CUDA thread on host
    curandState *dev_random;
    cudaMalloc(&dev_random, num_block * num_threads * sizeof(curandState));

    int *g_count;
    double gpu_start = cpuSecond();
    cudaMallocManaged(&g_count, sizeof(int) * num_threads);
    pi_seq<<<num_block, num_threads>>>(g_count, dev_random);
    cudaDeviceSynchronize();
    
    int g_sum = 0;
    for (int i = 0; i < num_threads; i++)
        g_sum += g_count[i];
    
    double g_pi = ((double)g_sum / ((double)TRIALS_PER_THREAD * num_threads)) * 4.0;
    double gpu_end = cpuSecond();
    
    printf("The result estimated by GPU is %lf in %lfs\n", g_pi, gpu_end - gpu_start);
    
    return 0;
}

