#include <stdio.h>
#include <random> 
#include <sys/time.h>
#define SEED 123
#define MARGIN 1e-6
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

float Uniform(){
    std::default_random_engine generator;
    std::uniform_real_distribution<float> uniform(-10,10);
    return uniform(generator);
}

__host__ __device__ float gen_random(int id, int iter, int NUM_PARTICLES) {
    return (SEED*id+iter) % NUM_PARTICLES;
}


class Particle {
public:
    float3 position,velocity;
    Particle() {
        position.x=Uniform();
        position.y=Uniform();
        position.z=Uniform();
        velocity.x=Uniform()/4;
        velocity.y=Uniform()/4;
        velocity.z=Uniform()/4;
    }
    __device__ __host__ void position_update() {
        position.x+=velocity.x;
        position.y+=velocity.y;
        position.z+=velocity.z;
    }
};

__global__ void one_step (Particle* particles,int iter,int NUM_PARTICLES){
    int id =blockIdx.x*blockDim.x+threadIdx.x;
    if(id<NUM_PARTICLES){
        particles[id].position_update();
        particles[id].velocity.x+=gen_random(id, iter, NUM_PARTICLES)/5;
        particles[id].velocity.y+=gen_random(id, iter, NUM_PARTICLES)/4;
        particles[id].velocity.z+=gen_random(id, iter, NUM_PARTICLES)/3;
    }
}

int main(int argc, char* argv[]) {
    double start,gpu_time=0;
    int NUM_PARTICLES = 10000000;
    int NUM_ITERATIONS = 100;
    int BLOCK_SIZE = 256;

   // printf("NUM_PARTICLES:%d\nNUM_ITERATIONS:%d\nBLOCK_SIZE:%d\n",NUM_PARTICLES,NUM_ITERATIONS,BLOCK_SIZE);
    int nBytes=sizeof(Particle)*NUM_PARTICLES;
    int grid_size =(NUM_PARTICLES+BLOCK_SIZE-1)/BLOCK_SIZE;

    Particle* particles=new Particle[NUM_PARTICLES];



    start=cpuSecond();
    Particle* d_particles;
    cudaMalloc(&d_particles, nBytes);
    for(int i=0;i<NUM_ITERATIONS;i++){
      cudaMemcpy(d_particles, particles, nBytes, cudaMemcpyHostToDevice);
      one_step<<<grid_size,BLOCK_SIZE>>>(d_particles,i,NUM_ITERATIONS);   
      cudaDeviceSynchronize();
      cudaMemcpy(particles, d_particles, nBytes, cudaMemcpyDeviceToHost);
    }
    gpu_time+=cpuSecond()-start;
    printf("GPU costs:%lfs\n",gpu_time);
    cudaFree(d_particles);
}