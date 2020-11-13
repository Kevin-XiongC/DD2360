#include <stdio.h>
#include <random>
#include <sys/time.h>
#include <stdlib.h>
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

__global__ void launch_mover (Particle* particles,int N,int NUM_ITERATIONS){
    int id =blockIdx.x*blockDim.x+threadIdx.x;
    if(id<N)
        for(int i=0;i<NUM_ITERATIONS;i++)
        {   
            particles[id].position_update();
            particles[id].velocity.x+=gen_random(id, i, N)/5;
            particles[id].velocity.y+=gen_random(id, i, N)/4;
            particles[id].velocity.z+=gen_random(id, i, N)/3;
         }         
}

void one_timestep_cpu(Particle* particles,int iter,int N) {
    for(int i=0;i<N;i++)
    {
        particles[i].position_update();
        particles[i].velocity.x+=gen_random(i, iter, N)/5;
        particles[i].velocity.y+=gen_random(i, iter, N)/4;
        particles[i].velocity.z+=gen_random(i, iter, N)/3;   
    }
}
int main(int argc, char* argv[]) {
    double start,gpu_time=0,cpu_time=0;
    int NUM_PARTICLES = atoi(argv[1]);
    int NUM_ITERATIONS = atoi(argv[2]);
    int BLOCK_SIZE = atoi(argv[3]);

    printf("NUM_PARTICLES:%d\nNUM_ITERATIONS:%d\nBLOCK_SIZE:%d\n",NUM_PARTICLES,NUM_ITERATIONS,BLOCK_SIZE);
    int nBytes=sizeof(Particle)*NUM_PARTICLES;
    int grid_size =(NUM_PARTICLES+BLOCK_SIZE-1)/BLOCK_SIZE;

    Particle* particles=new Particle[NUM_PARTICLES];
    Particle* res=(Particle*)malloc(nBytes);



    start=cpuSecond();
    Particle* d_particles;
    cudaMalloc(&d_particles, nBytes);
    cudaMemcpy(d_particles, particles, nBytes, cudaMemcpyHostToDevice);
    gpu_time+=cpuSecond()-start;

    start=cpuSecond();
    for(int i=0;i<NUM_ITERATIONS;i++)
        one_timestep_cpu(particles,i,NUM_PARTICLES);
    cpu_time+=cpuSecond()-start;
    printf("CPU costs:%lf\n",cpu_time);


    start=cpuSecond();
    launch_mover<<<grid_size,BLOCK_SIZE>>>(d_particles,NUM_PARTICLES,NUM_ITERATIONS);
    cudaDeviceSynchronize();
    cudaMemcpy(res, d_particles, nBytes, cudaMemcpyDeviceToHost);
    gpu_time+=cpuSecond()-start;
    printf("GPU costs:%lf\n",gpu_time);

    int c = 0;
    for (int i=0;i<NUM_PARTICLES;i++){
        float xCPU = particles[i].position.x;
        float yCPU = particles[i].position.y;
        float zCPU = particles[i].position.z;
        float xGPU = res[i].position.x;
        float yGPU = res[i].position.y;
        float zGPU = res[i].position.z;
        if(fabs(xCPU - xGPU) > MARGIN | fabs(yCPU - yGPU) > MARGIN | fabs(zCPU - zGPU) > MARGIN) 
            c++;
    }
    printf("mismatch:%d\n",c);


    cudaFree(d_particles);


}
