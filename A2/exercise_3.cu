#include <stdio.h>
#include <random>
#include <sys/time.h>
#include <stdlib.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__device__ __host__ float random(bool flag){
    std::default_random_engine generator;
    std::uniform_real_distribution<float> uniform(-5,5);
    std::normal_distribution<float> Normal(0,0.8);
    return flag?uniform(generator):Normal(generator);
}
class Particle {
public:
    float3 position,velocity;
    Particle() {
        position.x=random(1);
        position.y=random(1);
        position.z=random(1);
        velocity.x=random(1)/4;
        velocity.y=random(1)/4;
        velocity.z=random(1)/4;
    }
    __device__ __host__  void velocity_update() {
        velocity.x+=random(0);
        velocity.y+=random(0);
        velocity.z+=random(0);
    }
    __device__ __host__ void position_update() {
        position.x+=velocity.x;
        position.y+=velocity.y;
        position.z+=velocity.z;
    }
};

__device__ void one_timestep(Particle& particle) {
    particle.position_update();
    particle.velocity_update();
}
__global__ void launch_mover (Particle* particles,int N,int NUM_ITERATIONS){
    int id =blockIdx.x*blockDim.x+threadIdx.x;
    if(id<N)
    {   
        for(int i=0;i<NUM_ITERATIONS;i++)
            one_timestep(particles[id]);
    }       
}

void one_timestep_cpu(Particle* particles,int N) {
    for(int i=0;i<N;i++)
    {
        particles[i].position_update();
        particles[i].velocity.x+=random(0);
        particles[i].velocity.y+=random(0);
        particles[i].velocity.z+=random(0);   
    }
}
int main(int argc, char* argv[]) {
    double start,gpu_time=0,cpu_time=0;
    int NUM_PARTICLES = atoi(argv[1]);
    int BLOCK_SIZE = atoi(argv[3]);
    int NUM_ITERATIONS = atoi(argv[2]);
    int nBtyes=sizeof(Particle)*NUM_PARTICLES;
    int grid_size =(NUM_PARTICLES+BLOCK_SIZE-1)/BLOCK_SIZE;

    Particle* particles=new Particle[NUM_PARTICLES];


    start=cpuSecond();
    Particle* d_particles;
    cudaMalloc(&d_particles, nBtyes);
    cudaMemcpy(d_particles, particles, nBtyes, cudaMemcpyHostToDevice);
    gpu_time+=cpuSecond()-start;

    start=cpuSecond();
    for(int i=0;i<NUM_ITERATIONS;i++)
        one_timestep_cpu(particles,NUM_PARTICLES);
    cpu_time+=cpuSecond()-start;
    printf("CPU costs:%lf\n",cpu_time);


    start=cpuSecond();
    launch_mover<<<grid_size,BLOCK_SIZE>>>(d_particles,NUM_PARTICLES,NUM_ITERATIONS);
    cudaDeviceSynchronize();
    gpu_time+=cpuSecond()-start;
    printf("GPU costs:%lf\n",gpu_time);
    
    cudaFree(d_particles);
    return 0;

}
