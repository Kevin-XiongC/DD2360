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

__global__ void one_step (Particle* particles,int iter,int NUM_PARTICLES,int offset,int id_stream,int Batch_size){
    int id =blockIdx.x*blockDim.x+threadIdx.x+offset;
    if(id<NUM_PARTICLES&&id<(id_stream+1)*Batch_size){
        particles[id].position_update();
        particles[id].velocity.x+=gen_random(id, iter, NUM_PARTICLES)/5;
        particles[id].velocity.y+=gen_random(id, iter, NUM_PARTICLES)/4;
        particles[id].velocity.z+=gen_random(id, iter, NUM_PARTICLES)/3;
    }
}
Particle* init(int NUM_PARTICLES){
      int nBytes= sizeof(Particle)*NUM_PARTICLES;
      Particle* particles;
      cudaMallocHost(&particles, nBytes);
      for (int i=0;i<NUM_PARTICLES;i++){
          particles[i].position.x=Uniform();
          particles[i].position.y=Uniform();
          particles[i].position.z=Uniform();
          particles[i].velocity.x=Uniform()/4;
          particles[i].velocity.y=Uniform()/4;
          particles[i].velocity.z=Uniform()/4;
      }
      return particles;
}


int main(int argc, char* argv[]) {
    double start,gpu_time=0;
    int NUM_PARTICLES = 10000000;
    int NUM_ITERATIONS = 100;
    int BLOCK_SIZE = 256;
    int NUM_STREAMS=2;

  //  printf("NUM_PARTICLES:%d\nNUM_ITERATIONS:%d\nBLOCK_SIZE:%d\n",NUM_PARTICLES,NUM_ITERATIONS,BLOCK_SIZE);
    int nBytes=sizeof(Particle)*NUM_PARTICLES;
    int grid_size =(NUM_PARTICLES+BLOCK_SIZE-1)/BLOCK_SIZE;
    int Batch_size = NUM_PARTICLES / NUM_STREAMS;
    int Batch_Bytes = Batch_size * sizeof(Particle);

    Particle* particles=init(NUM_PARTICLES);
    cudaStream_t* streams =(cudaStream_t*)malloc(sizeof(cudaStream_t)*NUM_STREAMS);
    for(int i = 0; i < NUM_STREAMS; i++)
      cudaStreamCreate(streams+i);
    start=cpuSecond();
    Particle* d_particles;
    cudaMalloc(&d_particles, nBytes);
    for(int i=0;i<NUM_ITERATIONS;i++)
        for(int j=0;j<NUM_STREAMS;j++){
          int offset = Batch_size*j;
          cudaMemcpyAsync(d_particles+offset, particles+offset,Batch_Bytes, cudaMemcpyHostToDevice,streams[j]);
          one_step<<<grid_size,BLOCK_SIZE,0,streams[j]>>>(d_particles,i,NUM_ITERATIONS,offset,j,Batch_size);   
          cudaMemcpyAsync(particles+offset, d_particles+offset,Batch_Bytes, cudaMemcpyDeviceToHost,streams[j]);   
        } 
    cudaDeviceSynchronize();
    gpu_time+=cpuSecond()-start;
    printf("GPU costs:%lfs\n",gpu_time);
    cudaFreeHost(d_particles);
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamDestroy(streams[i]);
    }
}