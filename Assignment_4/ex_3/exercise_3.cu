#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <CL/cl.h>
#include <math.h>


#define MARGIN (1e-6)

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp, NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// This is a macro for checking the error variable.
#define CHK_ERROR(err) if (err != CL_SUCCESS) fprintf(stderr, "Error: %s\n", clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char *clGetErrorString(int errorCode) {
  switch (errorCode) {
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case -69: return "CL_INVALID_PIPE_SIZE";
    case -70: return "CL_INVALID_DEVICE_QUEUE";
    case -71: return "CL_INVALID_SPEC_ID";
    case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
    case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
    case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
    case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
    case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
    case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
    case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
    case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
    case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
    case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
    case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
    case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
    case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
    case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
    case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
    case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
    case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
    case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
    default: return "CL_UNKNOWN_ERROR";
  }
}



double Uniform(){
    return rand()%100*0.00001;
}

double gen_random(int id, int iter, int NUM_PARTICLES) {
    return 1e-3*((1234*id+iter) % NUM_PARTICLES);
}
typedef struct bbb{
  double x,y,z;
  double asda; // for alignment
}_double3;
typedef struct _Particle{
    _double3 position,velocity;
}Particle;
Particle* init(int NUM_PARTICLES){
      int nBytes= sizeof(Particle)*NUM_PARTICLES;
      Particle* particles;
      particles=(Particle*)malloc(nBytes);
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

void one_timestep_cpu(Particle* particles,int iter,int N) {
    for(int id=0;id<N;id++)
    {
        particles[id].position.x+=particles[id].velocity.x;
        particles[id].position.y+=particles[id].velocity.y;
        particles[id].position.z+=particles[id].velocity.z;
        particles[id].velocity.x+=gen_random(id, iter, N)/5;
        particles[id].velocity.y+=gen_random(id, iter, N)/4;
        particles[id].velocity.z+=gen_random(id, iter, N)/3;   
    }
}

//TODO: Write your kernel here
const char *mykernel = "\
typedef struct _Particle{\
    double3 position,velocity;\
}Particle;\
double gen_random(int id, int iter, int NUM_PARTICLES) {\
    return 1e-3*((1234*id+iter) % NUM_PARTICLES);\
}\
__kernel void launch(__global Particle* particles,__global int* Nt,__global int* NUM_ITERATIONSt){\
    int id =get_global_id(0);\
    int N=*Nt;\
    int NUM_ITERATIONS=*NUM_ITERATIONSt;\
    if(id<N)\
      for(int i=0;i<NUM_ITERATIONS;i++){\
        particles[id].position.x+=particles[id].velocity.x;\
        particles[id].position.y+=particles[id].velocity.y;\
        particles[id].position.z+=particles[id].velocity.z;\
        particles[id].velocity.x+=gen_random(id, i, N)/5;\
        particles[id].velocity.y+=gen_random(id, i, N)/4;\
        particles[id].velocity.z+=gen_random(id, i, N)/3;\
      }\
}\
";

int main(int argc, char **argv) {
  cl_platform_id *platforms; cl_uint n_platform;

  // Find OpenCL Platforms
  cl_int err = clGetPlatformIDs(0, NULL, &n_platform); CHK_ERROR(err);
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * n_platform);
  err = clGetPlatformIDs(n_platform, platforms, NULL); CHK_ERROR(err);

  // Find and sort devices
  cl_device_id *device_list; cl_uint n_devices;
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices); CHK_ERROR(err);
  device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * n_devices);
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL); CHK_ERROR(err);
  

  // Create and initialize an OpenCL context
  cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err); CHK_ERROR(err);


  

  // Create a command queue
  cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err); CHK_ERROR(err); 

  double start,gpu_time=0,cpu_time=0;
  int NUM_PARTICLES = atoi(argv[1]);
  int NUM_ITERATIONS = atoi(argv[2]);
  int BLOCK_SIZE = atoi(argv[3]);

 // printf("NUM_PARTICLES:%d\nNUM_ITERATIONS:%d\nBLOCK_SIZE:%d\n",NUM_PARTICLES,NUM_ITERATIONS,BLOCK_SIZE);
  int nBytes=sizeof(Particle)*NUM_PARTICLES;
  int grid_size =(NUM_PARTICLES+BLOCK_SIZE-1)/BLOCK_SIZE;

  Particle* particles=init(NUM_PARTICLES);
  Particle* res=(Particle*)malloc(nBytes);
  start=cpuSecond();
  cl_mem p_dev = clCreateBuffer(context, CL_MEM_READ_WRITE, nBytes, NULL, &err); CHK_ERROR(err);
  cl_mem p_N = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err); CHK_ERROR(err);
  cl_mem p_N_it = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err); CHK_ERROR(err);
  
  err = clEnqueueWriteBuffer(cmd_queue, p_dev, CL_TRUE, 0, nBytes, particles, 0, NULL, NULL); CHK_ERROR(err);
  err = clEnqueueWriteBuffer(cmd_queue, p_N, CL_TRUE, 0, sizeof(int), &NUM_PARTICLES, 0, NULL, NULL); CHK_ERROR(err);
  err = clEnqueueWriteBuffer(cmd_queue, p_N_it, CL_TRUE, 0, sizeof(int), &NUM_ITERATIONS, 0, NULL, NULL); CHK_ERROR(err);
  gpu_time+=cpuSecond()-start;







  start=cpuSecond();
  for(int i=0;i<NUM_ITERATIONS;i++)
     one_timestep_cpu(particles,i,NUM_PARTICLES);
  cpu_time+=cpuSecond()-start;
  printf("CPU costs:%lf\n",cpu_time);

  int id=1000;
  //printf("CPU:v:\n%f,%f,%f\n",particles[id].velocity.x,particles[id].velocity.y,particles[id].velocity.z);
  //printf("p:\n%f,%f,%f\n",particles[id].position.x,particles[id].position.y,particles[id].position.z);
  

  /* Insert your own code here */
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&mykernel, NULL, &err);
  err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);CHK_ERROR(err);

  cl_kernel kernel = clCreateKernel(program, "launch", &err);CHK_ERROR(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&p_dev); CHK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&p_N); CHK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&p_N_it); CHK_ERROR(err);

  size_t workgroup_size = BLOCK_SIZE;
  size_t n_workitem = grid_size*BLOCK_SIZE;
 // printf("n_workitem=%ld,workgroup_size=%ld\n",n_workitem,workgroup_size);
  start = cpuSecond();
  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &n_workitem, &workgroup_size, 0, NULL, NULL);CHK_ERROR(err);
  err = clEnqueueReadBuffer(cmd_queue, p_dev, CL_TRUE, 0, nBytes, res, 0, NULL, NULL); CHK_ERROR(err);
  clFinish(cmd_queue);
  gpu_time+=cpuSecond()-start;
  printf("Done! GPU costs %lf\n", gpu_time);

  err = clFlush(cmd_queue); CHK_ERROR(err);
  err = clFinish(cmd_queue); CHK_ERROR(err);

  // test the result
  int c = 0;
  for (int i=0;i<NUM_PARTICLES;i++){
      double xCPU = particles[i].position.x;
      double yCPU = particles[i].position.y;
      double zCPU = particles[i].position.z;
      double xGPU = res[i].position.x;
      double yGPU = res[i].position.y;
      double zGPU = res[i].position.z;
      if(fabs(xCPU - xGPU) > MARGIN | fabs(yCPU - yGPU) > MARGIN | fabs(zCPU - zGPU) > MARGIN) 
          c++;
  }
 // printf("mismatch:%d\n",c);

  // Finally, release all that we have allocated.
  err = clReleaseCommandQueue(cmd_queue); CHK_ERROR(err);
  err = clReleaseContext(context); CHK_ERROR(err);
  free(platforms);
  free(device_list);
  
  return 0;
}