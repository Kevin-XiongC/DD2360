#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <stdlib.h>
#include <math.h>
#include "cifar10_reader.hpp"
using namespace std;

#define double double
#define DIM_INPUT (3072)
#define DIM_OUTPUT (10)
#define NUM_TRAINING (5000)
#define NUM_VALIDATION (2000)
#define NUM_TEST (1000)

#define BATCH_SIZE (100)
#define LEARNING_RATE (0.001)
#define LAMBDA (0.3)
#define EPOCHS (40)

#define GRID_DIM (1)
#define BLOCK_DIM BATCH_SIZE
#define NUM_BLOCK (dim3(GRID_DIM, GRID_DIM))
#define NUM_THREAD (dim3(DIM_OUTPUT, BATCH_SIZE))


struct Gradient {
    double *grad_W;
    double *grad_b;
};

void normalize(double *a, const int N) {
    // @param a: of size (N, 3072), that is (NUM, DIM_INPUT)
    // result: (N, 3072)
    for (int i = 0; i < N; i++) {
        double mean = 0, std = 0;

        // calculate mean
        for (int j = 0; j < DIM_INPUT; j++) mean += a[i * DIM_INPUT + j];
        mean /= DIM_INPUT;

        // calculate standard deviation
        for (int j = 0; j < DIM_INPUT; j++) std += pow(a[i * DIM_INPUT + j] - mean, 2);
        std = sqrt(std / DIM_INPUT);

        // normalize the data
        for (int j = 0; j < DIM_INPUT; j++) 
            a[i * DIM_INPUT + j] = (a[i * DIM_INPUT + j] - mean) / std;
    }
}

void initialize(double *w, double *b, const double mu, const double sigma) {
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    default_random_engine generator;
    normal_distribution<double> distribution(mu, sigma);

    // initialize w
    for (int i = 0; i < DIM_INPUT; i++) {
        for (int j = 0; j < DIM_OUTPUT; j++)
            w[i * DIM_OUTPUT + j] = distribution(generator);
    }
    // initialzie b
    for (int i = 0; i < DIM_OUTPUT; i++)
        b[i] = distribution(generator);
}

double *dot(double *a, double *b, const int n, const int m, const int p, const bool transpose) {
    // @param a: of size (n, m)
    // @param b: of size (m, p)
    // @param transpose: if transpose matrix a
    // @return: dot(a, b), of size (n, p)
    double *ab = (double *)calloc(n * p, sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++) {
                if (transpose) ab[i * p + j] += a[k * n + i] * b[k * p + j];
                else ab[i * p + j] += a[i * m + k] * b[k * p + j];
            }
        }
    }
    return ab;
}

__device__ void gpu_dot(double *a, double *b, const int n, const int m, const int p, double *r, const bool transpose) {
    // @param a: of size (n, m)
    // @param b: of size (m, p)
    // @param r: already allocated space of size (n, p), initialized 0
    // result: dot(a, b), of size (n, p)
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;

    for (int i = index_y; i < n; i += stride_y) {
        for (int j = index_x; j < p; j += stride_x) {
            for (int k = 0; k < m; k++) {
                if (transpose) atomicAdd(&r[i * p + j], a[k * n + i] * b[k * p + j]);
                else atomicAdd(&r[i * p + j], a[i * m + k] * b[k * p + j]);
            }
        }
    }
}

double *transpose(double *a, const int n, const int m) {
    // @param a: of size (n, m)
    // @return: a.T, of size (m, n)
    double *b = (double *)calloc(n * m, sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            b[j * n + i] = a[i * m + j];
    }
    return b;
}

void softmax(double *a, const int N) {
    // @param a: of size (N, 10), that is (NUM, DIM_OUTPUT)
    // result: (N, 10)
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < DIM_OUTPUT; j++) {
            double exponential = exp(a[i * DIM_OUTPUT + j]);
            a[i * DIM_OUTPUT + j] = exponential;
            sum += exponential;
        }
        for (int j = 0; j < DIM_OUTPUT; j++) a[i * DIM_OUTPUT + j] /= sum; 
    }
}

__device__ double sum[BLOCK_DIM];
__device__ void gpu_softmax(double *a, const int N) {
    // @param a: of size (N, 10), that is (NUM, DIM_OUTPUT)
    // result: (N, 10)
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;

    for (int i = index_y; i < N; i += stride_y) {
        if (index_x == 0) sum[index_y] = 0;     // initialize by the first thread
        __syncthreads();

        for (int j = index_x; j < DIM_OUTPUT; j += stride_x) {
            double exponential = exp(a[i * DIM_OUTPUT + j]);
            a[i * DIM_OUTPUT + j] = exponential;
            atomicAdd(&sum[index_y], exponential);
        }
        __syncthreads();

        for (int j = index_x; j < DIM_OUTPUT; j += stride_x) a[i * DIM_OUTPUT + j] /= sum[index_y];
        __syncthreads();
    }
}

double *evaluate(double *x, double *w, double *b, const int N) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    // @return: softmax(dot(x, w) + b), of size (N, 10)
    double *wx = dot(x, w, N, DIM_INPUT, DIM_OUTPUT, false);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DIM_OUTPUT; j++)
            wx[i * DIM_OUTPUT + j] += b[j];
    }
    softmax(wx, N);
    return wx;
}

__device__ void gpu_evaluate(double *x, double *w, double *b, const int N, double *r) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    // @param r: already allocated space of size (N, 10), initialized 0
    // result: softmax(dot(x, w) + b), of size (N, 10)
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;

    gpu_dot(x, w, N, DIM_INPUT, DIM_OUTPUT, r, false);
    __syncthreads();
    for (int i = index_y; i < N; i += stride_y) {
        for (int j = index_x; j < DIM_OUTPUT; j += stride_x) 
            atomicAdd(&r[i * DIM_OUTPUT + j], r[i * DIM_OUTPUT + j] + b[j]);
    }
    __syncthreads();
    gpu_softmax(r, N);
}

double computeCost(double *x, int *y, double *w, double *b, const int N) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param y: of size (N, 1), that is (NUM, 1)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    // @return: cost
    double *p = evaluate(x, w, b, N);
    double cost = 0;

    for (int i = 0; i < N; i++) cost += -log(p[i * DIM_OUTPUT + y[i]]);
    cost /= N;
    free(p);    // free p

    double w2 = 0;
    for (int i = 0; i < DIM_INPUT; i++) {
        for (int j = 0; j < DIM_OUTPUT; j++)
            w2 += pow(w[i * DIM_OUTPUT + j], 2);
    }
    cost += LAMBDA * w2;

    return cost;
}

__device__ double w2;
__global__ void __gpu_computeCost(double *x, int *y, double *w, double *b, const int N, double *p, double *res) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;

    gpu_evaluate(x, w, b, N, p);
    __syncthreads();
    
    if (index_x == 0) {
        for (int i = index_y; i < N; i += stride_y) {
            atomicAdd(res, -log(p[i * DIM_OUTPUT + y[i]]));
        }
    }
    __syncthreads();

    if (index_x == 0 && index_y == 0) {
        *res /= N;
        w2 = 0;
    }
    __syncthreads();
    
    for (int i = index_y; i < N; i += stride_y) {
        for (int j = index_x; j < DIM_OUTPUT; j += stride_x) 
            atomicAdd(&w2, pow(w[i * DIM_OUTPUT + j], 2));
    }
    __syncthreads();

    if (index_x == 0 && index_y == 0) {
        *res += LAMBDA * w2;
    }
}

double gpu_computeCost(double *x, int *y, double *w, double *b, const int N) {
    double *p;
    double *res;

    cudaMalloc(&p, BATCH_SIZE * DIM_OUTPUT * sizeof(double));
    cudaMemset(p, 0, BATCH_SIZE * DIM_OUTPUT * sizeof(double));
    cudaMallocManaged(&res, sizeof(double));
    cudaMemset(res, 0, sizeof(double));

    __gpu_computeCost<<<NUM_BLOCK, NUM_THREAD>>>(x, y, w, b, N, p, res);
    cudaDeviceSynchronize();
    double result = *res;

    cudaFree(p);
    cudaFree(res);
    return result;
}

double computeAccuracy(double *x, int *y, double *w, double *b, const int N) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param y: of size (N, 1), that is (NUM, 1)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    // @return: accuracy
    double *p = evaluate(x, w, b, N);
    int num_correct = 0;

    for (int i = 0; i < N; i++) {
        int argmax = -1;
        double max = -1;
        for (int j = 0; j < DIM_OUTPUT; j++) {
            if (p[i * DIM_OUTPUT + j] > max) {
                max = p[i * DIM_OUTPUT + j];
                argmax = j;
            }
        }
        if (argmax == y[i]) num_correct++;
    }
    free(p);

    return (double)num_correct/N;
}

__global__ void computeGradient(double *x, int *y, double *w, double *b, const int N, Gradient *gradient, double *p) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param y: of size (N, 1), that is (NUM, 1)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    // @param p: of size (N, 10), space for storing the result of evaluate
    // result: gradient for w and b
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;

    gpu_evaluate(x, w, b, N, p);
    __syncthreads();

    for (int i = index_y; i < N; i += stride_y) {
        for (int j = index_x; j < DIM_OUTPUT; j += stride_x) {
            if (j == y[i]) p[i * DIM_OUTPUT + j] -= 1;
        }
    }
    __syncthreads();


    gpu_dot(x, p, DIM_INPUT, N, DIM_OUTPUT, gradient->grad_W, true);
    __syncthreads();

    for (int i = index_y; i < DIM_INPUT; i += stride_y) {
        for (int j = index_x; j < DIM_OUTPUT; j += stride_x) {
            int index = i * DIM_OUTPUT + j;
            gradient->grad_W[index] = gradient->grad_W[index] / N + 2 * LAMBDA * w[index];
        }
    }
    __syncthreads();

    for (int i = index_y; i < N; i += stride_y) {
        for (int j = index_x; j < DIM_OUTPUT; j += stride_x) 
            gradient->grad_b[j] += p[i * DIM_OUTPUT + j];
    }
    __syncthreads();

    if (index_y == 0) {
        for (int i = index_x; i < DIM_OUTPUT; i += stride_x) gradient->grad_b[i] /= N;
    }
}

__global__ void updateGradient(double *w, double *b, Gradient *gradient) {
    const int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int stride_x = blockDim.x * gridDim.x;
    const int stride_y = blockDim.y * gridDim.y;

    for (int i = index_y; i < DIM_INPUT; i +=stride_y) {
        for (int j = index_x; j < DIM_OUTPUT; j += stride_x) {
            int index = i * DIM_OUTPUT + j;
            w[index] -= LEARNING_RATE * gradient->grad_W[index];
        }
    }

    if (index_y == 0) {
        for (int i = index_x; i < DIM_OUTPUT; i += stride_x) {
            b[i] -= LEARNING_RATE * gradient->grad_b[i];
        }
    }
}

void batchGradientDescent(double *x, int *y, double *w, double *b, double *x_t, int *y_t) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param y: of size (N, 1), that is (NUM, 1)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    ofstream outfile;
    outfile.open("plots/perceptron_gpu.txt");
    
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int n_batch = 0; n_batch < NUM_TRAINING / BATCH_SIZE; n_batch++) {
            int start = n_batch * BATCH_SIZE;
            double *batch_x = x + start * DIM_INPUT;
            int *batch_y = y + start;

            // cuda memory allocation
            Gradient *gradient;
            double *p;

            cudaMallocManaged(&gradient, sizeof(Gradient));
            cudaMallocManaged(&(gradient->grad_W), DIM_INPUT * DIM_OUTPUT * sizeof(double));
            cudaMemset(gradient->grad_W, 0, DIM_INPUT * DIM_OUTPUT * sizeof(double));
            cudaMallocManaged(&(gradient->grad_b), DIM_OUTPUT * sizeof(double));
            cudaMemset(gradient->grad_b, 0, DIM_OUTPUT * sizeof(double));
            cudaMalloc(&p, BATCH_SIZE * DIM_OUTPUT * sizeof(double));
            cudaMemset(p, 0, BATCH_SIZE * DIM_OUTPUT * sizeof(double));

            // compute and update gradient
            computeGradient<<<NUM_BLOCK, NUM_THREAD>>>(batch_x, batch_y, w, b, BATCH_SIZE, gradient, p);
            updateGradient<<<NUM_BLOCK, NUM_THREAD>>>(w, b, gradient);
            cudaDeviceSynchronize();

            // free memory
            cudaFree(gradient->grad_W);
            cudaFree(gradient->grad_b);
            cudaFree(gradient);
            cudaFree(p);
        }
        // compute cost on training and validation set
        double cost_training = gpu_computeCost(x, y, w, b, NUM_TRAINING);
        double cost_validation = gpu_computeCost(x_t, y_t, w, b, NUM_VALIDATION);
        printf("Epoch %d: cost_training = %lf cost_validation = %lf \n", epoch, cost_training, cost_validation);
        outfile << epoch << " " << cost_training << " " << cost_validation << endl;
    }

    outfile.close();
}


class Data {
public:
    double *training_images;    // N * 3072
    int *training_labels;    // N
    double *validation_images;
    int *validation_labels;
    double *test_images;
    int *test_labels;

    Data() {
        cudaMallocManaged(&(this->training_images), NUM_TRAINING * DIM_INPUT * sizeof(double));
        cudaMallocManaged(&(this->training_labels), NUM_TRAINING * sizeof(int));
        cudaMallocManaged(&(this->validation_images), NUM_VALIDATION * DIM_INPUT * sizeof(double));
        cudaMallocManaged(&(this->validation_labels), NUM_VALIDATION * sizeof(int));    
        this->test_images = (double *)calloc(NUM_TEST * DIM_INPUT, sizeof(double));
        this->test_labels = (int *)calloc(NUM_TEST, sizeof(int));
    }

    ~Data() {
        cudaFree(this->training_images);
        cudaFree(this->training_labels);
        cudaFree(this->validation_images);
        cudaFree(this->validation_labels);
        free(this->test_images);
        free(this->test_labels);
    }

    // read data from CIFAR10_dataset structure
    void operator=(const cifar::CIFAR10_dataset<vector, vector<uint8_t>, uint8_t>& cifar) {
        for (int i = 0; i < NUM_TRAINING; i++) {
            auto image = cifar.training_images[i];
            for (int j = 0; j < DIM_INPUT; j++) {
                this->training_images[i * DIM_INPUT + j] = image[j];
            }
            this->training_labels[i] = cifar.training_labels[i];
        }
        for (int i = 0; i < NUM_VALIDATION; i++) {
            auto image = cifar.training_images[i + NUM_TRAINING];
            for (int j = 0; j < DIM_INPUT; j++) {
                this->validation_images[i * DIM_INPUT + j] = image[j];
            }
            this->validation_labels[i] = cifar.training_labels[i + NUM_TRAINING];
        }
        for (int i = 0; i < NUM_TEST; i++) {
            auto image = cifar.test_images[i];
            for (int j = 0; j < DIM_INPUT; j++) {
                this->test_images[i * DIM_INPUT + j] = image[j];
            }
            this->test_labels[i] = cifar.test_labels[i];
        }
    }

    void normalize() {
        ::normalize(this->training_images, NUM_TRAINING);
        ::normalize(this->validation_images, NUM_VALIDATION);
        ::normalize(this->test_images, NUM_TEST);
    }

    void debug_print() {
        for (int i = 0; i < DIM_INPUT; i++) {
            cout << this->training_images[i] << " ";
        }
        cout << endl;
    }
};

int main() {
    auto cifar = cifar::read_dataset<vector, vector, uint8_t, uint8_t>(NUM_TRAINING + NUM_VALIDATION, NUM_TEST);
    Data data;
    data = cifar;
    
    // normalize data
    data.normalize();

    // create w and b and initialize
    double sigma = 0.01, mu = 0;
    double *w, *b;
    cudaMallocManaged(&w, DIM_INPUT * DIM_OUTPUT * sizeof(double));
    cudaMallocManaged(&b, DIM_OUTPUT * sizeof(double));
    initialize(w, b, mu, sigma);

    batchGradientDescent(data.training_images, data.training_labels, w, b, data.validation_images, data.validation_labels);

    // accuracy on test set
    double accuracy = computeAccuracy(data.test_images, data.test_labels, w, b, NUM_TEST);
    cout << "Accuracy on test set: " << accuracy << endl;

    cudaFree(w);
    cudaFree(b);
    return 0;
}