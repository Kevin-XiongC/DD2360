#include <iostream>
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
#define NUM_TEST (1000)

#define BATCH_SIZE (100)
#define LEARNING_RATE (0.001)
#define LAMBDA (0.3)
#define EPOCHS (40)

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

double *dot(double *a, double *b, const int n, const int m, const int p) {
    // @param a: of size (n, m)
    // @param b: of size (m, p)
    // @return: dot(a, b), of size (n, p)
    double *ab = (double *)calloc(n * p, sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < m; k++)
                ab[i * p + j] += a[i * m + k] * b[k * p + j];
        }
    }
    return ab;
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

double *evaluate(double *x, double *w, double *b, const int N) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    // @return: softmax(dot(x, w) + b), of size (N, 10)
    double *wx = dot(x, w, N, DIM_INPUT, DIM_OUTPUT);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DIM_OUTPUT; j++)
            wx[i * DIM_OUTPUT + j] += b[j];
    }
    softmax(wx, N);
    return wx;
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

void computeGradient(double *x, int *y, double *w, double *b, const int N, Gradient *gradient) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param y: of size (N, 1), that is (NUM, 1)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    // result: gradient for w and b
    double *p = evaluate(x, w, b, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DIM_OUTPUT; j++) {
            if (j == y[i]) p[i * DIM_OUTPUT + j] -= 1;
        }
    }

    double *xT = transpose(x, N, DIM_INPUT);    // of size (3072, N)
    gradient->grad_W = dot(xT, p, DIM_INPUT, N, DIM_OUTPUT);
    for (int i = 0; i < DIM_INPUT; i++) {
        for (int j = 0; j < DIM_OUTPUT; j++) {
            int index = i * DIM_OUTPUT + j;
            gradient->grad_W[index] = gradient->grad_W[index] / N + 2 * LAMBDA * w[index];
        }
    }
    free(xT);

    gradient->grad_b = (double *)calloc(DIM_OUTPUT, sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < DIM_OUTPUT; j++)
            gradient->grad_b[j] += p[i * DIM_OUTPUT + j];
    }
    for (int i = 0; i < DIM_OUTPUT; i++) gradient->grad_b[i] /= N;
    free(p);
}

void batchGradientDescent(double *x, int *y, double *w, double *b, double *x_t, int *y_t) {
    // @param x: of size (N, 3072), that is (NUM, DIM_INPUT)
    // @param y: of size (N, 1), that is (NUM, 1)
    // @param w: of size (3072, 10), that is (DIM_INPUT, DIM_OUTPUT)
    // @param b: of size (10, 1), that is (DIM_OUTPUT, 1)
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // compute cost
        double cost = computeCost(x, y, w, b, NUM_TRAINING);
        // compute accuracy
        double acc = computeAccuracy(x, y, w, b, NUM_TRAINING);
        printf("Epoch %d: cost = %lf accuracy = %lf \n", epoch, cost, acc);

        for (int n_batch = 0; n_batch < NUM_TRAINING / BATCH_SIZE; n_batch++) {
            int start = n_batch * BATCH_SIZE;
            double *batch_x = x + start * DIM_INPUT;
            int *batch_y = y + start;
            Gradient *gradient = (Gradient *)calloc(1, sizeof(Gradient));
            computeGradient(batch_x, batch_y, w, b, BATCH_SIZE, gradient);
            // update gradient
            for (int i = 0; i < DIM_INPUT; i++) {
                for (int j = 0; j < DIM_OUTPUT; j++) {
                    int index = i * DIM_OUTPUT + j;
                    w[index] -= LEARNING_RATE * gradient->grad_W[index];
                }
            }
            for (int i = 0; i < DIM_OUTPUT; i++) {
                b[i] -= LEARNING_RATE * gradient->grad_b[i];
            }
            free(gradient);
        }
    }
}


class Data {
public:
    double *training_images;    // N * 3072
    int *training_labels;    // N
    double *test_images;
    int *test_labels;

    Data() {
        this->training_images = (double *)calloc(NUM_TRAINING * DIM_INPUT, sizeof(double));
        this->training_labels = (int *)calloc(NUM_TRAINING, sizeof(int));
        this->test_images = (double *)calloc(NUM_TEST * DIM_INPUT, sizeof(double));
        this->test_labels = (int *)calloc(NUM_TEST, sizeof(int));
    }

    ~Data() {
        free(this->training_images);
        free(this->training_labels);
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
    auto cifar = cifar::read_dataset<vector, vector, uint8_t, uint8_t>(NUM_TRAINING, NUM_TEST);
    Data data;
    data = cifar;
    
    // normalize data
    data.normalize();

    // create w and b and initialize
    double sigma = 0.01, mu = 0;
    double *w = (double *)calloc(DIM_INPUT * DIM_OUTPUT, sizeof(double));
    double *b = (double *)calloc(DIM_OUTPUT, sizeof(double));
    initialize(w, b, mu, sigma);

    batchGradientDescent(data.training_images, data.training_labels, w, b, data.test_images, data.test_labels);
}