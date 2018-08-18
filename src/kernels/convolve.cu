#include <stdint.h>

static int DivUp(int a, int b) { return (a + b - 1) / b; }

__global__ void int8_to_float_kernel(int8_t *input,
                                     float *output,
                                     int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        output[tid] = (float)input[tid];
    }
}

__global__ void scale_float_tensor_kernel(float *input,
								   float scale,
                                   int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        input[tid] = scale * input[tid];
    }
}

void int8_to_float(int8_t *input, float *output, size_t N) {
    const int blockSize = 256;
    int blocks = DivUp(N, blockSize);
    int8_to_float_kernel<<<blocks, blockSize>>>(input, output, N);
}

void scale_float_tensor(float *input, float scale, size_t N) {
    const int blockSize = 256;
    int blocks = DivUp(N, blockSize);
    scale_float_tensor_kernel<<<blocks, blockSize>>>(input, scale, N);
}
