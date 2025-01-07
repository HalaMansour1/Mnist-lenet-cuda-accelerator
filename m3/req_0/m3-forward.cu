#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256


static float *pinned_input = nullptr;
static float *pinned_mask = nullptr;
static float *pinned_output = nullptr;

static cudaStream_t streams[2]; 

__global__ void matrix_unrolling_kernel(const float *input, float *output, const int Batch, const int Channel, const int Height, const int Width, const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const size_t Height_out = Height - K + 1;
    const size_t Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    size_t b = blockIdx.y;  
    size_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    size_t W_unroll = Height_out * Width_out;

//     int th = (blockIdx.y / num_of_block_w) * TILE_WIDTH + threadIdx.y;
//    int tw = (blockIdx.y % num_of_block_w) * TILE_WIDTH + threadIdx.x;

    if (idx < Channel * W_unroll) {
        size_t c = idx / W_unroll;
        size_t s = idx % W_unroll;
        size_t h_out = s / Width_out;
        size_t w_out = s % Width_out;

        size_t w_unroll = h_out * Width_out + w_out;
        size_t w_base = c * K * K;

        for (size_t p = 0; p < K; p++) {   // mask loop
            for (size_t q = 0; q < K; q++) { // mask loop 
                size_t h_unroll = w_base + p * K + q;
                output[h_unroll * W_unroll * Batch + W_unroll * b + w_unroll] = in_4d(b, c, p + h_out, w_out + q);
            }
        }
    }

    #undef in_4d
}


__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out, int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}
__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_output, const float *host_input, const float *host_mask,
    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;

    size_t output_size = Batch * Map_out * H_out * W_out * sizeof(float);
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);

    // pinned host memory
    cudaMallocHost((void **)&pinned_input, input_size);
    cudaMallocHost((void **)&pinned_mask, mask_size);
    cudaMallocHost((void **)&pinned_output, output_size);


    memcpy(pinned_input, host_input, input_size);
    memcpy(pinned_mask, host_mask, mask_size);
  

    // Allocate device memory
    cudaMalloc((void**)device_output_ptr, output_size);
    cudaMalloc((void**)device_input_ptr, input_size);
    cudaMalloc((void**)device_mask_ptr, mask_size);


    cudaMemcpy(*device_mask_ptr, pinned_mask, mask_size, cudaMemcpyHostToDevice);

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
}

__host__ void GPUInterface::conv_forward_gpu(
    float *device_output, const float *device_input, const float *device_mask,
    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;
    int Height_unrolled = Channel * K * K;
    int Width_unrolled = Batch * H_out * W_out;


    float *unrolled_matrix;
    float *matmul_output;
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * H_out * W_out * sizeof(float));
    cudaMalloc((void**)&matmul_output, (size_t) Batch * Map_out * H_out * W_out * sizeof(float));

    const int streamN = 8; 
    int image_size = H_out * W_out;
   
    dim3 stream_block(BLOCK_SIZE, 1, 1);

    int cur_size = (Batch + streamN - 1) / streamN;

    for (int i = 0; i < cur_size; i++) {
        int batch_offset = i * streamN;
        int stream_id = i % 2;
        int curr_batch= std::min(streamN, Batch - batch_offset);

        size_t stream_size = curr_batch * Channel * Height * Width * sizeof(float);

        // Async copy input chunk
        cudaMemcpyAsync((float*)device_input + batch_offset * Channel * Height * Width, pinned_input + batch_offset * Channel * Height * Width, stream_size, cudaMemcpyHostToDevice, streams[stream_id]);
        dim3 stream_grid(((Channel * H_out * W_out) + BLOCK_SIZE - 1)/BLOCK_SIZE, curr_batch, 1);
        matrix_unrolling_kernel<<<stream_grid, stream_block, 0, streams[stream_id]>>>((device_input + (batch_offset * Channel * Height * Width)), (unrolled_matrix + batch_offset * image_size), Batch, Channel, Height, Width, K);
    }


    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);



    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim2((Width_unrolled + TILE_WIDTH - 1)/TILE_WIDTH, (Height_unrolled + TILE_WIDTH - 1)/TILE_WIDTH, 1);
    matrixMultiplyShared<<<gridDim2, blockDim2>>>(device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);

    cudaDeviceSynchronize();

  
    dim3 permute_kernel_grid_dim((image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(matmul_output, device_output, Map_out, Batch, image_size);
    cudaDeviceSynchronize();

    size_t output_size = Batch * Map_out * H_out * W_out * sizeof(float);
    cudaMemcpy(pinned_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // for (int streamIdx = 0; streamIdx < streamN; ++streamIdx) {
    //     cudaStreamDestroy(streams[streamIdx]);
    // }

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(
    float *host_output, float *device_output, float *device_input, float *device_mask,
    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int H_out = Height - K + 1;
    int W_out = Width - K + 1;
    size_t output_size = Batch * Map_out * H_out * W_out * sizeof(float);

    memcpy(host_output, pinned_output, output_size);


    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_mask);
    cudaFreeHost(pinned_output);
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
