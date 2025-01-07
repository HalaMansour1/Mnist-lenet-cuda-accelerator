#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>
using namespace nvcuda;

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define S 1

// #define wbCheck(stmt) 

__global__ void matrix_unrolling_kernel(const float * __restrict__ input, const float * __restrict__ mask, 
                                        float * __restrict__ output, int Batch, int Map_out, int Channel,
                                        int Height, int Width, int K) 
{
    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;
    const int CKK = Channel * K * K;
    const int HW_out = H_out * W_out;

    #define in_4D(b, c, h, w) input[(b) * (Channel * Height * Width) + \
                                      (c) * (Height * Width) + (h) * Width + (w)]

    #define out_4D(b, f, h, w) output[(b) * (Map_out * H_out * W_out) + \
                                        (f) * (H_out * W_out) + (h) * W_out + (w)]

    __shared__ half inputTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ half maskTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float out[TILE_WIDTH][TILE_WIDTH];

    // made all constants 
        const int b = blockIdx.z;
        const int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
        const int out_row = blockIdx.y * TILE_WIDTH + threadIdx.y;

    //faaster pre-computation
    const int h = (out_col < HW_out) ? (out_col / W_out) : 0;
    const int w = (out_col < HW_out) ? (out_col % W_out) : 0;

    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_f;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_f;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> outf;
    wmma::fill_fragment(outf, 0.0f);


    const int numTiles = (CKK + TILE_WIDTH - 1) / TILE_WIDTH;

    #pragma unroll
    for (int tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        int maskColIdx = tileIdx * TILE_WIDTH + threadIdx.x; 
        int maskRowIdx = tileIdx * TILE_WIDTH + threadIdx.y;

        
        half maskVal = __float2half(0.0f);
        if (out_row < Map_out && maskColIdx < CKK) {
            maskVal = __float2half(mask[out_row * CKK + maskColIdx]);
        }
        maskTile[threadIdx.y][threadIdx.x] = maskVal;

        // faster if statements 
        int inputChannel = (maskRowIdx < CKK) ? (maskRowIdx / (K * K)) : 0;
        int posInK = (maskRowIdx < CKK) ? (maskRowIdx % (K * K)) : 0;
        int Row = posInK / K;
        int Col = posInK % K;

        // added shared inputTile
        half inputVal = __float2half(0.0f);
        if (out_col < HW_out && maskRowIdx < CKK) {
            int globalH = h + Row;
            int globalW = w + Col;
            inputVal = __float2half(in_4D(b, inputChannel, globalH, globalW));
        }
        inputTile[threadIdx.y][threadIdx.x] = inputVal;

        __syncthreads();

        if (threadIdx.y < 2) { 
            wmma::load_matrix_sync(a_f, (half*)maskTile, TILE_WIDTH);
            wmma::load_matrix_sync(b_f, (half*)inputTile, TILE_WIDTH);
            wmma::mma_sync(outf, a_f, b_f, outf);
        }
        __syncthreads(); // removable ?
    }

    if (threadIdx.y < 2) {
        wmma::store_matrix_sync((float*)out, outf, TILE_WIDTH, wmma::mem_row_major);
    }

    __syncthreads();

    if (out_row < Map_out && out_col < HW_out) {
        out_4D(b, out_row, h, w) = out[threadIdx.y][threadIdx.x];
    }

    #undef in_4D
    #undef out_4D
}

__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                     int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

// Host functions remain largely the same. Ensure that constants and parameters are 
// precomputed before launching kernels to minimize overhead.

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask,
                                                    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    cudaMalloc(device_output_ptr, (size_t) Batch * Map_out * Height_out * Width_out * sizeof(float));
    cudaMalloc(device_input_ptr, (size_t) Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc(device_mask_ptr, (size_t) Map_out * Channel * K * K * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, (size_t) Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, (size_t) Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
                                             const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;   

    dim3 grid_dim((int)ceil((Height_out * Width_out) / (1.0 * TILE_WIDTH)),
                  (int)ceil(Map_out / (TILE_WIDTH * 1.0)), Batch);
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);

    matrix_unrolling_kernel<<<grid_dim, block_dim>>>(device_input, device_mask, device_output, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    cudaMemcpy(host_output, device_output, (size_t) Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
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

