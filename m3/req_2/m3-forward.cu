#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

#define S 1 

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      std::cout<<"CUDA error: "<<cudaGetErrorString(err)<<std::endl;   \
      exit(-1);                                                          \
    }                                                                     \
  } while (0)

__global__ void matrix_unrolling_kernel(const float *input, const float *mask, float *output,int Batch, int Map_out, int Channel ,int Height, int Width, int K) {

            __shared__ float input_s[TILE_WIDTH][TILE_WIDTH];
            __shared__ float mask_s[TILE_WIDTH][TILE_WIDTH];
            const int H_out = Height - K + 1;
            const int W_out = Width - K + 1;
            int b = blockIdx.z;
            int out_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
            int out_row = blockIdx.y * TILE_WIDTH + threadIdx.y;

            #define in_4d(b, c, h, w) input[(b) * (Channel  * Height * Width) + \
                    (c) * (Height * Width) + \
                    (h) * Width + (w)]

            #define out_4d(b, f, h, w) output[(b) * (Map_out * H_out * W_out) + \
                    (f) * (H_out * W_out) + \
                    (h) * W_out + (w)]

        
            float tot = 0;
                    for (int tileIdx = 0; tileIdx < ceil((float)(Channel  * K * K) / TILE_WIDTH); ++tileIdx) {
                    int maskColIdx = tileIdx * TILE_WIDTH + threadIdx.x;
                    int maskRowIdx = tileIdx * TILE_WIDTH + threadIdx.y;

                    if (out_row < Map_out && maskColIdx < Channel  * K * K) {
                    mask_s[threadIdx.y][threadIdx.x] = mask[out_row * Channel  * K * K + maskColIdx];
                    } else {
                    mask_s[threadIdx.y][threadIdx.x] = 0;
                    }

                            int inputChannel = maskRowIdx / (K * K);
                            int row = (maskRowIdx % (K * K)) / K;
                            int col = (maskRowIdx % K);

                    if (out_col < W_out * H_out && maskRowIdx < Channel  * K * K) {
                        input_s[threadIdx.y][threadIdx.x] = in_4d(b, inputChannel, out_col / W_out + row, out_col % W_out + col);
                            } else {
                        input_s[threadIdx.y][threadIdx.x] = 0;
                    }

                            __syncthreads();

                            for (int i = 0; i < TILE_WIDTH; ++i) {
                            tot += mask_s[threadIdx.y][i] * input_s[i][threadIdx.x];
                            }

                    __syncthreads();
                    }

                    if (out_row < Map_out && out_col < W_out * H_out) {
                    out_4d(b, out_row, out_col / W_out, out_col % W_out) = tot;
                    }

            #undef in_4d
            #undef out_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
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

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{


// Free device memory
int Height_out = Height - K + 1;
int Width_out = Width - K + 1;
cudaMalloc(device_output_ptr, (size_t) Batch * Map_out * Height_out * Width_out * sizeof(float));
cudaMalloc(device_input_ptr, (size_t) Batch * Channel * Height * Width * sizeof(float));
cudaMalloc(device_mask_ptr, (size_t) Map_out * Channel * K * K * sizeof(float));

cudaMemcpy(*device_input_ptr, host_input, (size_t) Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(*device_mask_ptr, host_mask, (size_t) Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
}



    



__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
 
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;    

    // batch x feature maps x image size
    dim3 grid_dim(ceil((Height_out * Width_out) / (1.0 * TILE_WIDTH)), ceil(Map_out / (TILE_WIDTH * 1.0)), Batch);
    dim3 block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    matrix_unrolling_kernel<<<grid_dim, block_dim>>>(device_input, device_mask, device_output, Batch, Map_out, Channel, Height, Width, K);
}  



__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // int Height_out = Height - K + 1;
    // int Width_out = Width - K + 1;

    // const int size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    // // TODO: Copy the output back to host
    // cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_output, device_output, (size_t) Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // // TODO: Free device memory
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


