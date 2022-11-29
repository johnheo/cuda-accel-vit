/*
 *  https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu
 *
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16
#define TOKENS 196

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void gpu_softmax(int *attn, int dim)
{

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ int expsum[TOKENS]; // sum for each row 

    for (int s=1; s<blockDim.x; s*=2) {
        if (tid % (2*s) == 0)
            expsum[tid] += attn[tid + s];

        __syncthreads();
    }
    if (tid == 0) expsum[blockIdx.x] = attn[0];

    // each thread responsible for one row? 
    for (int i=0; i<dim; i++) {
        attn[i] /= expsum[tid];
    }
}


__global__ void gpu_shared_matrix_mult(int *d_a, int *d_b, int *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}


void cpu_matmul(int *h_a, int *h_b, int *h_result, int m, int n, int k)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

void cpu_softmax(int *attention_map, int dim) {
    for (int i=0; i<dim; i++) {
        // exponentiate logits and find sum 
        int sum=0.0;
        for (int j=0; j<dim; j++) {
            // TODO: numerical error when casting back to int
            attention_map[i*dim+j] = (int)std::exp((double)attention_map[i*dim+j]);
            sum += attention_map[i*dim+j];
        }
        // normalize by sum to turn into probabilities
        printf("attn[0] = %d | sum = %d",attention_map[0], sum);
        for (int j=0; j<dim; j++) {
            attention_map[i*dim+j] /= sum;
        }
    }
}

/*
*********************************************************************
function name: main
description: test and compare
parameters: 
            none
return: none
*********************************************************************
*/
int main(int argc, char const *argv[])
{
    int T, D, H;
    /* Fixed seed for illustration */
    srand(3333);

    // receive input parameters
    sscanf(argv[1], "%d", &T);
    sscanf(argv[2], "%d", &D);
    sscanf(argv[3], "%d", &H);

    // printf("command line input : \n (T,D) = (%d,%d) || # Heads (H) = %d\n", T, D, H);

    // T = 196;
    // D = 384;
    // H = 100;
    printf("\n=================================\n");
    printf("command line input : \n (T,D) = (%d,%d) || # Heads (H) = %d\n", T, D, H);

    // allocate memory in host RAM, cpu_result is used to store CPU result
    int *mat_Q, *mat_K, *mat_V, *gpu_result, *cpu_attn, *cpu_result;
    cudaMallocHost((void **) &mat_Q, sizeof(int)*T*D/H);
    cudaMallocHost((void **) &mat_K, sizeof(int)*T*D/H);
    cudaMallocHost((void **) &mat_V, sizeof(int)*T*D/H);
    cudaMallocHost((void **) &gpu_result, sizeof(int)*T*D/H);

    cudaMallocHost((void **) &cpu_attn, sizeof(int)*T*T);
    cudaMallocHost((void **) &cpu_result, sizeof(int)*T*D/H);

    // random initialize matrix A
    for (int i = 0; i < T; ++i) {
        for (int j = 0; j < D/H; ++j) {
            mat_Q[i * D/H + j] = rand() % 1024;
            mat_V[i * D/H + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < D/H; ++i) {
        for (int j = 0; j < T; ++j) {
            mat_K[i * H + j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    bool gpu,cpu;
    gpu = true;
    cpu = true;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (gpu) {
        // start to count execution time of GPU version
        cudaEventRecord(start, 0);
        // Allocate memory space on the device 
        int *gpu_Q, *gpu_K, *gpu_V, *gpu_attn, *cuda_result;
        cudaMalloc((void **) &gpu_Q, sizeof(int)*T*D/H);
        cudaMalloc((void **) &gpu_K, sizeof(int)*T*D/H);
        cudaMalloc((void **) &gpu_V, sizeof(int)*T*D/H);
        cudaMalloc((void **) &gpu_attn, sizeof(int)*T*T);
        cudaMalloc((void **) &cuda_result, sizeof(int)*T*D/H);

        // copy matrix A and B from host to device memory
        cudaMemcpy(gpu_Q, mat_Q, sizeof(int)*T*D/H, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_K, mat_K, sizeof(int)*D/H*T, cudaMemcpyHostToDevice);
        cudaMemcpy(gpu_V, mat_V, sizeof(int)*T*D/H, cudaMemcpyHostToDevice);

        unsigned int grid_rows = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int grid_cols = (D/H + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(gpu_Q, gpu_K, gpu_attn, T, D/H, T);
        // gpu_softmax<<<dimGrid, dimBlock>>>(gpu_attn, T);
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(gpu_attn, gpu_V, cuda_result, T, T, D/H);

        // Transefr results from device to host 
        cudaMemcpy(gpu_result, cuda_result, sizeof(int)*T*D/H, cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        // time counting terminate
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // compute time elapse on GPU computing
        cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU: %f ms.\n\n", T, D, D, H, gpu_elapsed_time_ms);

        cudaFree(gpu_Q);
        cudaFree(gpu_K);
        cudaFree(gpu_V);
    }

    // ==============================start the CPU version===============================================
    if (cpu) {
        float total_cpu_time_ms = 0;
        cudaEventRecord(start, 0);

        cpu_matmul(mat_Q, mat_K, cpu_attn, T, D/H, T);
        // cpu_softmax(cpu_attn, T);
        cpu_matmul(cpu_attn, mat_V, cpu_result, T, T, D/H);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on CPU: %f ms.\n\n", T, D/H, D/H, H, cpu_elapsed_time_ms);
    }

    // optionally validate results computed by GPU
    // cannot do apples-to-apples valid b/c GPU is computing D/H while cpu is computing D
    if (false) {
        int all_ok = 1;
        for (int i = 0; i < T; ++i)
        {
            for (int j = 0; j < H; ++j)
            {
                #if defined(DEBUG)
                printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, cpu_result[i*H + j], i, j, mat_V[i*H + j]);
                if (j == H-1) printf("\n");
                #endif
                if(cpu_result[i*H + j] != gpu_result[i*H + j])
                {
                    all_ok = 0;
                }
            }
        }

        // roughly compute speedup
        if(all_ok)
        {
            printf("all results are correct!!!, speedup = %fx\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
        }
        else
        {
            printf("incorrect results\n");
        }
    }

    if (gpu && cpu) printf("speedup = %fx\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);

    // free memory
    if (gpu) {
        cudaFreeHost(mat_Q);
        cudaFreeHost(mat_K);
        cudaFreeHost(mat_V);
        cudaFreeHost(cpu_result);
        // ...
    }

    return 0;
}
