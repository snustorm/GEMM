
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define M 4
#define K 4
#define N 4
#define BLOCK_SIZE 4


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void init_matrix(float* matrix, int rows, int cols);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void matmul_cpu(float* A, float* B, float* C, int m, int k, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < k; l++)
            {
				sum += A[i * k + l] * B[l * n + j];
            }
			C[i * n + j] = sum;
        }
    }
}

__global__ void matmul_gpu(float* A, float* B, float* C, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
	{
		float sum = 0.0f;
		for (int i = 0; i < k; i++)
		{
			sum += A[row * k + i] * B[i * n + col];
            printf("A: %d: B: %d\n", row * k + i, i * n + col);
		}
		C[row * n + col] = sum;
	}
}

int main()
{

    float* h_A, * h_B, *h_C_cpu, *h_C_gpu;
	float* d_A, * d_B, * d_C;

    int size_A = M * N * sizeof(float);
	int size_B = K * N * sizeof(float); 
	int size_C = M * N * sizeof(float);

	h_A = (float*)malloc(size_A);  
	h_B = (float*)malloc(size_B);
	h_C_cpu = (float*)malloc(size_C);
	h_C_gpu = (float*)malloc(size_C);

	srand(time(NULL));
	init_matrix(h_A, M, K);
	init_matrix(h_B, K, N);

    //for (int i = 0; i < M * N; i++)
    //{
    //    std::cout << h_A[i] << std::endl;
    //}

    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);

    for (int i = 0; i < M * N; i++)
    {
        std::cout << h_C_cpu[i] << std::endl;
    }

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "blockDim X: " << blockDim.x << std::endl;
    std::cout << "blockDim Y: " << blockDim.y << std::endl;

	std::cout << "gridDim X: " << gridDim.x << std::endl;   
	std::cout << "gridDim Y: " << gridDim.y << std::endl;

	// 启动核函数
	matmul_gpu<<<gridDim, blockDim>>> (d_A, d_B, d_C, M, K, N);
    
    // 检查核函数启动错误
    cudaError_t kernalErr = cudaGetLastError();
	if (kernalErr != cudaSuccess)
	{
		std::cout << "Kernel launch error: " << cudaGetErrorString(kernalErr) << std::endl;
        return 1;
	}

    // 同步等待核函数完成
	cudaError_t syncErr = cudaDeviceSynchronize();
	if (syncErr != cudaSuccess)
	{
		std::cout << "Kernel sync error: " << cudaGetErrorString(syncErr) << std::endl;
		return 1;
	}

    // 拷贝结果回主机
    cudaError_t copyErr = cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    if (copyErr != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(copyErr) << std::endl;
        return 1;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C_gpu[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

	free(h_A);
	free(h_B);
	free(h_C_cpu);
	free(h_C_gpu);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    // Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}



    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}

void init_matrix(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
