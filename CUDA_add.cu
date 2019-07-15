
// Modified CUDA Add Example
// This example takes 2 float arrays of size 1M and adds them together.
// Prints out Total Runtime.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#include <iostream>
#include <iomanip>
#include <chrono>



cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size);

__global__ void addKernel(float *c, const float *a, const float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];

	//printf("ThreadIdx.x : %d " , i );

	//printf("a[i] = %.2f", a[i]);
	//printf("b[i] = %.2f", b[i]);
	//printf("c[i] = %.2f \n", c[i]);
}

int main()
{


	// Instance
    const int arraySize =  1 << 20;  // 1 Million elements
	float *x = new float[arraySize]; // Input 1
	float *y = new float[arraySize]; // Input 2
	float *z = new float[arraySize]; // Output

	// Initialize
	for (int i = 0; i < arraySize; i++)
	{
		x[i] = 1.0f;
		y[i] = 2.0f;
		z[i] = 0.0f;
	}

    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

	//Timing Variables
	auto start = std::chrono::high_resolution_clock::now();
	std::ios_base::sync_with_stdio(false);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(z, x, y, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	// Calculating total time taken by the program. 
	auto end = std::chrono::high_resolution_clock::now();
	double time_taken = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
	time_taken *= 1e-9;
	std::cout << "Time taken by program: " << std::fixed << time_taken << std::setprecision(9) << " sec. \n" << std::endl;


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }


	//Print arrays
	printf("{x[0], x[1], x[2], x[3],x[4], ...} + {y[0], y[1], y[2], y[3], y[4], ...} = {%.2f,%.2f,%.2f,%.2f,%.2f, ...}\n",
		z[0], z[1], z[2], z[3], z[4]);

	printf("Complete.\n");


    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, 256>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
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
