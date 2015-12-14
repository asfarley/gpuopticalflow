#include "atlimage.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "limits.h"
#include "time.h"
#include "windows.h"
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 640
#define HEIGHT 480
#define DEBUG false
#define RENDER_GAIN 30
#define ZERO_PRIOR 1000

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

 void renderOpticalFlow(int flow[WIDTH * HEIGHT * 2], CImage *image)
{
	for (int i = 0; i < WIDTH; i++)
		for (int j = 0; j < HEIGHT; j++)
		{
			byte r = (byte) abs(RENDER_GAIN * flow[(i * HEIGHT * 2) + (j * 2) + 0]);
			byte g = (byte) abs(RENDER_GAIN * flow[(i * HEIGHT * 2) + (j * 2) + 1]);
			byte b = 0;
			COLORREF color = RGB(r,g,b);
			(*image).SetPixel(i, j, color);
		}
}

 __device__ void threshold(int *value, int max)
 {
	 if (*value < 0)
		 *value = 0;

	 if (*value > max)
		 *value = max;
 }

 //Calculate match value for a fixed x,y location and shift value
__device__ void sumSquareErrorOffsetKernel(int frame1[WIDTH * HEIGHT], int frame2[WIDTH * HEIGHT], int x, int y, int dx, int dy, int *sse, int block_width)
 {
	 int x_start = x - block_width / 2;
	 int y_start = y - block_width / 2;
	 int x_end = x_start + block_width;
	 int y_end = y_start + block_width;

	 threshold(&x, WIDTH);
	 threshold(&y, HEIGHT);

	 int sumSquareError = 0;
	 for (int i = x_start; i < x_end; i++)
		 for (int j = y_start; j < y_end; j++)
			 sumSquareError += (frame1[i * HEIGHT + j] - frame2[(i + dx) * HEIGHT + j + dy])*(frame1[i * HEIGHT + j] - frame2[(i + dx) * HEIGHT + j + dy]);

	 *sse = sumSquareError;
 }

//Find maximum match value at a specified pixel for all possible shifts 
__global__ void opticalFlowPixelKernel(int frame1[WIDTH * HEIGHT], int frame2[WIDTH * HEIGHT], int flow[WIDTH * HEIGHT * 2], int max_shift, int block_width)
{
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int best_dx=0;
	int best_dy=0;
	int best_match = INT_MAX;
	if (x >= WIDTH || y >= HEIGHT) return;

	for (int dx = -max_shift; dx < max_shift; dx++)
		for (int dy = -max_shift; dy < max_shift; dy++)
		{
			int sse = INT_MAX;

			sumSquareErrorOffsetKernel(frame1, frame2, x, y, dx, dy, &sse, block_width);

			if (dx == 0 && dy == 0)
				sse -= ZERO_PRIOR;

			if (sse < best_match)
			{
				best_match = sse;
				best_dx = dx;
				best_dy = dy;
			}
		}

	flow[(x * HEIGHT * 2) + (y * 2) + 0] = best_dx;
	flow[(x * HEIGHT * 2) + (y * 2) + 1] = best_dy;
}

 cudaError_t opticalFlow(const int *frame1,const int *frame2, int *flow, int max_shift, int block_width)
 {
	 int *frame1_in = 0;
	 int *frame2_in = 0;
	 int *flow_out = 0;
	 cudaError_t cudaStatus;
	 int size = WIDTH * HEIGHT * sizeof(int);

	 dim3 dimBlock(16, 16);
	 dim3 dimGrid;
	 dimGrid.x = (WIDTH + dimBlock.x - 1) / dimBlock.x;  /*< Greater than or equal to image width */
	 dimGrid.y = (HEIGHT + dimBlock.y - 1) / dimBlock.y; /*< Greater than or equal to image height */

	 // Choose which GPU to run on, change this on a multi-GPU system.
	 cudaStatus = cudaSetDevice(0);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		 goto Error;
	 }

	 // Allocate GPU buffers for three arrays (two input, one output)    .
	 cudaStatus = cudaMalloc((void**)&frame1_in, size);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "cudaMalloc failed!");
		 goto Error;
	 }

	 cudaStatus = cudaMalloc((void**)&frame2_in, size);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "cudaMalloc failed!");
		 goto Error;
	 }

	 cudaStatus = cudaMalloc((void**)&flow_out, 2 * size);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "cudaMalloc failed!");
		 goto Error;
	 }

	 // Copy input vectors from host memory to GPU buffers.
	 cudaStatus = cudaMemcpy(frame1_in, frame1, size, cudaMemcpyHostToDevice);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "cudaMemcpy failed!");
		 goto Error;
	 }

	 cudaStatus = cudaMemcpy(frame2_in, frame2, size, cudaMemcpyHostToDevice);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "cudaMemcpy failed!");
		 goto Error;
	 }

	 // Launch a kernel on the GPU with one thread for each element.
	 opticalFlowPixelKernel<<< dimGrid, dimBlock >>>(frame1_in, frame2_in, flow_out, max_shift, block_width);

	 // Check for any errors launching the kernel
	 cudaStatus = cudaGetLastError();
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "opticalFlowPixelKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		 goto Error;
	 }

	 // cudaDeviceSynchronize waits for the kernel to finish, and returns
	 // any errors encountered during the launch.
	 cudaStatus = cudaDeviceSynchronize();
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching opticalFlowPixelKernel!\n", cudaStatus);
		 goto Error;
	 }

	 // Copy output vector from GPU buffer to host memory.
	 cudaStatus = cudaMemcpy(flow, flow_out, 2 * size, cudaMemcpyDeviceToHost);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "cudaMemcpy failed!");
		 goto Error;
	 }

 Error:
	 cudaFree(frame1_in);
	 cudaFree(frame2_in);
	 cudaFree(flow_out);

	 return cudaStatus;
 }

void CImageToArray(CImage image, int imarray[WIDTH * HEIGHT])
{
	for (int i = 0; i < WIDTH; i++)
		for (int j = 0; j < HEIGHT; j++)
		{
			COLORREF color = image.GetPixel(i, j);
			imarray[i * HEIGHT + j] = GetBValue(color) + GetRValue(color) + GetGValue(color);
		}
}


int frame1_array[WIDTH * HEIGHT];
int frame2_array[WIDTH * HEIGHT];
int flow_array[WIDTH * HEIGHT * 2];

int main(int argc, char *argv[])
{
	CImage frame1;
	CImage frame2;
	CImage flow = CImage();
	
	//Code to load/create image goes here
	frame1.Load(_T("C:\\images\\frame1.png"));
	frame2.Load(_T("C:\\images\\frame2.png"));

	flow.Create(WIDTH, HEIGHT, 24); //May want to check return value (0 -> success)

	CImageToArray(frame1, frame1_array);
	CImageToArray(frame2, frame2_array);

	if (argc != 3)
	{
		printf("Wrong argument count - expected 2 (MAX_SHIFT and BLOCK_WIDTH)");
		return 1; //Code 1: Wrong parameters
	}
	int max_shift = atoi(argv[1]);
	int block_width = atoi(argv[2]);

	cudaError_t cudaStatus = opticalFlow(frame1_array, frame2_array, flow_array, max_shift, block_width);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	renderOpticalFlow(flow_array, &flow);

	flow.Save(_T("C:\\images\\opticalflow.png"));

	// 
    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };*/

    // Add vectors in parallel.
   /* cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);*/

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
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
