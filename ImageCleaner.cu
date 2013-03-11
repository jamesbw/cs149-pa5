#include "ImageCleaner.h"

#ifndef SIZEX
#error Please define SIZEX.
#endif
#ifndef SIZEY
#error Please define SIZEY.
#endif

#define SIZE SIZEX
#define PI  3.14159256f

//----------------------------------------------------------------
// TODO:  CREATE NEW KERNELS HERE.  YOU CAN PLACE YOUR CALLS TO
//        THEM IN THE INDICATED SECTION INSIDE THE 'filterImage'
//        FUNCTION.
//
// BEGIN ADD KERNEL DEFINITIONS
//----------------------------------------------------------------


__global__ void exampleKernel(float *real_image, float *imag_image, int size_x, int size_y)
{
  // Currently does nothing
}

__global__ void forwardDFTRow(float *real_image, float *imag_image, int size)
{
  int row = blockIdx.x;
  int col = threadIdx.x;

  __shared__ float real[SIZE];
  __shared__ float imag[SIZE];

  real[col] = real_image[row * SIZE + col];
  imag[col] = imag_image[row * SIZE + col];

  __syncthreads();

  if(row == 0 && col == 0)
  {
    printf("\n1st real row: \n");
    for (int i = 0; i < SIZE; ++i)
    {
      printf("%f, ", real[i]);
    }
    printf("\n1st imag row: \n");
    for (int i = 0; i < SIZE; ++i)
    {
      printf("%f, ", imag[i]);
    }
  }

  float real_val = 0.f;
  float imag_val = 0.f;

  for (int n = 0; n < SIZE; ++n)
  {
    float angle = -2 * PI * col * n / size;

    real_val += (real[n]* cos(angle)) - (imag[n]* sin(angle));
    imag_val += (imag[n]* cos(angle)) + (real[n]* sin(angle));
  }

  real_image[row * SIZE + col] = real_val;
  imag_image[row * SIZE + col] = imag_val;

  if(row == 0 && col == 0)
  {
    printf("\n1st transform real: \n");
    printf("%f, ", real_val);
    printf("\n1st transform imag: \n");
    printf("%f, ", imag_val);
  }

}

__global__ void forwardDFTCol(float *real_image, float *imag_image, int size)
{
  int col = blockIdx.x;
  int row = threadIdx.x;

  __shared__ float real[SIZE];
  __shared__ float imag[SIZE];

  real[row] = real_image[row * SIZE + col];
  imag[row] = imag_image[row * SIZE + col];

  __syncthreads();

  float real_val = 0.f;
  float imag_val = 0.f;

  for (int n = 0; n < SIZE; ++n)
  {
    float angle = -2 * PI * row * n / size;

    real_val += (real[n]* cos(angle)) - (imag[n]* sin(angle));
    imag_val += (imag[n]* cos(angle)) + (real[n]* sin(angle));
  }

  real_image[row * SIZE + col] = real_val;
  imag_image[row * SIZE + col] = imag_val;
}

__global__ void filter(float *real_image, float *imag_image, int size)
{
  int row = blockIdx.x;
  int col = threadIdx.x;

  int eighth = size / 8;
  int seven_eighth = size - eighth;

  if ((row >= eighth && row < seven_eighth) || (col >= eighth && col < seven_eighth))
  {
    real_image[row * SIZE + col] = 0.f;
    imag_image[row * SIZE + col] = 0.f;
  }
}

__global__ void inverseDFTRow(float *real_image, float *imag_image, int size)
{
  int row = blockIdx.x;
  int col = threadIdx.x;

  __shared__ float real[SIZE];
  __shared__ float imag[SIZE];

  real[col] = real_image[row * SIZE + col];
  imag[col] = imag_image[row * SIZE + col];

  __syncthreads();

  float real_val = 0.f;
  float imag_val = 0.f;

  for (int n = 0; n < SIZE; ++n)
  {
    float angle = 2 * PI * col * n / size;

    real_val += (real[n]* cos(angle)) - (imag[n]* sin(angle));
    imag_val += (imag[n]* cos(angle)) + (real[n]* sin(angle));
  }

  real_image[row * SIZE + col] = real_val / size;
  imag_image[row * SIZE + col] = imag_val / size;
}

__global__ void inverseDFTCol(float *real_image, float *imag_image, int size)
{
  int col = blockIdx.x;
  int row = threadIdx.x;

  __shared__ float real[SIZE];
  __shared__ float imag[SIZE];

  real[row] = real_image[row * SIZE + col];
  imag[row] = imag_image[row * SIZE + col];

  __syncthreads();

  float real_val = 0.f;
  float imag_val = 0.f;

  for (int n = 0; n < SIZE; ++n)
  {
    float angle = 2 * PI * row * n / size;

    real_val += (real[n]* cos(angle)) - (imag[n]* sin(angle));
    imag_val += (imag[n]* cos(angle)) + (real[n]* sin(angle));
  }

  real_image[row * SIZE + col] = real_val / size;
  imag_image[row * SIZE + col] = imag_val / size;
}

//----------------------------------------------------------------
// END ADD KERNEL DEFINTIONS
//----------------------------------------------------------------

__host__ float filterImage(float *real_image, float *imag_image, int size_x, int size_y)
{
  // check that the sizes match up
  assert(size_x == SIZEX);
  assert(size_y == SIZEY);

  int size = size_x;

  int matSize = size_x * size_y * sizeof(float);

  // These variables are for timing purposes
  float transferDown = 0, transferUp = 0, execution = 0;
  cudaEvent_t start,stop;

  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  // Create a stream and initialize it
  cudaStream_t filterStream;
  CUDA_ERROR_CHECK(cudaStreamCreate(&filterStream));

  // Alloc space on the device
  float *device_real, *device_imag;
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_real, matSize));
  CUDA_ERROR_CHECK(cudaMalloc((void**)&device_imag, matSize));

  // Start timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));
  
  // Here is where we copy matrices down to the device 
  CUDA_ERROR_CHECK(cudaMemcpy(device_real,real_image,matSize,cudaMemcpyHostToDevice));
  CUDA_ERROR_CHECK(cudaMemcpy(device_imag,imag_image,matSize,cudaMemcpyHostToDevice));
  
  // Stop timing for transfer down
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferDown,start,stop));

  // Start timing for the execution
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  //----------------------------------------------------------------
  // TODO: YOU SHOULD PLACE ALL YOUR KERNEL EXECUTIONS
  //        HERE BETWEEN THE CALLS FOR STARTING AND
  //        FINISHING TIMING FOR THE EXECUTION PHASE
  // BEGIN ADD KERNEL CALLS
  //----------------------------------------------------------------

  // This is an example kernel call, you should feel free to create
  // as many kernel calls as you feel are needed for your program
  // Each of the parameters are as follows:
  //    1. Number of thread blocks, can be either int or dim3 (see CUDA manual)
  //    2. Number of threads per thread block, can be either int or dim3 (see CUDA manual)
  //    3. Always should be '0' unless you read the CUDA manual and learn about dynamically allocating shared memory
  //    4. Stream to execute kernel on, should always be 'filterStream'
  //
  // Also note that you pass the pointers to the device memory to the kernel call
  exampleKernel<<<1,128,0,filterStream>>>(device_real,device_imag,size_x,size_y);

  printf("\n1st row real\n");
  for (int i = 0; i < size; ++i)
  {
    printf("%f, ", real_image[i]);
  }
  printf("\n1st row imag\n");
  for (int i = 0; i < size; ++i)
  {
    printf("%f, ", imag_image[i]);
  }

  forwardDFTRow<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag, size);

  CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

  printf("\n1st row tranform real\n");
  for (int i = 0; i < size; ++i)
  {
    printf("%f, ", real_image[i]);
  }
  printf("\n1st row tranform imag\n");
  for (int i = 0; i < size; ++i)
  {
    printf("%f, ", imag_image[i]);
  }

  forwardDFTCol<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag, size);
  filter<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag, size);
  inverseDFTRow<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag, size);
  inverseDFTCol<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag, size);

  //---------------------------------------------------------------- 
  // END ADD KERNEL CALLS
  //----------------------------------------------------------------

  // Finish timimg for the execution 
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&execution,start,stop));

  // Start timing for the transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(start,filterStream));

  // Here is where we copy matrices back from the device 
  CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
  CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

  // Finish timing for transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferUp,start,stop));

  // Synchronize the stream
  CUDA_ERROR_CHECK(cudaStreamSynchronize(filterStream));
  // Destroy the stream
  CUDA_ERROR_CHECK(cudaStreamDestroy(filterStream));
  // Destroy the events
  CUDA_ERROR_CHECK(cudaEventDestroy(start));
  CUDA_ERROR_CHECK(cudaEventDestroy(stop));

  // Free the memory
  CUDA_ERROR_CHECK(cudaFree(device_real));
  CUDA_ERROR_CHECK(cudaFree(device_imag));

  // Dump some usage statistics
  printf("CUDA IMPLEMENTATION STATISTICS:\n");
  printf("  Host to Device Transfer Time: %f ms\n", transferDown);
  printf("  Kernel(s) Execution Time: %f ms\n", execution);
  printf("  Device to Host Transfer Time: %f ms\n", transferUp);
  float totalTime = transferDown + execution + transferUp;
  printf("  Total CUDA Execution Time: %f ms\n\n", totalTime);
  // Return the total time to transfer and execute
  return totalTime;
}

