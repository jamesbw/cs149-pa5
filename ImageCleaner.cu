#include "ImageCleaner.h"

#ifndef SIZEX
#error Please define SIZEX.
#endif
#ifndef SIZEY
#error Please define SIZEY.
#endif

#define SIZE SIZEX
#define PI     3.14159256f
#define TWO_PI 6.28318530f

//----------------------------------------------------------------
// TODO:  CREATE NEW KERNELS HERE.  YOU CAN PLACE YOUR CALLS TO
//        THEM IN THE INDICATED SECTION INSIDE THE 'filterImage'
//        FUNCTION.
//
// BEGIN ADD KERNEL DEFINITIONS
//----------------------------------------------------------------



__device__ float roots_real[SIZE / 2];
__device__ float roots_imag[SIZE / 2];
// __device__ int bit_reverse[SIZE];


__global__ void populateRoots()
{
  int index = threadIdx.x;
  float angle = -2 * PI * index / SIZE;
  roots_real[index] = cos(angle);
  roots_imag[index] = sin(angle);


}

__device__ char fft(int pos, float (*real)[SIZE], float (*imag)[SIZE])
{
  //copy into second array

  __shared__ float roots_real_local[SIZE/2];
  __shared__ float roots_imag_local[SIZE/2];

  if (pos < SIZE / 2)
  {
    float angle = - TWO_PI * pos / SIZE;
    roots_real_local[pos] = __cosf(angle);
    roots_imag_local[pos] = __sinf(angle);
  }


  __syncthreads();

  char curr = 0;
  char next = 1;

  int span = SIZE >> 1;
  int temp;

  for (int unit_size = 1; unit_size < SIZE ; unit_size <<= 1)
  {
    int pos_in_unit = pos % unit_size;
    temp = pos_in_unit * (SIZE >> 1) / unit_size; // twiddle index
    float twiddle_real = roots_real_local[temp];
    float twiddle_imag = roots_imag_local[temp];

    if (pos < span)
    {
      //x1 = x1 + twiddle * x2
      temp = pos + span;
      float r1 = real[curr][pos];
      float r2 = real[curr][temp];
      float i1 = imag[curr][pos];
      float i2 = imag[curr][temp];
      temp = (pos << 1) - pos_in_unit;
      real[next][temp] = r1 + (twiddle_real * r2 - twiddle_imag * i2);
      imag[next][temp] = i1 + (twiddle_real * i2 + twiddle_imag * r2);
    }
    else
    {
      // x2 = x1 - twiddle *x2
      temp = pos - span;
      float r1 = real[curr][temp];
      float r2 = real[curr][pos];
      float i1 = imag[curr][temp];
      float i2 = imag[curr][pos];
      temp = ((pos - span) << 1) - pos_in_unit + unit_size;
      real[next][temp] = r1 - (twiddle_real * r2 - twiddle_imag * i2);
      imag[next][temp] = i1 - (twiddle_real * i2 + twiddle_imag * r2);
    }
    __syncthreads();
    next = curr;
    curr = 1 - curr;
  }
  return curr;
}

__global__ void forwardFFTRow(float *real_image, float *imag_image, int size)
{
  int row = blockIdx.x;
  int col = threadIdx.x;

  __shared__ float real[2][SIZE];
  __shared__ float imag[2][SIZE];


  int offset = row * SIZE + col;

  real[0][col] = real_image[offset];
  imag[0][col] = imag_image[offset];


  char curr = fft(col, real, imag);

  real_image[offset] = real[curr][col];
  imag_image[offset] = imag[curr][col];
}

__global__ void forwardDFTRow(float *real_image, float *imag_image, int size)
{
  int row = blockIdx.x;
  int col = threadIdx.x;

  __shared__ float real[SIZE];
  __shared__ float imag[SIZE];
  __shared__ float roots_real_local[SIZE];
  __shared__ float roots_imag_local[SIZE];

  int offset = row * SIZE + col;

  real[col] = real_image[offset];
  imag[col] = imag_image[offset];
  // roots_real_local[col] = roots_real[col];
  // roots_imag_local[col] = roots_imag[col];
  float angle = - 2 * PI * col / SIZE;
  roots_real_local[col] = __cosf(angle);
  roots_imag_local[col] = __sinf(angle);

  __syncthreads();

  // if(row == 0 && col == 0)
  // {
  //   printf("\n1st real row: \n");
  //   for (int i = 0; i < SIZE; ++i)
  //   {
  //     printf("%f, ", real[i]);
  //   }
  //   printf("\n1st imag row: \n");
  //   for (int i = 0; i < SIZE; ++i)
  //   {
  //     printf("%f, ", imag[i]);
  //   }
  // }

  float real_val = 0.f;
  float imag_val = 0.f;

  for (int n = 0; n < SIZE; ++n)
  {
    int index = n * col % SIZE;
    float root_real = roots_real_local[index];
    float root_imag = roots_imag_local[index];

    real_val += real[n]* root_real - imag[n]* root_imag;
    imag_val += imag[n]* root_real + real[n]* root_imag;
  }

  real_image[offset] = real_val;
  imag_image[offset] = imag_val;

  // if(row == 0 && col == 0)
  // {
  //   printf("\n1st transform real: \n");
  //   printf("%f, ", real_val);
  //   printf("\n1st transform imag: \n");
  //   printf("%f, ", imag_val);
  // }

}

__global__ void forwardDFTCol(float *real_image, float *imag_image, int size)
{
  int col = blockIdx.x;
  int row = threadIdx.x;
  if (col >= SIZE / 8)
  {
    col += 3 * SIZE / 4;
  }

  __shared__ float real[SIZE];
  __shared__ float imag[SIZE];
  __shared__ float roots_real_local[SIZE];
  __shared__ float roots_imag_local[SIZE];

  real[row] = real_image[row * SIZE + col];
  imag[row] = imag_image[row * SIZE + col];
  float angle = - 2 * PI * row / SIZE;
  roots_real_local[row] = __cosf(angle);
  roots_imag_local[row] = __sinf(angle);

  __syncthreads();

  float real_val = 0.f;
  float imag_val = 0.f;

  for (int n = 0; n < SIZE; ++n)
  {
    int index = n * row % SIZE;

    float root_real = roots_real_local[index];
    float root_imag = roots_imag_local[index];

    real_val += real[n]* root_real - imag[n]* root_imag;
    imag_val += imag[n]* root_real + real[n]* root_imag;
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
  if (row >= SIZE / 8)
  {
    row += 3 * SIZE / 4;
  }
  int col = threadIdx.x;

  __shared__ float real[SIZE];
  __shared__ float imag[SIZE];
  __shared__ float roots_real_local[SIZE];
  __shared__ float roots_imag_local[SIZE];

  real[col] = real_image[row * SIZE + col];
  imag[col] = imag_image[row * SIZE + col];
  float angle = - 2 * PI * col / SIZE;
  roots_real_local[col] = __cosf(angle);
  roots_imag_local[col] = __sinf(angle);

  __syncthreads();

  float real_val = 0.f;
  float imag_val = 0.f;

  for (int n = 0; n < SIZE; ++n)
  {
    int index = n * col % SIZE;

    float root_real = roots_real_local[index];
    float root_imag = -roots_imag_local[index];

    real_val += real[n]* root_real - imag[n]* root_imag;
    imag_val += imag[n]* root_real + real[n]* root_imag;
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
  __shared__ float roots_real_local[SIZE];
  __shared__ float roots_imag_local[SIZE];

  real[row] = real_image[row * SIZE + col];
  imag[row] = imag_image[row * SIZE + col];
  float angle = - 2 * PI * row / SIZE;
  roots_real_local[row] = __cosf(angle);
  roots_imag_local[row] = __sinf(angle);

  __syncthreads();

  float real_val = 0.f;
  float imag_val = 0.f;

  for (int n = 0; n < SIZE; ++n)
  {
    int index = n * row % SIZE;

    float root_real = roots_real_local[index];
    float root_imag = -roots_imag_local[index];

    real_val += real[n]* root_real - imag[n]* root_imag;
    imag_val += imag[n]* root_real + real[n]* root_imag;
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

  float fftr = 0.f, fftc = 0.f, ifftr = 0.f, ifftc = 0.f, filter_time = 0.f, roots = 0.f;
  cudaEvent_t start_bis, stop_bis;

  CUDA_ERROR_CHECK(cudaEventCreate(&start));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop));

  CUDA_ERROR_CHECK(cudaEventCreate(&start_bis));
  CUDA_ERROR_CHECK(cudaEventCreate(&stop_bis));

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

  // printf("\n1st row real\n");
  // for (int i = 0; i < size; ++i)
  // {
  //   printf("%f, ", real_image[i]);
  // }
  // printf("\n1st row imag\n");
  // for (int i = 0; i < size; ++i)
  // {
  //   printf("%f, ", imag_image[i]);
  // }
  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));

  populateRoots<<<1, SIZE / 2, 0, filterStream>>>();

  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&roots,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));


  forwardFFTRow<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag, size);

  // CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
  // CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

  // printf("\n1st row tranform real\n");
  // for (int i = 0; i < size; ++i)
  // {
  //   printf("%f, ", real_image[i]);
  // }
  // printf("\n1st row tranform imag\n");
  // for (int i = 0; i < size; ++i)
  // {
  //   printf("%f, ", imag_image[i]);
  // }

  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&fftr,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));

  forwardDFTCol<<<SIZE / 4, SIZE, 0, filterStream>>>(device_real, device_imag, size);
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&fftc,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));
  filter<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag, size);
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&filter_time,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));
  inverseDFTRow<<<SIZE / 4, SIZE, 0, filterStream>>>(device_real, device_imag, size);
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&ifftr,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));
  inverseDFTCol<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag, size);
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&ifftc,start_bis,stop_bis));


  //---------------------------------------------------------------- 
  // END ADD KERNEL CALLS
  //----------------------------------------------------------------

  // Finish timimg for the execution 
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&execution,start,stop_bis));

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

  printf("  Roots Time: %f ms\n\n", roots);
  printf("  Row DFT Time: %f ms\n\n", fftr);
  printf("  Col DFT Time: %f ms\n\n", fftc);
  printf("  Filter Time: %f ms\n\n", filter_time);
  printf("  Row IDFT Time: %f ms\n\n", ifftr);
  printf("  Col IDFT Time: %f ms\n\n", ifftc);
  // Return the total time to transfer and execute
  return totalTime;
}

