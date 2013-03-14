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


// __device__ char forwardFFT_arbitrary_radix(int radix, float (*real)[SIZE], float (*imag)[SIZE])
// {
//   int span = SIZE / radix;

//   for (int unit_size = 1; unit_size < SIZE ; unit_size *= span)
//   {
//     int pos_in_unit = threadIdx.x % unit_size;
//   }

// }

__device__ char forwardFFT_radix4(float (*real)[SIZE], float (*imag)[SIZE])
{

  __shared__ float roots_real_local[SIZE];
  __shared__ float roots_imag_local[SIZE];


  float angle = - TWO_PI * threadIdx.x / SIZE;
  roots_real_local[threadIdx.x] = __cosf(angle);
  roots_imag_local[threadIdx.x] = __sinf(angle);
  

  __syncthreads();

  char curr = 0;
  char next = 1;

  // int span = SIZE >> 2;
  // int temp;

  for (int unit_size = 1; unit_size < SIZE ; unit_size <<= 2)
  {
    int pos_in_unit = threadIdx.x % unit_size;
    // temp = pos_in_unit * (SIZE >> 2) / unit_size; // twiddle index
    // float twiddle1k_real = roots_real_local[temp];
    // float twiddle1k_imag = roots_imag_local[temp];
    // float twiddle2k_real = roots_real_local[temp << 1];
    // float twiddle2k_imag = roots_imag_local[temp << 1];
    // float twiddle3k_real = roots_real_local[3*temp];
    // float twiddle3k_imag = roots_imag_local[3*temp];


    if (threadIdx.x < (SIZE >> 2))
    {
      //x1 = x1 + twiddle1k * x2 + twiddle2k * x3 + twiddle3k * x4

      //no need to store x1, already there
      __syncthreads();

      int ind1 = threadIdx.x;
      int ind2 = threadIdx.x + (SIZE >> 2);
      int ind3 = threadIdx.x + (SIZE >> 1);
      int ind4 = threadIdx.x + (SIZE >> 1) + (SIZE >> 2); 

      int new_pos = ((threadIdx.x - pos_in_unit) << 2) + pos_in_unit; //new index of x1
      real[next][new_pos] = real[curr][ind1] + real[curr][ind2] + real[curr][ind3] + real[curr][ind4];
      imag[next][new_pos] = imag[curr][ind1] + imag[curr][ind2] + imag[curr][ind3] + imag[curr][ind4];

      // temp = threadIdx.x + (SIZE >> 2); // index of x2
      // int ind3 = threadIdx.x + (SIZE >> 1);
      // int ind4 = ind3 + (SIZE >> 2); 
      // float r2 = real[curr][temp];
      // float r3 = real[curr][ind3];
      // float r4 = real[curr][ind4];
      // float i2 = imag[curr][temp];
      // float i3 = imag[curr][ind3];
      // float i4 = imag[curr][ind4];
      // temp = ((threadIdx.x - pos_in_unit) << 2) + pos_in_unit; //new index of x1
      // real[next][temp] = real[curr][threadIdx.x] + (twiddle1k_real * r2 - twiddle1k_imag * i2) + (twiddle2k_real * r3 - twiddle2k_imag * i3) + (twiddle3k_real * r4 - twiddle3k_imag * i4);
      // imag[next][temp] = imag[curr][threadIdx.x] + (twiddle1k_real * i2 + twiddle1k_imag * r2) + (twiddle2k_real * i3 + twiddle2k_imag * r3) + (twiddle3k_real * i4 + twiddle3k_imag * r4);
    }
    else if (threadIdx.x < SIZE >> 1)
    {
      //x2 = x1 - j*twiddle1k * x2 - twiddle2k * x3 + j twiddle3k * x4
      int twiddle_index = pos_in_unit * (SIZE >> 2) / unit_size;
      float twiddle1k_real = roots_real_local[twiddle_index];
      float twiddle1k_imag = roots_imag_local[twiddle_index];

      //store twiddle1k * x2
      float r = real[curr][threadIdx.x], i = imag[curr][threadIdx.x];
      real[curr][threadIdx.x] = twiddle1k_real * r - twiddle1k_imag * i;
      imag[curr][threadIdx.x] = twiddle1k_imag * r + twiddle1k_real * i;
      __syncthreads();

      int ind1 = threadIdx.x - (SIZE >> 2);
      int ind2 = threadIdx.x;
      int ind3 = threadIdx.x + (SIZE >> 2);
      int ind4 = threadIdx.x + (SIZE >> 1); 

      int new_pos = ((threadIdx.x - pos_in_unit - (SIZE >> 2)) << 2) + (unit_size + pos_in_unit) ; //new index of x2
      real[next][new_pos] = real[curr][ind1] + imag[curr][ind2] - real[curr][ind3] - imag[curr][ind4];
      imag[next][new_pos] = imag[curr][ind1] - real[curr][ind2] - imag[curr][ind3] + real[curr][ind4];


      // temp = threadIdx.x - (SIZE >> 2); // index of x1
      // int ind3 = threadIdx.x + (SIZE >> 2);
      // int ind4 = threadIdx.x + (SIZE >> 1); 
      // float r1 = real[curr][temp];
      // // float r2 = real[curr][threadIdx.x];
      // float r3 = real[curr][ind3];
      // float r4 = real[curr][ind4];
      // float i1 = imag[curr][temp];
      // // float i2 = imag[curr][threadIdx.x];
      // float i3 = imag[curr][ind3];
      // float i4 = imag[curr][ind4];
      // temp = ((threadIdx.x - pos_in_unit - (SIZE >> 2)) << 2) + (unit_size + pos_in_unit) ; //new index of x2
      // real[next][temp] = r1 + (twiddle1k_real * imag[curr][threadIdx.x] + twiddle1k_imag * real[curr][threadIdx.x]) - (twiddle2k_real * r3 - twiddle2k_imag * i3) - (twiddle3k_real * i4 + twiddle3k_imag * r4);
      // imag[next][temp] = i1 - (twiddle1k_real * real[curr][threadIdx.x] - twiddle1k_imag * imag[curr][threadIdx.x]) - (twiddle2k_real * i3 + twiddle2k_imag * r3) + (twiddle3k_real * r4 - twiddle3k_imag * i4);

    }
    else if (threadIdx.x < (SIZE >> 1) + (SIZE >> 2))
    {
      //x3 = x1 - twiddle1k * x2 + twiddle2k * x3 - twiddle3k * x4
      int twiddle_index = pos_in_unit * (SIZE >> 1) / unit_size;
      float twiddle2k_real = roots_real_local[twiddle_index];
      float twiddle2k_imag = roots_imag_local[twiddle_index];

      //store twiddle2k * x3
      float r = real[curr][threadIdx.x], i = imag[curr][threadIdx.x];
      real[curr][threadIdx.x] = twiddle2k_real * r - twiddle2k_imag * i;
      imag[curr][threadIdx.x] = twiddle2k_imag * r + twiddle2k_real * i;
      __syncthreads();

      int ind1 = threadIdx.x - (SIZE >> 1);
      int ind2 = threadIdx.x - (SIZE >> 2);
      int ind3 = threadIdx.x;
      int ind4 = threadIdx.x + (SIZE >> 2); 

      int new_pos = ((threadIdx.x - pos_in_unit - (SIZE >> 1)) << 2) + ((unit_size >> 1) + pos_in_unit) ; //new index of x3
      real[next][new_pos] = real[curr][ind1] - real[curr][ind2] + real[curr][ind3] - real[curr][ind4];
      imag[next][new_pos] = imag[curr][ind1] - imag[curr][ind2] + imag[curr][ind3] - imag[curr][ind4];

      // temp = threadIdx.x - (SIZE >> 1); // index of x1
      // int ind2 = threadIdx.x - (SIZE >> 2);
      // int ind4 = threadIdx.x + (SIZE >> 2); 
      // float r1 = real[curr][temp];
      // float r2 = real[curr][ind2];
      // // float r3 = real[curr][threadIdx.x];
      // float r4 = real[curr][ind4];
      // float i1 = imag[curr][temp];
      // float i2 = imag[curr][ind2];
      // // float i3 = imag[curr][threadIdx.x];
      // float i4 = imag[curr][ind4];
      // temp = ((threadIdx.x - pos_in_unit - (SIZE >> 1)) << 2) + ((unit_size << 1) + pos_in_unit) ; //new index of x3
      // real[next][temp] = r1 - (twiddle1k_real * r2 - twiddle1k_imag * i2) + (twiddle2k_real * real[curr][threadIdx.x] - twiddle2k_imag * imag[curr][threadIdx.x]) - (twiddle3k_real * r4 - twiddle3k_imag * i4);
      // imag[next][temp] = i1 - (twiddle1k_real * i2 + twiddle1k_imag * r2) + (twiddle2k_real * imag[curr][threadIdx.x] + twiddle2k_imag * real[curr][threadIdx.x]) - (twiddle3k_real * i4 + twiddle3k_imag * r4);
    }
    else
    {
      //x4 = x1 +j twiddle1k * x2 - twiddle2k * x3 -j twiddle3k * x4
      int twiddle_index = pos_in_unit * 3 * (SIZE >> 2) / unit_size;
      float twiddle3k_real = roots_real_local[twiddle_index];
      float twiddle3k_imag = roots_imag_local[twiddle_index];

      //store twiddle1k * x2
      float r = real[curr][threadIdx.x], i = imag[curr][threadIdx.x];
      real[curr][threadIdx.x] = twiddle3k_real * r - twiddle3k_imag * i;
      imag[curr][threadIdx.x] = twiddle3k_imag * r + twiddle3k_real * i;
      __syncthreads();

      int ind1 = threadIdx.x - (SIZE >> 2) - (SIZE >> 1);
      int ind2 = threadIdx.x - (SIZE >> 1);
      int ind3 = threadIdx.x - (SIZE >> 2);
      int ind4 = threadIdx.x; 

      int new_pos = ((threadIdx.x - pos_in_unit - 3 * (SIZE >> 2)) << 2) + (3 * unit_size + pos_in_unit) ; //new index of x4
      real[next][new_pos] = real[curr][ind1] - imag[curr][ind2] - real[curr][ind3] + imag[curr][ind4];
      imag[next][new_pos] = imag[curr][ind1] + real[curr][ind2] - imag[curr][ind3] - real[curr][ind4];

      // temp = threadIdx.x - 3 * (SIZE >> 2); // index of x1
      // int ind2 = threadIdx.x - (SIZE >> 1);
      // int ind3 = threadIdx.x - (SIZE >> 2); 
      // float r1 = real[curr][temp];
      // float r2 = real[curr][ind2];
      // float r3 = real[curr][ind3];
      // // float r4 = real[curr][threadIdx.x];
      // float i1 = imag[curr][temp];
      // float i2 = imag[curr][ind2];
      // float i3 = imag[curr][ind3];
      // // float i4 = imag[curr][threadIdx.x];
      // temp = ((threadIdx.x - pos_in_unit - 3 * (SIZE >> 2)) << 2) + (3 * unit_size + pos_in_unit) ; //new index of x4
      // real[next][temp] = r1 - (twiddle1k_real * i2 + twiddle1k_imag * r2) - (twiddle2k_real * r3 - twiddle2k_imag * i3) + (twiddle3k_real * imag[curr][threadIdx.x] + twiddle3k_imag * real[curr][threadIdx.x]);
      // imag[next][temp] = i1 + (twiddle1k_real * r2 - twiddle1k_imag * i2) - (twiddle2k_real * i3 + twiddle2k_imag * r3) - (twiddle3k_real * real[curr][threadIdx.x] - twiddle3k_imag * imag[curr][threadIdx.x]);

    }
  __syncthreads();
    next = curr;
    curr = 1 - curr;
  }
  return curr;
}

__device__ char inverseFFT_radix4(float (*real)[SIZE], float (*imag)[SIZE])
{

  __shared__ float roots_real_local[SIZE];
  __shared__ float roots_imag_local[SIZE];


  float angle = TWO_PI * threadIdx.x / SIZE;
  roots_real_local[threadIdx.x] = __cosf(angle);
  roots_imag_local[threadIdx.x] = __sinf(angle);
  

  __syncthreads();

  char curr = 0;
  char next = 1;

  // int span = SIZE >> 2;
  int temp;

  for (int unit_size = 1; unit_size < SIZE ; unit_size <<= 2)
  {
    int pos_in_unit = threadIdx.x % unit_size;
    temp = pos_in_unit * (SIZE >> 2) / unit_size; // twiddle index
    float twiddle1k_real = roots_real_local[temp];
    float twiddle1k_imag = roots_imag_local[temp];
    float twiddle2k_real = roots_real_local[temp << 1];
    float twiddle2k_imag = roots_imag_local[temp << 1];
    float twiddle3k_real = roots_real_local[3*temp];
    float twiddle3k_imag = roots_imag_local[3*temp];


    if (threadIdx.x < (SIZE >> 2))
    {
      //x1 = x1 + twiddle1k * x2 + twiddle2k * x3 + twiddle3k * x4
      temp = threadIdx.x + (SIZE >> 2); // index of x2
      int ind3 = threadIdx.x + (SIZE >> 1);
      int ind4 = ind3 + (SIZE >> 2); 
      float r2 = real[curr][temp];
      float r3 = real[curr][ind3];
      float r4 = real[curr][ind4];
      float i2 = imag[curr][temp];
      float i3 = imag[curr][ind3];
      float i4 = imag[curr][ind4];
      temp = ((threadIdx.x - pos_in_unit) << 2) + pos_in_unit; //new index of x1
      real[next][temp] = real[curr][threadIdx.x] + (twiddle1k_real * r2 - twiddle1k_imag * i2) + (twiddle2k_real * r3 - twiddle2k_imag * i3) + (twiddle3k_real * r4 - twiddle3k_imag * i4);
      imag[next][temp] = imag[curr][threadIdx.x] + (twiddle1k_real * i2 + twiddle1k_imag * r2) + (twiddle2k_real * i3 + twiddle2k_imag * r3) + (twiddle3k_real * i4 + twiddle3k_imag * r4);
    }
    else if (threadIdx.x < SIZE >> 1)
    {
      //x2 = x1 - j*twiddle1k * x2 - twiddle2k * x3 + j twiddle3k * x4
      temp = threadIdx.x - (SIZE >> 2); // index of x1
      int ind3 = threadIdx.x + (SIZE >> 2);
      int ind4 = threadIdx.x + (SIZE >> 1); 
      float r2 = real[curr][threadIdx.x];
      float r3 = real[curr][ind3];
      float r4 = real[curr][ind4];
      float i2 = imag[curr][threadIdx.x];
      float i3 = imag[curr][ind3];
      float i4 = imag[curr][ind4];
      temp = ((threadIdx.x - pos_in_unit - (SIZE >> 2)) << 2) + (unit_size + pos_in_unit) ; //new index of x2
      real[next][temp] = real[curr][temp] + (twiddle1k_real * i2 + twiddle1k_imag * r2) - (twiddle2k_real * r3 - twiddle2k_imag * i3) - (twiddle3k_real * i4 + twiddle3k_imag * r4);
      imag[next][temp] = imag[curr][temp] - (twiddle1k_real * r2 - twiddle1k_imag * i2) - (twiddle2k_real * i3 + twiddle2k_imag * r3) + (twiddle3k_real * r4 - twiddle3k_imag * i4);

    }
    else if (threadIdx.x < (SIZE >> 1) + (SIZE >> 2))
    {
      //x3 = x1 - twiddle1k * x2 + twiddle2k * x3 - twiddle3k * x4
      temp = threadIdx.x - (SIZE >> 1); // index of x1
      int ind2 = threadIdx.x - (SIZE >> 2);
      int ind4 = threadIdx.x + (SIZE >> 2); 
      float r2 = real[curr][ind2];
      float r3 = real[curr][threadIdx.x];
      float r4 = real[curr][ind4];
      float i2 = imag[curr][ind2];
      float i3 = imag[curr][threadIdx.x];
      float i4 = imag[curr][ind4];
      temp = ((threadIdx.x - pos_in_unit - (SIZE >> 1)) << 2) + ((unit_size >> 1) + pos_in_unit) ; //new index of x3
      real[next][temp] = real[curr][temp] - (twiddle1k_real * r2 - twiddle1k_imag * i2) + (twiddle2k_real * r3 - twiddle2k_imag * i3) - (twiddle3k_real * r4 - twiddle3k_imag * i4);
      imag[next][temp] = imag[curr][temp] - (twiddle1k_real * i2 + twiddle1k_imag * r2) + (twiddle2k_real * i3 + twiddle2k_imag * r3) - (twiddle3k_real * i4 + twiddle3k_imag * r4);
    }
    else
    {
      //x4 = x1 +j twiddle1k * x2 - twiddle2k * x3 -j twiddle3k * x4
      temp = threadIdx.x - 3 * (SIZE >> 2); // index of x1
      int ind2 = threadIdx.x - (SIZE >> 1);
      int ind3 = threadIdx.x - (SIZE >> 2); 
      float r2 = real[curr][ind2];
      float r3 = real[curr][ind3];
      float r4 = real[curr][threadIdx.x];
      float i2 = imag[curr][ind2];
      float i3 = imag[curr][ind3];
      float i4 = imag[curr][threadIdx.x];
      temp = ((threadIdx.x - pos_in_unit - 3 * (SIZE >> 2)) << 2) + (3 * unit_size + pos_in_unit) ; //new index of x3
      real[next][temp] = real[curr][temp] - (twiddle1k_real * i2 + twiddle1k_imag * r2) - (twiddle2k_real * r3 - twiddle2k_imag * i3) + (twiddle3k_real * i4 + twiddle3k_imag * r4);
      imag[next][temp] = imag[curr][temp] + (twiddle1k_real * r2 - twiddle1k_imag * i2) - (twiddle2k_real * i3 + twiddle2k_imag * r3) - (twiddle3k_real * r4 - twiddle3k_imag * i4);

    }
  __syncthreads();
    next = curr;
    curr = 1 - curr;
  }
  return curr;
}

__device__ char forwardFFT(int pos, float (*real)[SIZE], float (*imag)[SIZE])
{
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
      temp = pos + span; // index of x2
      float r1 = real[curr][pos];
      float r2 = real[curr][temp];
      float i1 = imag[curr][pos];
      float i2 = imag[curr][temp];
      temp = (pos << 1) - pos_in_unit; //new index of x1
      real[next][temp] = r1 + (twiddle_real * r2 - twiddle_imag * i2);
      imag[next][temp] = i1 + (twiddle_real * i2 + twiddle_imag * r2);
    }
    else
    {
      // x2 = x1 - twiddle *x2
      temp = pos - span; // index of x1
      float r1 = real[curr][temp];
      float r2 = real[curr][pos];
      float i1 = imag[curr][temp];
      float i2 = imag[curr][pos];
      temp = ((pos - span) << 1) - pos_in_unit + unit_size; //new index of x2
      real[next][temp] = r1 - (twiddle_real * r2 - twiddle_imag * i2);
      imag[next][temp] = i1 - (twiddle_real * i2 + twiddle_imag * r2);
    }
    __syncthreads();
    next = curr;
    curr = 1 - curr;
  }
  return curr;
}

__device__ char inverseFFT(int pos, float (*real)[SIZE], float (*imag)[SIZE])
{
  __shared__ float roots_real_local[SIZE/2];
  __shared__ float roots_imag_local[SIZE/2];

  if (pos < SIZE / 2)
  {
    float angle = TWO_PI * pos / SIZE;
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
      temp = pos + span; // index of x2
      float r1 = real[curr][pos];
      float r2 = real[curr][temp];
      float i1 = imag[curr][pos];
      float i2 = imag[curr][temp];
      temp = (pos << 1) - pos_in_unit; //new index of x1
      real[next][temp] = r1 + (twiddle_real * r2 - twiddle_imag * i2);
      imag[next][temp] = i1 + (twiddle_real * i2 + twiddle_imag * r2);
    }
    else
    {
      // x2 = x1 - twiddle *x2
      temp = pos - span; // index of x1
      float r1 = real[curr][temp];
      float r2 = real[curr][pos];
      float i1 = imag[curr][temp];
      float i2 = imag[curr][pos];
      temp = ((pos - span) << 1) - pos_in_unit + unit_size; //new index of x2
      real[next][temp] = r1 - (twiddle_real * r2 - twiddle_imag * i2);
      imag[next][temp] = i1 - (twiddle_real * i2 + twiddle_imag * r2);
    }
    __syncthreads();
    next = curr;
    curr = 1 - curr;
  }
  return curr;
}

__global__ void forwardFFTRow(float *real_image, float *imag_image)
{
  int row = blockIdx.x;
  int col = threadIdx.x;

  __shared__ float real[2][SIZE];
  __shared__ float imag[2][SIZE];


  int offset = row * SIZE + col;

  real[0][col] = real_image[offset];
  imag[0][col] = imag_image[offset];


  char curr = forwardFFT_radix4(real, imag);

  real_image[offset] = real[curr][col];
  imag_image[offset] = imag[curr][col];
}

__global__ void inverseFFTRow(float *real_image, float *imag_image)
{
  int row = blockIdx.x;
  int col = threadIdx.x;

  __shared__ float real[2][SIZE];
  __shared__ float imag[2][SIZE];


  int offset = row * SIZE + col;

  real[0][col] = real_image[offset];
  imag[0][col] = imag_image[offset];


  char curr = inverseFFT(col, real, imag);

  real_image[offset] = real[curr][col] / SIZE;
  imag_image[offset] = imag[curr][col] / SIZE;
}

__global__ void forwardFFTCol(float *real_image, float *imag_image)
{
  int col = blockIdx.x;
  int row = threadIdx.x;
  if (col >= SIZE / 8)
  {
    col += 3 * SIZE / 4;
  }
  __shared__ float real[2][SIZE];
  __shared__ float imag[2][SIZE];

  real[0][row] = real_image[row * SIZE + col];
  imag[0][row] = imag_image[row * SIZE + col];

  char curr = forwardFFT(row, real, imag);

  real_image[row * SIZE + col] = real[curr][row];
  imag_image[row * SIZE + col] = imag[curr][row];
}

__global__ void inverseFFTCol(float *real_image, float *imag_image)
{
  int col = blockIdx.x;
  int row = threadIdx.x;
  if (col >= SIZE / 8)
  {
    col += 3 * SIZE / 4;
  }
  __shared__ float real[2][SIZE];
  __shared__ float imag[2][SIZE];

  real[0][row] = real_image[row * SIZE + col];
  imag[0][row] = imag_image[row * SIZE + col];

  char curr = inverseFFT(row, real, imag);

  real_image[row * SIZE + col] = real[curr][row] / SIZE;
  imag_image[row * SIZE + col] = imag[curr][row] / SIZE;
}

__global__ void filter(float *real_image, float *imag_image)
{
  int row = blockIdx.x;
  int col = threadIdx.x;

  int eighth = SIZE / 8;
  int seven_eighth = SIZE - eighth;

  if ((row >= eighth && row < seven_eighth) || (col >= eighth && col < seven_eighth))
  {
    real_image[row * SIZE + col] = 0.f;
    imag_image[row * SIZE + col] = 0.f;
  }
}


//----------------------------------------------------------------
// END ADD KERNEL DEFINTIONS
//----------------------------------------------------------------

__host__ float filterImage(float *real_image, float *imag_image, int size_x, int size_y)
{
  // check that the sizes match up
  assert(size_x == SIZEX);
  assert(size_y == SIZEY);

  int matSize = size_x * size_y * sizeof(float);

  // These variables are for timing purposes
  float transferDown = 0, transferUp = 0, execution = 0;
  cudaEvent_t start,stop;

  float fftr = 0.f, fftc = 0.f, ifftr = 0.f, ifftc = 0.f, filter_time = 0.f, stream_creation = 0.f;
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
  // CUDA_ERROR_CHECK(cudaMemcpy(device_real,real_image,matSize,cudaMemcpyHostToDevice));
  // CUDA_ERROR_CHECK(cudaMemcpy(device_imag,imag_image,matSize,cudaMemcpyHostToDevice));
  
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

  // float *pinned_real_image, *pinned_imag_image;
  // CUDA_ERROR_CHECK(cudaMallocHost((void **) &pinned_real_image, matSize));
  // CUDA_ERROR_CHECK(cudaMallocHost((void **) &pinned_imag_image, matSize));
  // memcpy(pinned_real_image, real_image, matSize);
  // memcpy(pinned_imag_image, imag_image, matSize);


  // printf("\n1st row real\n");
  // for (int i = 0; i < SIZE; ++i)
  // {
  //   printf("%f, ", real_image[i]);
  // }
  // printf("\n1st row imag\n");
  // for (int i = 0; i < SIZE; ++i)
  // {
  //   printf("%f, ", imag_image[i]);
  // }

  #define ASYNC_BLOCKS 16

  cudaStream_t stream[ASYNC_BLOCKS];
  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));
  for (int i = 0; i < ASYNC_BLOCKS; ++i)
  {
    cudaStreamCreate(&stream[i]);
  }

  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&stream_creation,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));

  for (int i = 0; i < ASYNC_BLOCKS; ++i)
  {
    CUDA_ERROR_CHECK(cudaMemcpyAsync(device_real + i * SIZE*SIZE/ASYNC_BLOCKS, real_image + i * SIZE*SIZE/ASYNC_BLOCKS,matSize/ASYNC_BLOCKS,cudaMemcpyHostToDevice, stream[i]));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(device_imag + i * SIZE*SIZE/ASYNC_BLOCKS, imag_image + i * SIZE*SIZE/ASYNC_BLOCKS,matSize/ASYNC_BLOCKS,cudaMemcpyHostToDevice, stream[i]));
    forwardFFTRow<<<SIZE / ASYNC_BLOCKS, SIZE, 0, stream[i]>>>(device_real + i * SIZE*SIZE/ASYNC_BLOCKS, device_imag + i * SIZE*SIZE/ASYNC_BLOCKS);
  }

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());


  // forwardFFTRow<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag);

  // CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
  // CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

  // printf("\n1st row tranform real\n");
  // for (int i = 0; i < SIZE; ++i)
  // {
  //   printf("%f, ", real_image[i]);
  // }
  // printf("\n1st row tranform imag\n");
  // for (int i = 0; i < SIZE; ++i)
  // {
  //   printf("%f, ", imag_image[i]);
  // }

  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&fftr,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));

  forwardFFTCol<<<SIZE / 4, SIZE, 0, filterStream>>>(device_real, device_imag);
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&fftc,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));
  filter<<<SIZE, SIZE, 0, filterStream>>>(device_real, device_imag);
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&filter_time,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));
  inverseFFTCol<<<SIZE / 4, SIZE, 0, filterStream>>>(device_real, device_imag);
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&ifftc,start_bis,stop_bis));

  CUDA_ERROR_CHECK(cudaEventRecord(start_bis,filterStream));
  for (int i = 0; i < ASYNC_BLOCKS; ++i)
  {
  }
  for (int i = 0; i < ASYNC_BLOCKS; ++i)
  {
    inverseFFTRow<<<SIZE / ASYNC_BLOCKS, SIZE, 0, stream[i]>>>(device_real + i * SIZE*SIZE/ASYNC_BLOCKS, device_imag + i * SIZE*SIZE/ASYNC_BLOCKS);
    CUDA_ERROR_CHECK(cudaMemcpyAsync(real_image + i * SIZE*SIZE/ASYNC_BLOCKS,device_real + i * SIZE*SIZE/ASYNC_BLOCKS,matSize/ASYNC_BLOCKS,cudaMemcpyDeviceToHost, stream[i]));
    CUDA_ERROR_CHECK(cudaMemcpyAsync(imag_image + i * SIZE*SIZE/ASYNC_BLOCKS,device_imag + i * SIZE*SIZE/ASYNC_BLOCKS,matSize/ASYNC_BLOCKS,cudaMemcpyDeviceToHost, stream[i]));
    // CUDA_ERROR_CHECK(cudaMemcpyAsync(pinned_real_image + i * SIZE*SIZE/ASYNC_BLOCKS,device_real + i * SIZE*SIZE/ASYNC_BLOCKS,matSize/ASYNC_BLOCKS,cudaMemcpyDeviceToHost, stream[i]));
    // CUDA_ERROR_CHECK(cudaMemcpyAsync(pinned_imag_image + i * SIZE*SIZE/ASYNC_BLOCKS,device_imag + i * SIZE*SIZE/ASYNC_BLOCKS,matSize/ASYNC_BLOCKS,cudaMemcpyDeviceToHost, stream[i]));
  }

  CUDA_ERROR_CHECK(cudaDeviceSynchronize());

  // inverseFFTRow<<<SIZE , SIZE, 0, filterStream>>>(device_real, device_imag);
  CUDA_ERROR_CHECK(cudaEventRecord(stop_bis,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop_bis));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&ifftr,start_bis,stop_bis));


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
  // CUDA_ERROR_CHECK(cudaMemcpy(real_image,device_real,matSize,cudaMemcpyDeviceToHost));
  // CUDA_ERROR_CHECK(cudaMemcpy(imag_image,device_imag,matSize,cudaMemcpyDeviceToHost));

  // Finish timing for transfer up
  CUDA_ERROR_CHECK(cudaEventRecord(stop,filterStream));
  CUDA_ERROR_CHECK(cudaEventSynchronize(stop));
  CUDA_ERROR_CHECK(cudaEventElapsedTime(&transferUp,start,stop));

  // Synchronize the stream
  CUDA_ERROR_CHECK(cudaStreamSynchronize(filterStream));
  // Destroy the stream
  CUDA_ERROR_CHECK(cudaStreamDestroy(filterStream));
  for (int i = 0; i < ASYNC_BLOCKS; ++i)
    cudaStreamDestroy(stream[i]);
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

  printf("  Stream Creation Time: %f ms\n\n", stream_creation);
  printf("  Row DFT Time: %f ms\n\n", fftr);
  printf("  Col DFT Time: %f ms\n\n", fftc);
  printf("  Filter Time: %f ms\n\n", filter_time);
  printf("  Col IDFT Time: %f ms\n\n", ifftc);
  printf("  Row IDFT Time: %f ms\n\n", ifftr);
  // Return the total time to transfer and execute
  return totalTime;
}

