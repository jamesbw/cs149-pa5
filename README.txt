James Whitbeck
jamesbw
CS149 PA5

Thread blocks, threads
======================

One thread block is created for each row of the image. One thread is allocated for each pixel in that row. Since the rows are never longer than 1024, we will not go over the hardware limit of threads per block. The data for each row is loaded into shared memory, as well as all the twiddles.

Fourier
=======
I implemented both the radix-2 and radix-4 FFTs. Radix-4 is used for sizes that are a power of 4, other radix-2 is used. Radix-4 is slightly faster. I tried implementing a general sqrt(SIZE)-radix function that works recursively. It worked on the 512x512 image, but not on the 1024x1024 - I believe because of hardware limits.
I did the FFTs out of place, instead on in-place for PA4. This removed the needed for bit-reversal. However, I had to allocate twice the space in shared memory. At each iteration, data was being passed back and forth between two arrays.

Optimizations
=============
Unrolling the loops in the FFTs using #pragma unroll was surprisingly beneficial. Since SIZE is fixed for the compiler, the unrolling could be performed entirely.
Not calculating transforms for the rows that will be zeroed out helped quite a bit.
Interestingly, there was little difference between row or column based FFTs, so I didn't do any transposition.
I overlapped the transfering of data from host to device and from device to host with the first kernels. To do so, I created separate streams and ran them asynchronously. This required pinning the host pages.
