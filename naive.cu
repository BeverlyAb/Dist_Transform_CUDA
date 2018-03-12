//#include <cstdio>
//#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
//#include <cmath>
#include "timer.h"
#include "cuda_utils.h"
#include "pnmfile.h"
#include "imconv.h"
#include "dt.h"
typedef float dtype;

#define N_ (8 * 1024 * 1024)
#define MAX_THREADS 256
#define MAX_BLOCKS 64

#define MIN(x,y) ((x < y) ? x : y)


/* return the next power of 2 number that is larger than x */
unsigned int nextPow2( unsigned int x ) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

/* find out # of threads and # thread blocks for a particular kernel */
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
  if (whichKernel < 3)
    {
      /* 1 thread per element */
      threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
      blocks = (n + threads - 1) / threads;
    }
  else
    {
      /* 1 thread per 2 elements */
      threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
      blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }
  /* limit the total number of threads */
  if (whichKernel == 5)
    blocks = MIN(maxBlocks, blocks);
}

/* special type of reduction to account for floating point error */
dtype reduce_cpu(dtype *data, int n) {
  dtype sum = data[0];
  dtype c = (dtype)0.0;
  for (int i = 1; i < n; i++)
    {
      dtype y = data[i] - c;
      dtype t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
  return sum;
}



__global__ void
kernel_thresh (dtype *input, dtype *output, unsigned int n)
{
  __shared__  dtype scratch[MAX_THREADS];

  unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
  unsigned int i = bid * blockDim.x + threadIdx.x;

  if(i < n) {
    scratch[threadIdx.x] = input[i]; 
  } else {
    scratch[threadIdx.x] = 0;
  }
  __syncthreads ();

    if((threadIdx.x % (3)) == 0) {
      scratch[threadIdx.x] = scratch[threadIdx.x] + 20;
    }

    output[i] = scratch[threadIdx.x];
  __syncthreads ();
}


__global__ void
kernel0 (dtype *input, dtype *output, unsigned int n)
{
  __shared__  dtype scratch[MAX_THREADS];

  unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
  unsigned int i = bid * blockDim.x + threadIdx.x;

  if(i < n) {
    scratch[threadIdx.x] = input[i]; 
  } else {
    scratch[threadIdx.x] = 0;
  }
  __syncthreads ();

  for(unsigned int s = 1; s < blockDim.x; s = s << 1) {
    if((threadIdx.x % (2 * s)) == 0) {
      scratch[threadIdx.x] += scratch[threadIdx.x + s];
    }
    __syncthreads ();
  }

  if(threadIdx.x == 0) {
    output[bid] = scratch[0];
  }
}

int 
main(int argc, char** argv)
{

  if (argc != 3) {
    fprintf(stderr, "usage: %s input(pbm) output(pgm)\n", argv[0]);
    return 1;
  }
  char *input_name = argv[1];
  char *output_name = argv[2];
  image<uchar> *input = loadPGM(input_name);
  image<float> *out = dt(input);
//for (int y = 0; y < out->height(); y++) {
//    for (int x = 0; x < out->width(); x++) {
 //     imRef(out, x, y) = sqrt(imRef(out, x, y));
 //   }
 // }
 // image<uchar> *gray = imageFLOATtoUCHAR(out);
//  savePGM(input, output_name);
 // delete input;
 // delete out;
 // delete gray;

//================//

  int tN = 256;
  dtype *th_idata, *th_odata, th_cpu;
  dtype *td_idata, *td_odata;	

  th_idata = (dtype*) malloc (tN * sizeof (dtype));
  th_odata = (dtype*) malloc (tN * sizeof (dtype));
  CUDA_CHECK_ERROR (cudaMalloc (&td_idata,tN * sizeof (dtype)));
  CUDA_CHECK_ERROR (cudaMalloc (&td_odata, tN * sizeof (dtype)));
  for(int i = 0; i < tN; i++) {
    th_idata[i] = i;

	}

  dim3 tgb(1,1, 1);
  dim3 ttb(tN, 1, 1);

  /* warm up */
  

  CUDA_CHECK_ERROR (cudaMemcpy (td_idata,th_idata, tN * sizeof (dtype), 
				cudaMemcpyHostToDevice));



  kernel_thresh <<<tgb, ttb>>> (td_idata, td_odata, tN);
  cudaThreadSynchronize ();

  kernel_thresh <<<tgb, ttb>>> (td_idata, td_odata,tN);
  cudaThreadSynchronize ();

  CUDA_CHECK_ERROR (cudaMemcpy (th_odata, td_odata, tN* sizeof (dtype), cudaMemcpyDeviceToHost));
  
for(int i=0;i<20;i++)
	{
		printf("%d=%0.1f ",i,th_odata[i]);
	}
printf("\n");


/*===================================================*/



  return 0;
}
