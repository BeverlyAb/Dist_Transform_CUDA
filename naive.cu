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
typedef unsigned char dtype2;

#define N_ (8 * 1024 * 1024)
#define MAX_THREADS 256
#define MAX_BLOCKS 64

#define CUDA_ERROR_CHECK
#define MIN(x,y) ((x < y) ? x : y)
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

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


inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
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


__global__ void
kernel_all_pix (dtype2 *input, dtype2 *output, unsigned int width,unsigned int height)
{
//One row stored in shared memory
//Number of blocks = height
//For now launch 1 block with height  number of threads
 __shared__  dtype2 scratch[400];

  unsigned int img_index = threadIdx.x*width;
/*
for(int j=0;j<width;j++)
{
	scratch[j] = input[img_index+j];
}
*/

//  unsigned int bid = gridDim.x * blockIdx.y + blockIdx.x;
//  unsigned int i = bid * blockDim.x + threadIdx.x;

 
  __syncthreads ();
 /* for(int j=0;j<width;j++)
	{
		if(j>20 && j<80)
			scratch[j] = 80;
		else
			scratch[j] = input[img_index+j];
		
	}

  __syncthreads ();
*/
	for(int j=0;j<width;j++)
	{
		if(j>20 && j<80)
		output[img_index+j]= 40;
		else
		output[img_index+j]= input[img_index+j];
	}

  __syncthreads ();


}




void all_pix (dtype2 *input, dtype2 *output, unsigned int width,unsigned int height)

{
//One row stored in shared memory
//Number of blocks = height
//For now launch 1 block with height  number of threads

/*
for(int i=0;i<height;i++)
{

   dtype2 scratch[400];
  unsigned int img_index = i*width;
for(int j=0;j<width;j++)
{
	scratch[j] = input[img_index+j];
}

}
 */
/*  for(int j=0;j<width;j++)
	{
//		if(j>20 && j<80)
//			scratch[j] = 80;
//		else
			scratch[j] = input[img_index+j];
		
	}
*/

for(int i=0;i<height;i++)
{
  unsigned int img_index = i*width;
	for(int j=width/2;j<width;j++)
	{
		output[img_index+j]= input[img_index+j];
//		input[img_index+j]= scratch[j];
	}

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
  int height = input-> height();
  int width = input->width();
for (int y = 0; y < out->height(); y++) {
    for (int x = 0; x < out->width(); x++) {
      imRef(out, x, y) = sqrt(imRef(out, x, y));
    }
  }
  image<uchar> *gray = imageFLOATtoUCHAR(out);
  //savePGM(gray, output_name);
 // delete input;
 // delete out;
 // delete gray;
//=====Simple func==============//
/*
  int N = width*height;

  dtype2 *h_idata, *h_odata, h_cpu;

  image<uchar> *output_img = new image<uchar>(width, height, false);
  h_idata = (dtype2*) malloc (N * sizeof (dtype2));
  h_odata = (dtype2*) malloc (N * sizeof (dtype2));
  h_idata = input->data;
  all_pix(h_idata,h_odata,width,height);

  
  output_img->data = input->data;
for(int i=40;i<45;i++)
{

  unsigned int img_index = i*width;
for(int j=100;j<110;j++)
{

//   if(j>20 && j<80)
	//imRef(output_img,j,i) = h_idata[img_index+j];
	//h_idata[img_index+j]=0i;
       printf("o=%d ",(output_img->data)[img_index+j]);
       printf("i=%d ",(input->data)[img_index+j]);
	printf("\n");
}


}




//printf("input->data=%d",input->data);
//printf("output_img->data= %d",output_img->data);
output_img->data = h_odata;

  //savePGM(input, output_name);
  savePGM(output_img, output_name);
*/
//================//

  int N = width*height;
  dtype2 *h_idata, *h_odata, h_cpu;
  dtype2 *d_idata, *d_odata;	

  image<uchar> *output_img = new image<uchar>(width, height, false);

  

  h_idata = (dtype2*) malloc (N * sizeof (dtype2));
  h_odata = (dtype2*) malloc (N * sizeof (dtype2));
  CUDA_CHECK_ERROR (cudaMalloc (&d_idata,N * sizeof (dtype2)));
  CUDA_CHECK_ERROR (cudaMalloc (&d_odata, N * sizeof (dtype2)));

  h_idata = input->data;

  dim3 gb(1,1, 1);
  dim3 tb(height, 1, 1);

  

  CUDA_CHECK_ERROR (cudaMemcpy (d_idata,h_idata, N * sizeof (dtype2), 
				cudaMemcpyHostToDevice));



  kernel_all_pix <<<gb, tb>>> (d_idata, d_odata, width,height);
  cudaThreadSynchronize ();

  kernel_all_pix <<<gb, tb>>> (d_idata, d_odata,width,height);
  CudaCheckError();

  cudaThreadSynchronize ();

  CUDA_CHECK_ERROR (cudaMemcpy (h_odata, d_odata, N* sizeof (dtype2), cudaMemcpyDeviceToHost));
  
  output_img->data = h_odata;
  
  savePGM(output_img, output_name);

/*===================================================*/


/*===================================================*/
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
