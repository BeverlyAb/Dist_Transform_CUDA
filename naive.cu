//#include <cstdio>
//#include <cstdlib>
#include <assert.h> //This is assert to check for conditions in kernels
//#include <conio.h>
//#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
//#include <cmath>
#include "timer.h"
#include "cuda_utils.h"
#include "pnmfile.h"
#include "imconv.h"
#include "dt.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
typedef float dtype;
typedef unsigned char dtype2;
typedef float dtype3;
#include <cublas_v2.h>

#define N_ (8 * 1024 * 1024)
#define MAX_THREADS 256
#define MAX_BLOCKS 64
#define MAX_WIDTH_HEIGHT 500
#define CUDA_ERROR_CHECK
#define MIN(x,y) ((x < y) ? x : y)
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define INFDT 1E20

/**********************/
/* cuBLAS ERROR CHECK */
/**********************/
#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if( CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
	//	getch(); //causing error as nvcc unable to find conio.h
		cudaDeviceReset(); 
		assert(0); 
    }
}



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


__device__ void

dt_i(float f[], int n) 
{
    float d[MAX_WIDTH_HEIGHT];
    int v[MAX_WIDTH_HEIGHT];
    float z[MAX_WIDTH_HEIGHT+1];
    int k = 0;
    float temp_sum = 0.0;
    v[0] = 0;
    z[0] = -INFDT;
    z[1] = +INFDT;
	
    for (int q = 1; q <= n-1; q++) {
      float s  = ((f[q]+(q*q))-(f[v[k]]+(v[k]*v[k])))/(2*q-2*v[k]);
      while (s <= z[k]) {
        k--;
	
        s  = ((f[q]+(q*q))-(f[v[k]]+(v[k]*v[k])))/(2*q-2*v[k]);
       // s  = ((f[q]+(q*q))-(f[t]+(v[k]*v[k])))/(2*q-2*v[k]);
      }
      k++;
	
      v[k] = q;
      z[k] = s;
      z[k+1] = +INFDT;
    }
  
    k = 0;
    for (int q = 0; q <= n-1; q++) {
      while (z[k+1] < q)
        k++;
      d[q] = (q-v[k])*(q-v[k]) + f[v[k]] ; //!!!!!REMOVE_TEMP_SUM DEBUG - added for debugging
    }
  
    for(int q=0;q<n;q++)
    { 
      f[q] = d[q];
    }
  
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
kernel_all_pix_float (dtype *input, dtype *output, unsigned int width,unsigned int height)
{
//One row stored in shared memory
//Number of blocks = height
//For now launch 1 block with height  number of threads
 //__shared__  dtype2 scratch[400];

  //unsigned int img_index = threadIdx.x*width;

    unsigned int row_num = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int img_index = (row_num)*width;

  __syncthreads ();

    float f[MAX_WIDTH_HEIGHT];

    for (int x = 0; x < width; x++) 
    {
      f[x] = input[img_index+x];
    }
    dt_i(f, width);
  
  	__syncthreads ();
  
    for (int x = 0; x < width; x++) 
    {
      output[img_index+x] = f[x];
    }

  	__syncthreads ();
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
//------------Basic DT ------//
 image<float> *out = dt(input);
  int height = input-> height();
  int width = input->width();
//-----------------------------//
  int N = width*height;

  struct stopwatch_t* timer = NULL;
  long double t_kernel_ap1,t_kernel_ap2;

  dtype *h_idata, *h_odata, h_cpu;
  dtype *d_idata, *d_odata;	

  image<dtype> *input_float = imageUCHARtoFLOAT(input);
  image<dtype> *output_img = new image<dtype>(width, height, false);

  
  stopwatch_init();
  timer = stopwatch_create();

  h_idata = (dtype*) malloc (N * sizeof (dtype));
  h_odata = (dtype*) malloc (N * sizeof (dtype));
  CUDA_CHECK_ERROR (cudaMalloc (&d_idata,N * sizeof (dtype)));
  CUDA_CHECK_ERROR (cudaMalloc (&d_odata, N * sizeof (dtype)));



  h_idata = input_float->data;

  dim3 gb(1,1, 1);
  dim3 tb(height, 1, 1);

  


  CUDA_CHECK_ERROR (cudaMemcpy (d_idata,h_idata, N * sizeof (dtype), 
				cudaMemcpyHostToDevice));


  kernel_all_pix_float <<<gb, tb>>> (d_idata, d_odata, width,height);
  cudaThreadSynchronize ();
  
  
  stopwatch_start(timer);
  kernel_all_pix_float <<<gb, tb>>> (d_idata, d_odata,width,height);
  CudaCheckError();

  cudaThreadSynchronize ();

  //CUDA_CHECK_ERROR (cudaMemcpy (h_odata, d_odata, N* sizeof (dtype2), cudaMemcpyDeviceToHost));
//  CUDA_CHECK_ERROR (cudaMemcpy (h_odata, d_odata, N* sizeof (dtype), cudaMemcpyDeviceToHost));
 

 image<dtype> *transpose_img = new image<dtype>(height, width, false); //Note: Here height, width oppositve of above, doesn't matter though because memory allocated same

  /*----Below loop is to do transpose, make this part parallel later*/
/*   for(int i=0;i<width;i++)
  {
    for(int j=0;j<height;j++)
    {
      transpose_img->data[i*height+j] = h_odata[j*width+i];  

    }
  }
*/

 	int m1,n1;
	cublasHandle_t handle;
	dtype3 alpha = 1.;
	dtype3 beta  = 0.;
	m1 = height;
	n1 = width;
	cublasSafeCall(cublasCreate(&handle));
	cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m1, n1, &alpha, d_odata, n1, &beta, d_odata, n1, d_idata, m1));
  dim3 gb2(2,1, 1);
  dim3 tb2(200, 1, 1);

  dtype *hidata2;
  hidata2 = transpose_img->data;

  dtype *hodata2;
  hodata2 = transpose_img->data;

/*
  CUDA_CHECK_ERROR (cudaMemcpy (d_idata,hidata2, N * sizeof (dtype), 
        cudaMemcpyHostToDevice));
*/
t_kernel_ap1 = stopwatch_stop(timer);

  kernel_all_pix_float <<<gb2, tb2>>> (d_idata, d_odata, height,width); //reversed width,height
  cudaThreadSynchronize ();

stopwatch_start(timer);   


  kernel_all_pix_float <<<gb2, tb2>>> (d_idata, d_odata,height,width); //reversed width,height
  CudaCheckError();

  cudaThreadSynchronize ();


/*
  //CUDA_CHECK_ERROR (cudaMemcpy (h_odata, d_odata, N* sizeof (dtype2), cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERROR (cudaMemcpy (hodata2, d_odata, N* sizeof (dtype), cudaMemcpyDeviceToHost));
  
  
//This section is to do the tranpose again

   for(int i=0;i<width;i++)
  {
    for(int j=0;j<height;j++)
    {
      output_img->data[j*width+i] = hodata2[i*height+j];
    }
  }
 */ 
	m1 = width;
	n1 = height;
	cublasSafeCall(cublasCreate(&handle));
	cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m1, n1, &alpha, d_odata, n1, &beta, d_odata, n1, d_idata, m1));

  CUDA_CHECK_ERROR (cudaMemcpy (hodata2, d_idata, N* sizeof (dtype), cudaMemcpyDeviceToHost));
  
t_kernel_ap2 = stopwatch_stop(timer);
  
  //image<uchar> *out_res= imageFLOATtoUCHAR(output_img,0.0,255.0);
    image<uchar> *out_res = new image<uchar>(width,height,false);
    output_img->data = hodata2; 
    float min_val = output_img->data[0];
    float max_val = min_val;
long double t_min_max3;
stopwatch_start(timer);
for(int i=0;i<height;i++)
{
	for(int j=0;j<width;j++)
	{
		if(max_val < output_img->data[i*width+j]) max_val = output_img->data[i*width+j];
		if(min_val > output_img->data[i*width+j]) min_val = output_img->data[i*width+j];
	
	}
} 
t_min_max3 = stopwatch_stop(timer);  
printf("Time to execute DT: %Lg %Lg minmax:%Lg secs\n",t_kernel_ap1,t_kernel_ap2,t_min_max3);
	float scale = 255/(sqrt(max_val)-sqrt(min_val));
	printf("max:%0.2f min:%0.2f s=%0.2f\n",max_val,min_val,scale);	
  for(int i=0;i<height;i++)
  {
  	for(int j=0;j<width;j++)
  	{
  		out_res->data[i*width+j] = (uchar)(scale * (sqrt(output_img->data[i*width+j])-sqrt(min_val)));		//Hardcoding scale value here, need to find min ,max automatically and do it properly
  		//out_res->data[i*width+j] = (uchar)(output_img->data[i*width+j]);		//Hardcoding scale value here, need to find min ,max automatically and do it properly
  	}
  }



  savePGM(out_res, output_name);

//------------------------------------------------


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



  kernel_thresh <<<tgb, ttb>>> (td_idata, td_odata, tN);//To warm up
  cudaThreadSynchronize ();

  kernel_thresh <<<tgb, ttb>>> (td_idata, td_odata,tN);
  cudaThreadSynchronize ();

  CUDA_CHECK_ERROR (cudaMemcpy (th_odata, td_odata, tN* sizeof (dtype), cudaMemcpyDeviceToHost));
  
for(int i=0;i<20;i++)
	{
		printf("%d=%0.1f ",i,th_odata[i]);
	}
printf("\n");


  return 0;
}
