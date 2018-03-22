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
#include <vector>
#include <string>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
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
using namespace std;
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
  if( CUBLAS_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
    //	getch(); //causing error as nvcc unable to find conio.h
		cudaDeviceReset(); 
		assert(0); 
  }
}
int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
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
  //float temp_sum = 0.0;
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
  
  unsigned int row_num = blockIdx.y * blockDim.x + threadIdx.x;
  unsigned int img_index = (row_num)*width;
  unsigned int offset_curr_image = blockIdx.x * (width*height);
  //As the current pixel will be offset by the number of pixels before
  __syncthreads ();
  
  if(row_num < height)
  {
    float f[MAX_WIDTH_HEIGHT];
    
    for (int x = 0; x < width; x++) 
    {
      f[x] = input[offset_curr_image + img_index+x];
    }
    dt_i(f, width);
    
    // 	__syncthreads ();
    
    for (int x = 0; x < width; x++) 
    {
      output[offset_curr_image + img_index+x] = f[x];
    }
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
  //char *input_name = argv[1];
  //char *output_name = argv[2];
//---Print device properties----//

 int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf(" Shared Memory per block(bytes): %d\n", prop.sharedMemPerBlock);
    printf("Total memory available(bytes): %d\n",prop.totalConstMem);
  }
//-----------------------------//

  string dir = string("./img/");
  vector<string> files = vector<string>();

  getdir(dir,files);
  cout <<files.size();

  string in_name = "img/" + files[2];
  image<uchar> *input = loadPGM(in_name.c_str());
  //Read outside to read width and height , so that we can use malloc
  //to assign it to the entire image. 
  int height = input-> height();
  int width = input->width();
  int Num_Pixels= height*width;
  int Num_Files= files.size()-2; //-2 because . and .. are also included in list of files besides the image
  int Total_Num_Pixels = Num_Files*Num_Pixels;

  dtype *h_idata, *h_odata,*hodata2;
  dtype *d_idata, *d_odata;	
  
  h_idata = (dtype*) malloc (Total_Num_Pixels * sizeof (dtype));
  h_odata = (dtype*) malloc (Total_Num_Pixels * sizeof (dtype));
  hodata2 = (dtype*) malloc (Total_Num_Pixels * sizeof (dtype));
  CUDA_CHECK_ERROR (cudaMalloc (&d_idata,Total_Num_Pixels * sizeof (dtype)));
  CUDA_CHECK_ERROR (cudaMalloc (&d_odata, Total_Num_Pixels * sizeof (dtype)));
  
  

  for (unsigned int i = 0;i < files.size();i++)//change 5 to files.size() 
 {
//	cout << files[i] << " \n";		
  }
  
  for (unsigned int i = 2;i < files.size();i++)//change 5 to files.size() 
  {
    int img_index = i-2;
    string in_name = "img/" + files[i];
    image<uchar> *input = loadPGM(in_name.c_str());
    int curr_index= img_index* Num_Pixels;
    for(int i=0;i<height;i++)
    {
      for(int j=0;j<width;j++)
      {
        h_idata[curr_index+(i*width+j)] = (float)input->data[i*width+j];
      }
    }

  } 

  
  
  //----------------------------//
  //int N = width*height;
  
  struct stopwatch_t* timer = NULL;
  long double t_kernel_ap1,t_kernel_ap2,t_kernel_ap3,t_kernel_ap4;
  
  
  image<dtype> *input_float = imageUCHARtoFLOAT(input);
  image<dtype> *output_img = new image<dtype>(width, height, false);
  
  
  stopwatch_init();
  timer = stopwatch_create();
  
  
  
  
  unsigned int max_width_height = 0;
  max_width_height = (height>width)? height:width;
  
  int num_blocks =0;
  num_blocks =  (max_width_height/MAX_THREADS) + 1;  
  int num_threads = MAX_THREADS;
  dim3 gb(Num_Files,num_blocks, 1);
  dim3 tb(num_threads, 1, 1);
  printf("max_width_height:%d  MAX_THREADS: %d num_blocks:%d threads:%d\n",max_width_height,num_threads,num_blocks,num_threads); 
  
  
  CUDA_CHECK_ERROR (cudaMemcpy (d_idata,h_idata, Total_Num_Pixels * sizeof (dtype), 
  cudaMemcpyHostToDevice));
  
  
  kernel_all_pix_float <<<gb, tb>>> (d_idata, d_odata, width,height);
  cudaThreadSynchronize ();
  
  
  stopwatch_start(timer);
  kernel_all_pix_float <<<gb, tb>>> (d_idata, d_odata,width,height);
 
  cudaThreadSynchronize ();
  t_kernel_ap1 = stopwatch_stop(timer); 

 //CudaCheckError();
  
  
  
  stopwatch_start(timer);
  
  int m1,n1;
	cublasHandle_t handle;
	dtype3 alpha = 1.;
	dtype3 beta  = 0.;
	m1 = height;
	n1 = width;
  
  for(int i=0;i<Num_Files;i++)
  {
    dtype *curr_image_o,*curr_image_i;
    curr_image_o = d_odata+(Num_Pixels*i);
    curr_image_i = d_idata+(Num_Pixels*i);
    cublasSafeCall(cublasCreate(&handle));
	  cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m1, n1, &alpha, curr_image_o, n1, &beta, curr_image_o, n1, curr_image_i, m1));
  
  }
  
  t_kernel_ap4 = stopwatch_stop(timer);
  dim3 gb2(Num_Files,num_blocks, 1);
  dim3 tb2(num_threads, 1, 1);
  
  // dtype *hidata2;
  // hidata2 = transpose_img->data;
  
  // dtype *hodata2;
  // hodata2 = transpose_img->data;
  
  /*
  CUDA_CHECK_ERROR (cudaMemcpy (d_idata,hidata2, N * sizeof (dtype), 
  cudaMemcpyHostToDevice));
  */
  
  kernel_all_pix_float <<<gb2, tb2>>> (d_idata, d_odata, height,width); //reversed width,height
  cudaThreadSynchronize ();
  
  stopwatch_start(timer);   
  
  
  kernel_all_pix_float <<<gb2, tb2>>> (d_idata, d_odata,height,width); //reversed width,height
 
  cudaThreadSynchronize ();
  t_kernel_ap2 = stopwatch_stop(timer);
  
  
  
 
	m1 = width;
	n1 = height;
  
  

  stopwatch_start(timer);   
for(int i=0;i<Num_Files;i++)
  {
    dtype *curr_image_o,*curr_image_i;
    curr_image_o = d_odata+(Num_Pixels*i);
    curr_image_i = d_idata+(Num_Pixels*i);
    cublasSafeCall(cublasCreate(&handle));
	  cublasSafeCall(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m1, n1, &alpha, curr_image_o, n1, &beta, curr_image_o, n1, curr_image_i, m1));
  
  }
  CUDA_CHECK_ERROR (cudaMemcpy (hodata2, d_idata, Total_Num_Pixels* sizeof (dtype), cudaMemcpyDeviceToHost));
 t_kernel_ap3 = stopwatch_stop(timer);
printf("Time for  transpose: %Lg %Lg seconds\n",t_kernel_ap3,t_kernel_ap4); 
  
  //image<uchar> *out_res= imageFLOATtoUCHAR(output_img,0.0,255.0);
  
  output_img->data = hodata2; 
  float min_val = output_img->data[0];
  float max_val = min_val;
  //long double t_min_max3;

  printf("Time to execute DT: %Lg secs\n",t_kernel_ap1+t_kernel_ap2);
    
  for(int file_no = 0;file_no<Num_Files;file_no++)
  {
    int img_index= file_no*Num_Pixels;
    float min_val = hodata2[img_index];
    float max_val = min_val;
    for(int i=0;i<height;i++)
    {
      for(int j=0;j<width;j++)
      {
        if(max_val < hodata2[img_index+i*width+j]) max_val = hodata2[img_index+i*width+j];
        if(min_val > hodata2[img_index+i*width+j]) min_val = hodata2[img_index+i*width+j];
        
      }
    } 
    float scale = 255/(sqrt(max_val)-sqrt(min_val));
  //  printf("max:%0.2f min:%0.2f s=%0.2f\n",max_val,min_val,scale);	
    image<uchar> *out_res = new image<uchar>(width,height,false);
    for(int i=0;i<height;i++)
    {
      for(int j=0;j<width;j++)
      {
        out_res->data[i*width+j] = (uchar)(scale * (sqrt(hodata2[img_index+i*width+j])-sqrt(min_val)));		//Hardcoding scale value here, need to find min ,max automatically and do it properly
        //out_res->data[i*width+j] = (uchar)(output_img->data[i*width+j]);		//Hardcoding scale value here, need to find min ,max automatically and do it properly
      }
    }
  	string out_name = "result_img/res_"  + files[file_no+2]; //Due to the file offset
    	savePGM(out_res, out_name.c_str());
  }
  
  
  
  //------------------------------------------------
  
  
  /*===================================================*/
  int tN = 256;
  dtype *th_idata, *th_odata;
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
  
  for(int i=0;i<5;i++)
	{
		printf("%d=%0.1f ",i,th_odata[i]);
	}
  printf("\n");
  
  
  return 0;
}
