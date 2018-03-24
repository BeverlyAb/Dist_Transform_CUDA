# Computing Sampled Distance Transform on CUDA
This code is based on this(http://people.cs.uchicago.edu/~pff/papers/dt.pdf)  paper
  > P. Felzenszwalb, D. Huttenlocher "Distance Transforms of Sampled Functions"

This code was tested on Tesla M2090 with cuda 5.0 , gcc 4.4.3
The current speedup is about 3 times. I plan to work on optimizing the speed further. 
To run the code:
    make dt

    make run #if running on server(HPC)

    ./dt #if running directly

In the img folder place all your images. The current version of the code assumes same size and orientation for all images.
In the original implementation(https://cs.brown.edu/~pff/dt/) the image is thresholded before computing the transform we have removed the thresholding part. 




