# Computing Sampled Distance Transform on CUDA
This code is based on this(http://people.cs.uchicago.edu/~pff/papers/dt.pdf)  paper
  > P. Felzenszwalb, D. Huttenlocher "Distance Transforms of Sampled Functions"

This code was tested on Tesla M2090 with cuda 5.0 , gcc 4.4.3
The current speedup is about 3 times. I plan to work on optimizing the speed further. 
To run the code:
	make dt
	./dt




