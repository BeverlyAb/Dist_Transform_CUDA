naive: naive.cu timer.c
	nvcc naive.cu timer.c -o naive


cdt: naive.cu timer.c
	nvcc -I. naive.cu timer.c -o naive
