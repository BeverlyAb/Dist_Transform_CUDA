naive: naive.cu timer.c
	nvcc  -G -g naive.cu timer.c -o naive


cdt: naive.cu timer.c
	nvcc -I. naive.cu timer.c -o naive

run:
	qsub cuda.sh
