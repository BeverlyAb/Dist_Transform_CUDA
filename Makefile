naive: naive.cu timer.c
	nvcc  -G -g -lcublas naive.cu timer.c -o naive2


cdt: naive.cu timer.c
	nvcc -I. naive.cu timer.c -o naive

run:
	qsub cuda2.sh
