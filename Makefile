dt: naive.cu timer.c
	nvcc  -G -g -lcublas naive.cu timer.c -o dt

run:
	qsub cuda2.sh
