CC = nvcc
GEN = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75
LINK = -lnvidia-ml
CFLAG = $(GEN) $(LINK)
helper = cuda_helper.cpp nvml_helper.cpp

flops: flops.cu
	$(CC) $(CFLAG) flops.cu $(helper) -o flops

nvget: nvget.cu
	$(CC) $(CFLAG) nvget.cu $(helper) -o nvget

.PHONY: clean
clean:
	-rm ./flops
	-rm ./nvget