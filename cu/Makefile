CC = nvcc
CFLAG = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75

flops: flops.cu
	$(CC) $(CFLAG) flops.cu cuda_helper.cpp -o flops

.PHONY: clean
clean:
	-rm ./flops