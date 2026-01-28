# Left rotation with pycuda(GPU) and numpy(CPU)
# Import and initialize pycuda, numpy, and time
import time

import numpy as np
import pycuda.driver as cuda

# import pycuda.autoinit
from pycuda.compiler import SourceModule

#################### Array size
N = 1000000

#################### Create the input array on CPU (random integers between 1 and 100) and initialise the output array
a_cpu = np.random.uniform(1.0, 100.0, size=(N)).astype(np.uint32)
b_cpu = np.zeros(N, np.uint32)

#################### Write a GPU kernel
module = SourceModule(""" 
	__global__ void left_rotation(int* a_gpu, int* b_gpu, int N){
		// Global thread indices
		int id = threadIdx.x + blockIdx.x * blockDim.x;
		if(id < N){
			b_gpu[id] = a_gpu[(id + 1) % N];
		}
	}

""")

#################### Start GPU timing
start_gpu = cuda.Event()
end_gpu = cuda.Event()
start_gpu.record()

#################### Grid and Block size
block_size = 512
# grid_size = (N + block_size - 1) // block_size
grid_size = int(np.ceil(N / block_size))

#################### Launch the GPU kernel
func = module.get_function("left_rotation")
func(
    cuda.In(a_cpu),
    cuda.Out(b_cpu),
    np.uint32(N),
    grid=(grid_size, 1, 1),
    block=(block_size, 1, 1),
)

#################### End GPU timing
end_gpu.record()
cuda.Context.synchronize()
gpu_time = start_gpu.time_till(end_gpu) * 1e-3
print("Elapsed on GPU with PyCuda (sec): ", gpu_time)
print("---------------------")

#################### Sequential move on CPU for validation and comparison
b_seq = np.zeros(N, np.uint32)
start_cpu = time.time()
b_seq[-1] = a_cpu[0]
for i in range(N - 1):
    b_seq[i] = a_cpu[i + 1]
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("Elapsed time using CPU sequential for-loop (sec): ", cpu_time)
print("---------------------")

#################### Validation
dif = 0
for i in range(N):
    if b_cpu[i] != b_seq[i]:
        dif += 1
print("Validation: there are %d different element(s)! " % dif)
print("---------------------")
