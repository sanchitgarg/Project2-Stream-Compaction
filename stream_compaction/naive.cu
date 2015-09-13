#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <cmath>
#include <iostream>

namespace StreamCompaction {
namespace Naive {

#define SHOW_TIMING 0
int numBlocks, numThreads = 256;

int * dev_odata;
int * dev_idata;

 __global__ void scanStep(int n, int jump, int *odata, int *idata)
 {
	 int index = threadIdx.x + (blockIdx.x * blockDim.x);

	 if(index >= jump && index < n)
	 {
		 odata[index] = idata[index] + idata[index - jump];
	 }
 }

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

    cudaMalloc((void**)&dev_odata, n * sizeof(int));
    cudaMalloc((void**)&dev_idata, n * sizeof(int));

    cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
	numBlocks = n / numThreads + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int p=1;
    for(int i=1; i< ilog2ceil(n)+1; ++i)
    {
    	scanStep<<<numBlocks, numThreads>>>(n, p, dev_odata, dev_idata);
    	p <<= 1;
    	cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(stop);
   	cudaEventSynchronize(stop);
   	float milliseconds = 0;
   	cudaEventElapsedTime(&milliseconds, start, stop);
   	if(SHOW_TIMING)
   	   	std::cout<<"Total time in milliseconds : "<<milliseconds<<std::endl;

    cudaMemcpy(odata+1, dev_odata, (n-1) * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_odata);
    cudaFree(dev_idata);
}

}
}
