#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

namespace StreamCompaction {
namespace Efficient {

#define SHOW_TIMING 0

int numBlocks, numThreads = 1024;

void printArray(int n, int * a)
{
	printf("\n");
	for(int i=0; i<n; ++i)
		printf("%d ", a[i]);
	printf("\n");
}

__global__ void upSweep(int n, int p, int *data)
{
	 int index = threadIdx.x + (blockIdx.x * blockDim.x);

	 //If index is less than n and index+1 is divisible by the power of 2
	 if(index < n && ((index) % p) == 0 )
	 {
		 data[index + p - 1] += data[index + p/2 - 1];
	 }
}

__global__ void downSweep(int n, int p, int *data)
{
	 int index = threadIdx.x + (blockIdx.x * blockDim.x);
	 int p2 = p * 2;

	 if(index < n && ((index) % p2) == 0 )
	 {
		 int t = data[index + p - 1];
		 data[index + p - 1] = data[index + p2 - 1];
		 data[index + p2 - 1] += t;
	 }
}

__global__ void updateArray(int index, int value, int *data)
{
	data[index] = value;
}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {

	int p = ilog2ceil(n);
	n = 1;
	n <<= p;
	p = 1;
	numBlocks = n / numThreads + 1;

	int * dev_idata, i;
	cudaMalloc((void**)&dev_idata, n * sizeof(int));
    cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //upSweep
    for(i=0; i< ilog2ceil(n)-1; ++i)
    {
    	p *= 2;
    	upSweep<<<numBlocks, numThreads>>>(n, p, dev_idata);
    }

    //downSweep
    updateArray<<<1,1>>>(n-1, 0, dev_idata);

    i = ilog2ceil(n)-1;
    p = 1;
    p <<= i;

    for(; i>=0; --i)
    {
    	downSweep<<<numBlocks, numThreads>>>(n, p, dev_idata);
    	p >>= 1;
    }

    cudaEventRecord(stop);
   	cudaEventSynchronize(stop);
   	float milliseconds = 0;
   	cudaEventElapsedTime(&milliseconds, start, stop);
   	if(SHOW_TIMING)
   	   	std::cout<<"Total time in milliseconds : "<<milliseconds<<std::endl;

    cudaMemcpy(odata, dev_idata, (n) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_idata);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */

__global__ void setK(int * k, int * data, int index)
{
	(*k) = data[index];
}

int compact(int n, int *odata, const int *idata) {

	int oriN = n;

	int p = ilog2ceil(n);
	n = pow(2, p);

	int *k;

	int *dev_k,
		*dev_idata,
		*dev_odata,
		*dev_temp,
		*dev_scanData;

	cudaMalloc((void**)&dev_idata, n * sizeof(int));
	cudaMalloc((void**)&dev_scanData, n * sizeof(int));
	cudaMalloc((void**)&dev_temp, n * sizeof(int));
	cudaMalloc((void**)&dev_k, sizeof(int));

	k = new int;

	cudaMemset(dev_idata, 0, n * sizeof(int));
	cudaMemcpy(dev_idata, idata, oriN * sizeof(int), cudaMemcpyHostToDevice);

	numBlocks = n / numThreads + 1;

	StreamCompaction::Common::kernMapToBoolean<<<numBlocks, numThreads>>>(n, dev_temp, dev_idata);

	scan(n, dev_scanData, dev_temp);

	setK<<<1,1>>>(dev_k, dev_scanData, n-1);

	cudaMemcpy(k, dev_k, sizeof(int), cudaMemcpyDeviceToHost);

	cudaMalloc((void**)&dev_odata, (*k) * sizeof(int));

	StreamCompaction::Common::kernScatter<<<numBlocks, numThreads>>>(n, dev_odata, dev_idata, dev_temp, dev_scanData);

	cudaMemcpy(odata, dev_odata, (*k) * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_idata);
	cudaFree(dev_temp);
	cudaFree(dev_scanData);
	cudaFree(dev_k);

	return (*k);

}

}
}
