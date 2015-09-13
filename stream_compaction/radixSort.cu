#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radixSort.h"
#include "efficient.h"

#define TEST 0
int numBlocks, numThread = 1024;

namespace RadixSort {

void printArray(int size, int * a)
{
	printf("\n");
	for(int i=0; i<size; ++i)
	{
		printf("%d ", a[i]);
	}
	printf("\n");
}
	__global__ void createEArray(int n, int *e, int* b)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if(index < n)
		{
			e[index] = (b[index]==0) ? 1 : 0;
		}
	}

	__global__ void scan(int n, int i, int *odata, int *idata)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if(index < n)
		{
			odata[index] = (idata[index] & i) ? 1 : 0;
		}
	}

	__global__ void getTotalFalse(int index, int * totalFalse, int *dev_f, int *dev_e)
	{
		(*totalFalse) = dev_f[index] + dev_e[index];
	}

	__global__ void createTArray(int n, int *t, int*f, int *totalFalse)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if(index < n)
		{
			t[index] = index - f[index] + (*totalFalse);
		}
	}

	__global__ void createDArray(int n, int *d, int *b, int *t, int *f)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if(index < n)
		{
			d[index] = b[index] ? t[index] : f[index];
		}

	}

	__global__ void scatter(int n, int *odata, int *idata, int *d)
	{
		int index = threadIdx.x + (blockIdx.x * blockDim.x);

		if(index < n)
		{
			odata[d[index]] = idata[index];
		}

	}

	void split(int n, int i, int *dev_odata, int* dev_idata)
	{
	  	int *dev_b,
    		*dev_e,
    		*dev_f,
    		*dev_t,
    		*dev_d;

	  	int hst_temp[n];

		//Create b
	  	cudaMalloc((void**)&dev_b, n * sizeof(int));
    	scan<<<numBlocks, numThread>>>(n, i, dev_b, dev_idata);
    	if(TEST)
    	{
    		cudaMemcpy(hst_temp, dev_b, n*sizeof(int), cudaMemcpyDeviceToHost);
    		printArray(n, hst_temp);
    	}

    	//Create e
	   	cudaMalloc((void**)&dev_e, n * sizeof(int));
		createEArray<<<numBlocks, numThread>>>(n, dev_e, dev_b);
		if(TEST)
		{
			cudaMemcpy(hst_temp, dev_e, n*sizeof(int), cudaMemcpyDeviceToHost);
			printArray(n, hst_temp);
		}

		//Create f by using efficient scan
		cudaMalloc((void**)&dev_f, n * sizeof(int));
		StreamCompaction::Efficient::scan(n, dev_f, dev_e);
		if(TEST)
		{
			cudaMemcpy(hst_temp, dev_f, n*sizeof(int), cudaMemcpyDeviceToHost);
			printArray(n, hst_temp);
		}

		//Finding total false
		int *dev_totalFalse;
		cudaMalloc((void**)&dev_totalFalse, sizeof(int));
		getTotalFalse<<<1, 1>>>(n-1, dev_totalFalse, dev_f, dev_e);
		if(TEST)
		{
			cudaMemcpy(hst_temp, dev_totalFalse, sizeof(int), cudaMemcpyDeviceToHost);
			printf("\n%d %d\n", hst_temp[0], n-1);
		}

		//Create t
		cudaMalloc((void**)&dev_t, n * sizeof(int));
		createTArray<<<numBlocks, numThread>>>(n, dev_t, dev_f, dev_totalFalse);
		if(TEST)
		{
			cudaMemcpy(hst_temp, dev_t, n*sizeof(int), cudaMemcpyDeviceToHost);
			printArray(n, hst_temp);
		}

		//Create d
		cudaMalloc((void**)&dev_d, n * sizeof(int));
		createDArray<<<numBlocks, numThread>>>(n, dev_d, dev_b, dev_t, dev_f);
		if(TEST)
		{
			cudaMemcpy(hst_temp, dev_d, n*sizeof(int), cudaMemcpyDeviceToHost);
			printArray(n, hst_temp);
		}

		//Shuffle
		scatter<<<numBlocks, numThread>>>(n, dev_odata, dev_idata, dev_d);
		if(TEST)
		{
			cudaMemcpy(hst_temp, dev_odata, n*sizeof(int), cudaMemcpyDeviceToHost);
			printArray(n, hst_temp);
		}

		cudaFree(dev_b);
		cudaFree(dev_e);
		cudaFree(dev_f);
		cudaFree(dev_t);
		cudaFree(dev_d);
		cudaFree(dev_totalFalse);
	}

    void sort(int n, int maxValue, int *odata, const int *idata)
    {
    	int i = 1,
    	    *dev_idata,
    	    *dev_odata;

    	numBlocks = n / numThread + 1;

    	cudaMalloc((void**)&dev_odata, n * sizeof(int));
       	cudaMalloc((void**)&dev_idata, n * sizeof(int));

       	cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

    	while(i <= maxValue)
    	{
    		split(n, i, dev_odata, dev_idata);

    		cudaMemcpy(dev_idata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);

    		i<<=1;
    	}

    	cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

    	cudaFree(dev_idata);
    	cudaFree(dev_odata);

    }
}
