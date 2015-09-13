#include <cstdio>
#include "cpu.h"
#include <iostream>

namespace StreamCompaction {
namespace CPU {

#define SHOW_TIMING 0

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	odata[0] = 0;

	for(int i=1; i<n; ++i)
	{
		odata[i] = odata[i-1] + idata[i-1];
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	if(SHOW_TIMING)
		std::cout<<"Total time in milliseconds : "<<milliseconds<<std::endl;
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {

	int k = 0;
	for(int i=0; i<n; ++i)
	{
		if(idata[i] != 0)
		{
			++k;
		}
	}

	k = 0;

	for(int i=0; i<n; ++i)
	{
		if(idata[i] != 0)
		{
			odata[k] = idata[i];
			++k;
		}
	}

	return k;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {

	int k=0, i;
	int temp[n], exc_scan[n];

	//Finding temporary array
	for(i=0; i<n; ++i)
	{
		temp[i] = (idata[i] == 0) ? 0 : 1;

		if(temp[i] != 0)
			++k;
	}

	//Running exclusive scan
	scan(n, exc_scan, temp);

	//Scatter

	for(i=0; i<n; ++i)
	{
		if(temp[i] == 1)
		{
			odata[exc_scan[i]] = idata[i];
		}
	}

	return k;
}

}
}
