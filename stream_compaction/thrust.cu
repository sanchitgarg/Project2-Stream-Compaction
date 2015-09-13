#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

#define SHOW_TIMING 0

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

	thrust::host_vector<int> thrustHst_idata(idata, idata+n);

	thrust::device_vector<int> thrustDev_idata(thrustHst_idata);
	thrust::device_vector<int> thrustDev_odata(n);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    thrust::exclusive_scan(thrustDev_idata.begin(), thrustDev_idata.end(), thrustDev_odata.begin());

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
   	float milliseconds = 0;
   	cudaEventElapsedTime(&milliseconds, start, stop);
   	if(SHOW_TIMING)
	   	std::cout<<"Total time in milliseconds : "<<milliseconds<<std::endl;

	thrust::copy(thrustDev_odata.begin(), thrustDev_odata.end(), odata);
}

}
}
