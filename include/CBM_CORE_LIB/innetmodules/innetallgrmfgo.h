/*
 * innetallgrmfgo.h
 *
 *  Created on: Nov 25, 2013
 *      Author: consciousness
 */

#ifndef INNETALLGRMFGO_H_
#define INNETALLGRMFGO_H_

#include "innet.h"

class InNetAllGRMFGO: virtual public InNet
{
public:
	InNetAllGRMFGO(ConnectivityParams *conParams, ActivityParams *actParams,
			InNetConnectivityState *conState, InNetActivityState *actState,
			int gpuIndStart, int numGPUs);
	virtual ~InNetAllGRMFGO();
	virtual void updateMFtoGOOut();
	virtual void runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN);
	virtual void runSumGRGOOutCUDA(cudaStream_t **sts, int streamN);

protected:
	virtual void initAddGOCUDA();

	ct_uint32_t **totalGRGPU;
private:
	InNetAllGRMFGO();
};


#endif /* INNETALLGRMFGO_H_ */
