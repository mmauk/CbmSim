/*
 * InNet.h
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#ifndef INNET_H_
#define INNET_H_

#include <omp.h>
#include <cuda.h>

#include "stdDefinitions/pstdint.h"
#include "state/innetconnectivitystate.h"
#include "state/innetactivitystate.h"
#include "cuda/kernels.h"

class InNet
{
public:
	InNet();
	InNet(InNetConnectivityState *cs, InNetActivityState *as, int gpuIndStart, int numGPUs);
	~InNet();

	void writeToState();

	const ct_uint8_t* exportAPGO();
	const ct_uint8_t* exportAPMF();
	const ct_uint8_t* exportAPGR();

	const ct_uint32_t* exportSumGRInputGO();
	const float* exportSumGOInputGO();

	// used when initializing mzone
	ct_uint32_t** getApBufGRGPUPointer();
	ct_uint64_t** getHistGRGPUPointer();

	ct_uint32_t** getGRInputGOSumHPointer();

	const float* exportGESumGR();
	const float* exportGISumGR();
	const float* exportgSum_MFGO();
	const float* exportgSum_GRGO();

	void updateMFActivties(const ct_uint8_t *actInMF);
	void calcGOActivities();

	void updateMFtoGROut();
	void updateMFtoGOOut();
	void updateGOtoGROutParameters(float spillFrac);
	void updateGOtoGOOut();
	void resetMFHist(unsigned long t);

	void runGRActivitiesCUDA(cudaStream_t **sts, int streamN);
	void runSumGRGOOutCUDA(cudaStream_t **sts, int streamN);
	void cpyDepAmpMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	void cpyAPMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	
	void cpyDepAmpUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	void cpyAPUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	
	void cpyDepAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	void cpyDynamicAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	void cpyAPGOHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	void runUpdateMFInGRCUDA(cudaStream_t **sts, int streamN);
	void runUpdateMFInGRDepressionCUDA(cudaStream_t **sts, int streamN);
	
	void runUpdateUBCInGRDepressionCUDA(cudaStream_t **sts, int streamN);
	void runUpdateUBCInGRCUDA(cudaStream_t **sts, int streamN);
	
	void runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN);
	void runUpdateGOInGRDepressionCUDA(cudaStream_t **sts, int streamN);
	void runUpdateGOInGRDynamicSpillCUDA(cudaStream_t **sts, int streamN);
	
	void runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN);
	
	void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN,
		ct_uint32_t **grInputGOSumHost);
	void runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, unsigned long t);

protected:

	InNetConnectivityState *cs;
	InNetActivityState *as;

	int gpuIndStart;
	int numGPUs;
	int numGRPerGPU;

	unsigned int calcGRActNumGRPerB;
	unsigned int calcGRActNumBlocks;

	unsigned int updateGRGOOutNumGRPerR;
	unsigned int updateGRGOOutNumGRRows;

	unsigned int sumGRGOOutNumGOPerB;
	unsigned int sumGRGOOutNumBlocks;
	
	unsigned int updateMFInGRNumGRPerB;
	unsigned int updateMFInGRNumBlocks;
	
	unsigned int updateUBCInGRNumGRPerB;
	unsigned int updateUBCInGRNumBlocks;

	unsigned int updateGOInGRNumGRPerB;
	unsigned int updateGOInGRNumBlocks;
	
	unsigned int updateGRHistNumGRPerB;
	unsigned int updateGRHistNumBlocks;

	//UBC
	ct_uint32_t **apUBCH;
	ct_uint32_t **apUBCGPU;

	float **depAmpUBCH;
	float **depAmpUBCGPU;
	float **depAmpUBCGRGPU;

	//mossy fibers
	const ct_uint8_t *apMFOut;

	//gpu related variables
	ct_uint32_t **apMFH;
	ct_uint32_t **apMFGPU;

	float **depAmpMFH;
	float **depAmpMFGPU;
	float **depAmpMFGRGPU;

	int **numMFperGR;
	int **numUBCperGR;
	//end gpu related variables

	//golgi cell variables
	//gpu related variables

	ct_uint32_t **apGOH;
	ct_uint32_t **grInputGOSumH;

	float **depAmpGOH;
	float **dynamicAmpGOH;

	int *counter;

	ct_uint32_t **apGOGPU;
	ct_uint32_t **grInputGOGPU;
	ct_uint32_t **grInputGOSumGPU;

	size_t *grInputGOGPUP;

	float **depAmpGOGRGPU;
	float **depAmpGOGPU;
	float **dynamicAmpGOGPU;
	float **dynamicAmpGOGRGPU;
	//end gpu variables

	ct_uint32_t *sumGRInputGO;

	float *sumInputGOGABASynDepGO;
	float tempGIncGRtoGO;
	//end golgi cell variables

	//granule cell variables
	float **gMFGRT;
	float **gUBCGRT;
	float **gGOGRT;

	ct_uint32_t **pGRDelayfromGRtoGOT;
	ct_uint32_t **pGRfromMFtoGRT;
	ct_uint32_t **pGRfromUBCtoGRT;
	ct_uint32_t **pGRfromGOtoGRT;
	ct_uint32_t **pGRfromGRtoGOT;

	ct_uint32_t apBufGRHistMask;
	//gpu related variables
	//host variables
	ct_uint8_t *outputGRH;
	//end host variables

	float **gEGRGPU;
	size_t *gEGRGPUP;
	float **gEGRSumGPU;
	float **gEDirectGPU;
	float **gESpilloverGPU;
	float **gIDirectGPU;
	float **gISpilloverGPU;
	int   **apMFtoGRGPU;
	int   **apUBCtoGRGPU;
	float **gUBC_EGRGPU;
	size_t *gUBC_EGRGPUP;
	float **gUBC_EGRSumGPU;
	float **gUBC_EDirectGPU;
	float **gUBC_ESpilloverGPU;

	float **gIGRGPU;
	size_t *gIGRGPUP;
	float **gIGRSumGPU;

	ct_uint32_t **apBufGRGPU;
	ct_uint8_t  **outputGRGPU;
	ct_uint32_t **apGRGPU;

	float **threshGRGPU;
	float **vGRGPU;
	float **gLeakGRGPU;
	float **gNMDAGRGPU;
	float **gNMDAIncGRGPU;
	float **gKCaGRGPU;
	ct_uint64_t **historyGRGPU;

	//conduction delays
	ct_uint32_t **delayGOMasksGRGPU;
	size_t *delayGOMasksGRGPUP;

	//connectivity
	int contVar       = 1;
	int contVarOther  = 1;
	float sumGOFR     = 0;
	float sumExScaler = 0;
	int slowCounter   = 0;
	int timeStepPOne;
	int thingCounter  = 0;

	ct_int32_t  **numGOOutPerGRGPU;
	ct_uint32_t **grConGROutGOGPU;
	size_t *grConGROutGOGPUP;
	
	ct_int32_t  **numGOInPerGRGPU;
	ct_uint32_t **grConGOOutGRGPU;
	size_t *grConGOOutGRGPUP;

	ct_int32_t  **numMFInPerGRGPU;
	ct_uint32_t **grConMFOutGRGPU;
	size_t *grConMFOutGRGPUP;
	
	
	ct_int32_t **numUBCInPerGRGPU;
	ct_uint32_t **grConUBCOutGRGPU;
	size_t *grConUBCOutGRGPUP;
	
	//end gpu variables
	//end granule cell variables

	void initCUDA();
	void initMFCUDA();
	void initGRCUDA();
	void initGOCUDA();
	void initSCCUDA();

private:
	template<typename Type>
	cudaError_t getGRGPUData(Type **gpuData, Type *hostData);
};

#endif /* INNET_H_ */

