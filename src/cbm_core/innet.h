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

#include <cstdint>
#include "innetconnectivitystate.h"
#include "innetactivitystate.h"
#include "kernels.h"

class InNet
{
public:
	InNet();
	InNet(InNetConnectivityState *cs, InNetActivityState *as, int gpuIndStart, int numGPUs);
	~InNet();

	void writeToState();

	const uint8_t* exportAPGO();
	const uint8_t* exportAPMF();
	const uint8_t* exportHistMF();
	const uint8_t* exportAPGR();

	uint32_t** getApBufGRGPUPointer();
	uint64_t** getHistGRGPUPointer();
	uint32_t** getGRInputGOSumHPointer();

	const float* exportGESumGR();
	const float* exportGISumGR();
	const float* exportgSum_MFGO();
	const float* exportgSum_GRGO();

	void updateMFActivties(const uint8_t *actInMF);
	void runGOActivitiesCUDA(cudaStream_t **sts, int streamN);
	void cpyApGODevicetoHostCUDA(cudaStream_t **sts, int streamN);
	void cpyApGOHosttoDeviceCUDA(cudaStream_t **sts, int streamN);
	void cpyVGODevicetoHostCUDA(cudaStream_t **sts, int streamN);
	void cpyVGOHosttoDeviceCUDA(cudaStream_t **sts, int streamN);

	void updateMFtoGROut();
	void resetMFHist(uint32_t t);

	void runGRActivitiesCUDA(cudaStream_t **sts, int streamN);
	void cpyDepAmpMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	void cpyAPMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	
	void runUpdateMFInGRCUDA(cudaStream_t **sts, int streamN);
	void runUpdateMFInGRDepressionCUDA(cudaStream_t **sts, int streamN);
	void runUpdateMFInGOCUDA(cudaStream_t **sts, int streamN);
	void runUpdateGOInGOCUDA(cudaStream_t **sts, int streamN);
	void runUpdateGOCoupInGOCUDA(cudaStream_t **sts, int streamN);
	void runUpdateGOOutGRDynamicSpillCUDA(cudaStream_t **sts, int streamN, float spillFrac);
	void cpyGOGRDynamicSpillGPUtoHostCUDA(cudaStream_t **sts, int streamN);

	void cpyGOGRDynamicSpillHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	void runUpdateGOInGRDynamicSpillCUDA(cudaStream_t **sts, int streamN);
	void runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN);

	void runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, uint32_t t);
	void runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN);
	void runSumGRGOOutCUDA(cudaStream_t **sts, int streamN);
	void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	void runSumReductionGRGOInputHost();
	void cpyGRGOSumHosttoGPUCUDA(cudaStream_t **sts, int streamN);

protected:

	InNetConnectivityState *cs;
	InNetActivityState *as;

	int gpuIndStart;
	int numGPUs;
	int numGRPerGPU;

	uint32_t calcGRActNumGRPerB;
	uint32_t calcGRActNumBlocks;

	uint32_t numGOPerGPU;
	uint32_t calcGOActNumBlocks; 
	uint32_t calcGOActNumGOPerB; 

	uint32_t updateGRGOOutNumGRPerR;
	uint32_t updateGRGOOutNumGRRows;

	uint32_t sumGRGOOutNumGOPerB;
	uint32_t sumGRGOOutNumBlocks;

	uint32_t updateMFInGOnumGOPerB; // -> named because one GO per thread
	uint32_t updateMFInGONumBlocks;

	uint32_t updateGOInGONumGOPerB;
	uint32_t updateGOInGONumBlocks;

	uint32_t updateMFInGRNumGRPerB;
	uint32_t updateMFInGRNumBlocks;
	
	uint32_t updateGOInGRNumGRPerB;
	uint32_t updateGOInGRNumBlocks;
	
	uint32_t updateGRHistNumGRPerB;
	uint32_t updateGRHistNumBlocks;

	//mossy fibers
	const uint8_t *apMFOut;

	//gpu related variables
	uint32_t **apMFH;
	uint32_t **apMFGPU;

	uint32_t **pGOfromMFtoGOT; // dims: (maxMFperGO, num_go)

	int32_t  **numMFInPerGOGPU;
	uint32_t **goConMFOutGOGPU; // TODO: pretty sure these are the input arrs, change that name :weird_champ:
	size_t *goConMFOutGOGPUP;

	uint32_t **pGOGABAInGOGOT;

	int32_t **numGOInPerGOGPU;
	uint32_t **goConGOOutGOGPU; // TODO: pretty sure these are the input arrs, change that name :weird_champ:
	size_t *goConGOOutGOGPUP;

	uint32_t **pGOCoupInGOGOT;
	float **pGOCoupInGOGOCCoeffT;
	int32_t **numGOCoupInPerGOGPU;
	uint32_t **goConCoupGOInGOGPU;
	size_t *goConCoupGOInGOGPUP;
	float **goCoupCoeffInGOGPU;
	size_t *goCoupCoeffInGOGPUP;

	float **depAmpMFH;
	float **depAmpMFGPU;
	float **depAmpMFGRGPU;

	int **numMFperGR;
	//end gpu related variables

	//golgi cell variables
	//gpu related variables

	uint32_t *apGOH;
	float *vGOH;
	float *dynamicAmpGOH;
	uint32_t **grInputGOSumH; // per GPU sum
	uint32_t *grInputGOSumHost; // collected sum on host across gpus

	// new vars

	uint32_t **goIsiCounterGPU;

	float **vGOOutGPU;
	float **vGOInGPU;
	float **vCoupleGOGOGPU;
	float **threshGOGPU;
	uint32_t **apBufGOGPU;
	uint32_t **inputMFGOGPU;
	uint32_t **inputGOGOGPU;
	uint32_t **inputGRGOGPU;
	float **gSumMFGOGPU;
	float **gSumGOGOGPU;
	float **synWScalerGOGOGPU;
	float **synWScalerGRGOGPU;
	float **gNMDAMFGOGPU;
	float **gNMDAIncMFGOGPU;
	float **gGRGOGPU;
	float **gGRGO_NMDAGPU;

	// end new vars

	uint32_t **apGOOutGPU; // output spikes from GO. dims (numGOPerGPU, 1)
	uint32_t **apGOInGPU; // input to each GR (and other GO). dims (num_go, 1)
	uint32_t **grInputGOGPU;
	size_t *grInputGOGPUP;
	uint32_t **grInputGOSumGPU;

	float **dynamicAmpGOOutGPU; // output from each GO. dims (numGOPerGPU, 1)
	float **dynamicAmpGOInGPU; // input to each GR from GO. dims (num_go, 1)
	float **dynamicAmpGOGRGPU;
	//end gpu variables

	//end golgi cell variables

	//granule cell variables
	float **gMFGRT;
	float **gGOGRT;

	uint32_t **pGRDelayfromGRtoGOT;
	uint32_t **pGRfromMFtoGRT;
	uint32_t **pGRfromGOtoGRT;
	uint32_t **pGRfromGRtoGOT;

	uint32_t apBufGRHistMask;
	//gpu related variables
	//host variables
	uint8_t *outputGRH;
	//end host variables

	float **gEGRGPU;
	size_t *gEGRGPUP;
	float **gEGRSumGPU;
	float **gEDirectGPU;
	float **gESpilloverGPU;
	float **gIDirectGPU;
	float **gISpilloverGPU;
	int   **apMFtoGRGPU;

	float **gIGRGPU;
	size_t *gIGRGPUP;
	float **gIGRSumGPU;

	uint32_t **apBufGRGPU;
	uint8_t  **outputGRGPU;
	uint32_t **apGRGPU;

	float **threshGRGPU;
	float **vGRGPU;
	float **gLeakGRGPU;
	float **gNMDAGRGPU;
	float **gNMDAIncGRGPU;
	float **gKCaGRGPU;
	uint64_t **historyGRGPU;

	//conduction delays
	uint32_t **delayGOMasksGRGPU;
	size_t *delayGOMasksGRGPUP;

	//connectivity
	int32_t  **numGOOutPerGRGPU;
	uint32_t **grConGROutGOGPU;
	size_t *grConGROutGOGPUP;
	
	int32_t  **numGOInPerGRGPU;
	uint32_t **grConGOOutGRGPU;
	size_t *grConGOOutGRGPUP;

	int32_t  **numMFInPerGRGPU;
	uint32_t **grConMFOutGRGPU;
	size_t *grConMFOutGRGPUP;
	
	//end gpu variables
	//end granule cell variables

	void initCUDA();
	void initMFCUDA();
	void initGRCUDA();
	void initGOCUDA();
	void initSCCUDA();

private:
	template<typename Type>
	cudaError_t getGPUData(Type **gpuData, Type *hostData);
};

#endif /* INNET_H_ */

