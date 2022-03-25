/*
 * InNet.h
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#ifndef INNET_H_
#define INNET_H_

#ifdef INTELCC
#include <mathimf.h>
#else //otherwise use standard math library
#include <math.h>
#endif

#include <vector>
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>
#include <memoryMgmt/dynamic2darray.h>

#include <params/connectivityparams.h>
#include <params/activityparams.h>
#include <state/innetconnectivitystate.h>
#include <state/innetactivitystate.h>

#include "cuda/kernels.h"

#include "interface/innetinterface.h"

class InNet : virtual public InNetInterface
{
public:
	InNet(ConnectivityParams *conParams, ActivityParams *actParams,
			InNetConnectivityState *conState, InNetActivityState *actState,
			int gpuIndStart, int numGPUs);
	virtual ~InNet();

	void writeToState();
	void printVGRGPU();
	void getnumGPUs();
	void grStim(int startGR, int numGR);
	
	unsigned int totalGRIn;
	
	float goalRate 		  = 7.0;
	float avgISI 		  = 1000.0 / goalRate;
	float weightScalerEX  = 0.0000005;
	float weightScalerINH = 0.0000005;
	
	float exDecrease = weightScalerEX * avgISI;
	float exIncrease = weightScalerEX;
	
	float inhDecrease = weightScalerINH * avgISI;
	float inhIncrease = weightScalerINH;
	
	float **goGRGOScaler;	
	
	
	virtual void setGIncGRtoGO(float inc);
	virtual void resetGIncGRtoGO();

	virtual const ct_uint8_t* exportAPMF();
	virtual const ct_uint8_t* exportAPSC();
	virtual const ct_uint8_t* exportAPGO();
	virtual const ct_uint8_t* exportAPGR();
	virtual const ct_uint8_t* exportAPUBC();

	virtual const ct_uint8_t*  exportHistMF();
	virtual const ct_uint32_t* exportAPBufMF();
	virtual const ct_uint32_t* exportAPBufGR();
	virtual const ct_uint32_t* exportAPBufGO();
	virtual const ct_uint32_t* exportAPBufSC();
	
	virtual const float* exportvSum_GOGO();
	virtual const float* exportvSum_GRGO();
	virtual const float* exportvSum_MFGO();

	virtual const float* exportVmGR();
	virtual const float* exportVmGO();
	virtual const float* exportExGOInput();
	virtual const float* exportInhGOInput();
	virtual const float* exportVGOGOcouple();
	virtual const float* exportgSum_MFGO();
	virtual const float* exportgSum_GOGO();
	virtual const float* exportgSum_GRGO();
	
	virtual const float* exportVmSC();
	virtual const float* exportGESumGR();
	virtual const float* exportGUBCESumGR();
	virtual const float* exportDepSumUBCGR();
	
	virtual const float* exportDepSumGOGR();
	virtual const float* exportDynamicSpillSumGOGR();
	virtual const float* exportgNMDAGR();
	virtual const int*   exportAPfromMFtoGR();
	virtual const float* exportGISumGR();

	virtual const ct_uint32_t* exportSumGRInputGO();
	virtual const float* 	   exportSumGOInputGO();
	virtual const float* 	   exportGOOutSynScaleGOGO();
	virtual const float* 	   exportgGOGO();

	virtual const ct_uint32_t* exportPFBCSum();

	virtual ct_uint32_t** getApBufGRGPUPointer();
	virtual ct_uint32_t** getDelayBCPCSCMaskGPUPointer();
	virtual ct_uint64_t** getHistGRGPUPointer();

	virtual ct_uint32_t** getGRInputGOSumHPointer();
	virtual ct_uint32_t** getGRInputBCSumHPointer();

	virtual void updateMFActivties(const ct_uint8_t *actInMF);
	virtual void calcGOActivities(float goMin, int simNum, float GRGO, float MFGO, float GOGR, float gogoW);
	virtual void calcSCActivities();
	virtual void calcUBCActivities();
	virtual void updateMFtoUBCOut();
	virtual void updateGOtoUBCOut();
	virtual void updateUBCtoUBCOut();
	virtual void updateUBCtoGOOut();
	virtual void updateUBCtoGROut();

	virtual void updateMFtoGROut();	
	virtual void updateGOtoGROutParameters(float GOGR, float spillFrac);	
	virtual void updateMFtoGOOut();
	virtual void updateGOtoGOOut();
	virtual void resetMFHist(unsigned long t);

	virtual void runGRActivitiesCUDA(cudaStream_t **sts, int streamN);
	virtual void runSumPFBCCUDA(cudaStream_t **sts, int streamN);
	virtual void runSumPFSCCUDA(cudaStream_t **sts, int streamN);
	virtual void runSumGRGOOutCUDA(cudaStream_t **sts, int streamN);
	virtual void runSumGRBCOutCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyDepAmpMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyAPMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	
	virtual void cpyDepAmpUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyAPUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	
	virtual void cpyDepAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN);	
	virtual void cpyDynamicAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN);	
	virtual void cpyAPGOHosttoGPUCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdateMFInGRDepressionCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdateMFInGRCUDA(cudaStream_t **sts, int streamN);
	
	virtual void runUpdateUBCInGRDepressionCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdateUBCInGRCUDA(cudaStream_t **sts, int streamN);
		
	virtual void runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN, float GOGR);
	virtual void runUpdateGOInGRDepressionCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdateGOInGRDynamicSpillCUDA(cudaStream_t **sts, int streamN);	
	virtual void runUpdatePFBCSCOutCUDA(cudaStream_t **sts, int streamN);
	
	virtual void runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN);
	virtual void runUpdateGROutBCCUDA(cudaStream_t **sts, int streamN);
	
	virtual void cpyPFBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyPFSCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyGRBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
	virtual void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN, ct_uint32_t **grInputGOSumHost);
	virtual void cpyGRBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN, ct_uint32_t **grInputBCSumHost);
	virtual void runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, unsigned long t);

protected:
	virtual void initCUDA();
	virtual void initUBCCUDA();
	virtual void initMFCUDA();
	virtual void initGRCUDA();
	virtual void initGOCUDA();
	virtual void initBCCUDA();
	virtual void initSCCUDA();

	ConnectivityParams *cp;
	ActivityParams 	   *ap;

	InNetConnectivityState *cs;
	InNetActivityState 	   *as;

	CRandomSFMT0 *randGen;

	int gpuIndStart;
	int numGPUs;
	int numGRPerGPU;

	unsigned int calcGRActNumGRPerB;
	unsigned int calcGRActNumBlocks;

	unsigned int updateGRGOOutNumGRPerR;
	unsigned int updateGRGOOutNumGRRows;

	unsigned int sumGRGOOutNumGOPerB;
	unsigned int sumGRGOOutNumBlocks;
	
	unsigned int sumGRBCOutNumBCPerB;
	unsigned int sumGRBCOutNumBlocks;

	unsigned int updateMFInGRNumGRPerB;
	unsigned int updateMFInGRNumBlocks;
	
	unsigned int updateUBCInGRNumGRPerB;
	unsigned int updateUBCInGRNumBlocks;

	unsigned int updateGOInGRNumGRPerB;
	unsigned int updateGOInGRNumBlocks;
	
	unsigned int updateGRBCOutNumGRPerR;
	unsigned int updateGRBCOutNumGRRows;

	unsigned int updatePFBCSCNumGRPerB;
	unsigned int updatePFBCSCNumBlocks;

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
	//
	//gpu related variables
	ct_uint32_t **apMFH;
	ct_uint32_t **apMFGPU;

	float **depAmpMFH;
	float **depAmpMFGPU;
	float **depAmpMFGRGPU;

	int **numMFperGR;
	int **numUBCperGR;
	//
	//end gpu related variables

	//---------golgi cell variables
	//gpu related variables
	//GPU parameters

	ct_uint32_t **apGOH;
	ct_uint32_t **grInputGOSumH;

	float **depAmpGOH;
	float **dynamicAmpGOH;

	int *counter;

	size_t *grInputBCGPUP;

	ct_uint32_t **grInputBCGPU;
	ct_uint32_t **pGRDelayfromPFtoBCT;
	ct_uint32_t **pGRfromPFtoBCT;
	ct_uint32_t **grInputBCSumGPU;
	ct_uint32_t **grInputBCSumH;


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
	//---------end golgi cell variables

	//---------granule cell variables
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
	size_t 		*delayGOMasksGRGPUP;
	ct_uint32_t **delayBCPCSCMaskGRGPU;
	ct_uint32_t **delayBCMasksGRGPU;
	size_t 		*delayBCMasksGRGPUP;

	//connectivity
	float *plasScalerEx;
	float *plasScalerInh;
	
	float **goExScaler;
	float **goInhScaler;
	float **goFRArray;
	int   counterGOweight;	
	
	int contVar 	  = 1;
	int contVarOther  = 1;
	float sumGOFR	  = 0;
	float sumExScaler = 0;
	int slowCounter	  = 0;
	int timeStepPOne;
	int thingCounter  = 0;

	ct_int32_t  **numGOOutPerGRGPU;
	ct_uint32_t **grConGROutGOGPU;
	size_t 		*grConGROutGOGPUP;
	
	ct_int32_t  **numBCOutPerGRGPU;
	ct_uint32_t **grConGROutBCGPU;
	size_t 		*grConGROutBCGPUP;

	ct_int32_t  **numGOInPerGRGPU;
	ct_uint32_t **grConGOOutGRGPU;
	size_t 		*grConGOOutGRGPUP;

	ct_int32_t  **numMFInPerGRGPU;
	ct_uint32_t **grConMFOutGRGPU;
	size_t 		*grConMFOutGRGPUP;
	
	
	ct_int32_t 	**numUBCInPerGRGPU;	
	ct_uint32_t **grConUBCOutGRGPU;
	size_t 		*grConUBCOutGRGPUP;
	
	//end gpu variables

	//---------end granule cell variables

	//--------stellate cell variables

	//gpu related variables
	//host variables
	ct_uint32_t *inputSumPFSCH;
	//end host variables

	ct_uint32_t **inputPFSCGPU;
	size_t 		*inputPFSCGPUP;
	ct_uint32_t **inputSumPFSCGPU;
	//end gpu related variables

	//------------ end stellate cell variables

	//-----------basket cell variables
	//gpu related variables
	//host variables
	ct_uint32_t *inputSumPFBCH;

	//device variables
	ct_uint32_t **inputPFBCGPU;
	size_t 		*inputPFBCGPUP;
	ct_uint32_t **inputSumPFBCGPU;
	//end gpu related variables
	//-----------end basket cell variables

private:
	InNet();
	
	template<typename Type> cudaError_t getGRGPUData(Type **gpuData, Type *hostData);

};

#endif /* INNET_H_ */
