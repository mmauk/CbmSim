/*
 * mzone.h
 *
 *  Created on: Jun 13, 2011
 *      Author: consciousness
 */

#ifndef MZONE_H_
#define MZONE_H_

#ifdef INTELCC
#include <mathimf.h>
#else //otherwise use standard math library
#include <math.h>
#endif

#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "stdDefinitions/pstdint.h"
#include "randGenerators/sfmt.h"
#include "memoryMgmt/dynamic2darray.h"
#include "params/connectivityparams.h"
#include "params/activityparams.h"
#include "state/mzoneconnectivitystate.h"
#include "state/mzoneactivitystate.h"
#include "cuda/kernels.h"

class MZone
{
public:
	MZone();
	MZone(ConnectivityParams *cp, ActivityParams *ap, MZoneConnectivityState *cs,
			MZoneActivityState *as, int randSeed, ct_uint32_t **actBufGRGPU,
			ct_uint32_t **delayMaskGRGPU, ct_uint64_t **histGRGPU, int gpuIndStart, int numGPUs);
	~MZone();

	void writeToState();
	void cpyPFPCSynWCUDA();

	void setErrDrive(float errDriveRelative);
	void updateMFActivities(const ct_uint8_t *actMF);
	void updateTrueMFs(bool *trueMF);
	void updateSCActivities(const ct_uint8_t *actSC);
	void updatePFBCSum(const ct_uint32_t *pfBCSum);

	void calcPCActivities();
	void calcBCActivities(ct_uint32_t **pfInput);
	void calcIOActivities();
	void calcNCActivities();

	void updatePCOut();
	void updateBCPCOut();
	void updateSCPCOut();
	void updateIOOut();
	void updateNCOut();
	void updateMFNCOut();
	//void updateMFNCSyn(const std::unique_ptr<ct_uint8_t[]> histMF, unsigned long t);

	void runPFPCOutCUDA(cudaStream_t **sts, int streamN);
	void runPFPCSumCUDA(cudaStream_t **sts, int streamN);
	void cpyPFPCSumCUDA(cudaStream_t **sts, int streamN);
	void runPFPCPlastCUDA(cudaStream_t **sts, int streamN, unsigned long t);

	void setGRPCPlastSteps(float ltdStep, float ltpStep);
	void resetGRPCPlastSteps();

	const ct_uint8_t* exportAPNC();
	const ct_uint8_t* exportAPBC();
	const ct_uint8_t* exportAPPC();
	const ct_uint8_t* exportAPIO();

	const float* exportVmBC();
	const float* exportVmPC();
	const float* exportVmNC();
	const float* exportVmIO();
	const float* exportgBCPC();
	const float* exportgPFPC();
	const float* exportPFPCWeights();

	const ct_uint32_t* exportAPBufBC();
	const ct_uint32_t* exportAPBufPC();
	const ct_uint8_t* exportAPBufIO();
	const ct_uint32_t* exportAPBufNC();

private:
	ConnectivityParams *cp;
	ActivityParams *ap;
	MZoneConnectivityState *cs;
	MZoneActivityState *as;

	CRandomSFMT0 *randGen;

	int gpuIndStart;
	int numGPUs;
	int numGRPerGPU;

	unsigned int updatePFPCNumGRPerB;
	unsigned int updatePFPCNumBlocks;

	unsigned int updatePFPCSynWNumGRPerB;
	unsigned int updatePFPCSynWNumBlocks;

	//mossy fiber variables
	const ct_uint8_t *apMFInput;
	//const ct_uint8_t *histMFInput;
	bool *isTrueMF;

	//stellate cell variables
	const ct_uint8_t *apSCInput;

	//basket cell variables
	const ct_uint32_t *sumPFBCInput;

	//purkinje cell variables
	float **pfSynWeightPCGPU;
	float *pfSynWeightPCLinear;
	float **inputPFPCGPU;
	size_t *inputPFPCGPUPitch;
	float **inputSumPFPCMZGPU;
	float *inputSumPFPCMZH;

	ct_uint32_t **apBufGRGPU;
	ct_uint32_t **delayBCPCSCMaskGRGPU;
	ct_uint64_t **historyGRGPU;

	//IO cell variables
	float *pfPCPlastStepIO;
	float tempGRPCLTDStep;
	float tempGRPCLTPStep;

	void initCUDA();
	void testReduction();
};

#endif /* MZONE_H_ */

