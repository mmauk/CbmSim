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

#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>
#include <memoryMgmt/dynamic2darray.h>

#include <params/connectivityparams.h>
#include <params/activityparams.h>
#include <state/mzoneconnectivitystate.h>
#include <state/mzoneactivitystate.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include "cuda/kernels.h"

#include "interface/mzoneinterface.h"

class MZone : virtual public MZoneInterface
{
public:
	MZone(ConnectivityParams &cp, ActivityParams *ap,
			MZoneConnectivityState *conState, MZoneActivityState *actState, int randSeed,
			ct_uint32_t **actBufGRGPU, ct_uint32_t **delayMaskGRGPU, ct_uint64_t **histGRGPU,
			int gpuIndStart, int numGPUs);

	~MZone();

	void writeToState(ConnectivityParams &cp);
	void cpyPFPCSynWCUDA(ConnectivityParams &cp);

	void setErrDrive(ActivityParams *ap, float errDriveRelative);
	void updateMFActivities(const ct_uint8_t *actMF);
	void updateTrueMFs(bool *trueMF);
	void updateSCActivities(const ct_uint8_t *actSC);
	void updatePFBCSum(const ct_uint32_t *pfBCSum);

	void calcPCActivities(ConnectivityParams &cp, ActivityParams *ap);
	void calcBCActivities(ConnectivityParams &cp, ActivityParams *ap, ct_uint32_t **pfInput);
	void calcIOActivities(ConnectivityParams &cp, ActivityParams *ap);
	void calcNCActivities(ConnectivityParams &cp, ActivityParams *ap);

	void updatePCOut(ConnectivityParams &cp);
	void updateBCPCOut(ConnectivityParams &cp);
	void updateSCPCOut(ConnectivityParams &cp);
	void updateIOOut(ConnectivityParams &cp, ActivityParams *ap);
	void updateNCOut(ConnectivityParams &cp, ActivityParams *ap);
	void updateMFNCOut(ConnectivityParams &cp);
	void updateMFNCSyn(ConnectivityParams &cp, ActivityParams *ap,
		const ct_uint8_t *histMF, unsigned long t);

	void runPFPCOutCUDA(ConnectivityParams &cp, cudaStream_t **sts, int streamN);
	void runPFPCSumCUDA(ConnectivityParams &cp, cudaStream_t **sts, int streamN);
	void cpyPFPCSumCUDA(ConnectivityParams &cp, cudaStream_t **sts, int streamN);
	void runPFPCPlastCUDA(ConnectivityParams &cp, ActivityParams *ap,
		cudaStream_t **sts, int streamN, unsigned long t);

	void setGRPCPlastSteps(float ltdStep, float ltpStep);
	void resetGRPCPlastSteps(ActivityParams *ap);

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
	const float* exportPFPCWeights(ConnectivityParams &cp);

	const ct_uint32_t* exportAPBufBC();
	const ct_uint32_t* exportAPBufPC();
	const ct_uint32_t* exportAPBufIO();
	const ct_uint32_t* exportAPBufNC();

private:
	MZone();

	//ConnectivityParams *cp;
	//ActivityParams *ap;
	MZoneConnectivityState *cs;
	MZoneActivityState *as;

	void initCUDA(ConnectivityParams &cp);

	void testReduction(ConnectivityParams &cp);

	//global variables
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
	const ct_uint8_t *histMFInput;
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
};

#endif /* MZONE_H_ */
