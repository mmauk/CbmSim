/*
 * mzoneactivitystate.h
 *
 *  Created on: Nov 26, 2012
 *      Author: consciousness
 */

#ifndef MZONEACTIVITYSTATE_H_
#define MZONEACTIVITYSTATE_H_

#include <fstream>
#include <iostream>
#include <vector>

#include <memoryMgmt/dynamic2darray.h>
#include <memoryMgmt/arrayinitalize.h>
#include <memoryMgmt/arrayvalidate.h>
#include <fileIO/rawbytesrw.h>
#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>

#include "params/connectivityparams.h"
#include "params/activityparams.h"

#include "interfaces/imzoneactstate.h"

class MZoneActivityState : public virtual IMZoneActState
{
public:
	MZoneActivityState(ConnectivityParams *conParams, ActivityParams *actParams, int randSeed);
	MZoneActivityState(ConnectivityParams *conParams, ActivityParams *actParams, std::fstream &infile);
	MZoneActivityState(const MZoneActivityState &state);

	virtual ~MZoneActivityState();

	void writeState(std::fstream &outfile);

	bool equivalent(const MZoneActivityState &compState);

	std::vector<float> getGRPCSynWeightLinear();
//	std::vector<std::vector<float> > getGRPCSynWeight();
	void resetGRPCSynWeight();

	ConnectivityParams *cp;
	ActivityParams *ap;

	//basket cells
	ct_uint8_t *apBC;
	ct_uint32_t *apBufBC;
	ct_uint32_t *inputPCBC;
	float *gPFBC;
	float *gPCBC;
	float *threshBC;
	float *vBC;

	//purkinje cells
	ct_uint8_t *apPC;
	ct_uint32_t *apBufPC;
	ct_uint32_t *inputBCPC;
	ct_uint8_t **inputSCPC;
	float **pfSynWeightPC;
	float *inputSumPFPC;
	float *gPFPC;
	float *gBCPC;
	float **gSCPC;
	float *vPC;
	float *threshPC;
	ct_uint32_t *histPCPopAct;
	ct_uint32_t histPCPopActSum;
	ct_uint32_t histPCPopActCurBinN;
	ct_uint32_t pcPopAct;

	//inferior olivary cells
	ct_uint8_t *apIO;
	ct_uint32_t *apBufIO;
	ct_uint8_t **inputNCIO;
	float errDrive;
	float **gNCIO;
	float *threshIO;
	float *vIO;
	float *vCoupleIO;
	ct_int32_t *pfPCPlastTimerIO;

	//nucleus cells
	ct_uint8_t noLTPMFNC;
	ct_uint8_t noLTDMFNC;
	ct_uint8_t *apNC;
	ct_uint32_t *apBufNC;
	ct_uint8_t **inputPCNC;
	ct_uint8_t **inputMFNC;
	float **gPCNC;
	float **gPCScaleNC;
	float **mfSynWeightNC;
	float **gmaxNMDAofMFtoNC;
	float **gmaxAMPAofMFtoNC;
	float **gMFNMDANC;
	float **gMFAMPANC;
	float *threshNC;
	float *vNC;
	float *synIOPReleaseNC;

private:
	MZoneActivityState();

	void allocateMemory();
	void stateRW(bool read, std::fstream &file);
	void initializeVals(int randSeed);
};


#endif /* MZONEACTIVITYSTATE_H_ */
