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
#include <algorithm> /* std::fill */
#include <fileIO/rawbytesrw.h>
#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>
#include "params/connectivityparams.h"
#include "params/activityparams.h"
#include "interfaces/imzoneactstate.h"

class MZoneActivityState : public virtual IMZoneActState
{
public:
	MZoneActivityState();
	MZoneActivityState(ActivityParams &ap, int randSeed);
	MZoneActivityState(ActivityParams &ap, std::fstream &infile);

	virtual ~MZoneActivityState();

	void writeState(ActivityParams &ap, std::fstream &outfile);

	//virtual std::vector<float> getGRPCSynWeightLinear();
	//virtual void resetGRPCSynWeight(float initSynWofGRtoPC);

	//basket cells
	ct_uint8_t apBC[NUM_BC] = {0};
	ct_uint32_t apBufBC[NUM_BC] = {0};
	ct_uint32_t inputPCBC[NUM_BC] = {0};
	float gPFBC[NUM_BC] = {0};
	float gPCBC[NUM_BC] = {0};
	float threshBC[NUM_BC] = {0};
	float vBC[NUM_BC] = {0};

	//purkinje cells
	ct_uint8_t apPC[NUM_PC] = {0};
	ct_uint32_t apBufPC[NUM_PC] = {0};
	ct_uint32_t inputBCPC[NUM_PC] = {0};
	ct_uint8_t inputSCPC[NUM_PC][NUM_P_PC_FROM_SC_TO_PC] = {0};
	float pfSynWeightPC[NUM_PC][NUM_P_PC_FROM_GR_TO_PC] = {0};
	float inputSumPFPC[NUM_PC] = {0};
	float gPFPC[NUM_PC] = {0};
	float gBCPC[NUM_PC] = {0};
	float gSCPC[NUM_PC][NUM_P_PC_FROM_SC_TO_PC] = {0};
	float vPC[NUM_PC] = {0};
	float threshPC[NUM_PC] = {0};
	ct_uint32_t *histPCPopAct;

	ct_uint32_t histPCPopActSum;
	ct_uint32_t histPCPopActCurBinN;
	ct_uint32_t pcPopAct;

	//inferior olivary cells
	ct_uint8_t apIO[NUM_IO] = {0};
	ct_uint32_t apBufIO[NUM_IO] = {0};
	ct_uint8_t inputNCIO[NUM_IO][NUM_P_IO_FROM_NC_TO_IO] = {0};
	float errDrive;
	float gNCIO[NUM_IO][NUM_P_IO_FROM_NC_TO_IO] = {0};
	float threshIO[NUM_IO] = {0};
	float vIO[NUM_IO] = {0};
	float vCoupleIO[NUM_IO] = {0};
	ct_int32_t pfPCPlastTimerIO[NUM_IO] = {0};

	//nucleus cells
	ct_uint8_t noLTPMFNC;
	ct_uint8_t noLTDMFNC;
	ct_uint8_t apNC[NUM_NC] = {0};
	ct_uint32_t apBufNC[NUM_NC] = {0};
	ct_uint8_t inputPCNC[NUM_NC][NUM_P_NC_FROM_PC_TO_NC] = {0};
	ct_uint8_t inputMFNC[NUM_NC][NUM_P_NC_FROM_MF_TO_NC] = {0};
	float gPCNC[NUM_NC][NUM_P_NC_FROM_PC_TO_NC] = {0};
	float gPCScaleNC[NUM_NC][NUM_P_NC_FROM_PC_TO_NC] = {0};
	float mfSynWeightNC[NUM_NC][NUM_P_NC_FROM_MF_TO_NC] = {0};
	float gmaxNMDAofMFtoNC[NUM_NC][NUM_P_NC_FROM_MF_TO_NC] = {0};
	float gmaxAMPAofMFtoNC[NUM_NC][NUM_P_NC_FROM_MF_TO_NC] = {0};
	float gMFNMDANC[NUM_NC][NUM_P_NC_FROM_MF_TO_NC] = {0};
	float gMFAMPANC[NUM_NC][NUM_P_NC_FROM_MF_TO_NC] = {0};
	float threshNC[NUM_NC] = {0};
	float vNC[NUM_NC] = {0};
	float synIOPReleaseNC[NUM_NC] = {0};

private:
	void allocateMemory(ct_uint32_t numPopHistBinsPC);
	void initializeVals(ActivityParams &ap, int randSeed);
	void stateRW(ct_uint32_t numPopHistBinsPC, bool read, std::fstream &file);
};


#endif /* MZONEACTIVITYSTATE_H_ */
