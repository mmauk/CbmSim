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
#include <algorithm> /* std::fill */

#include <memoryMgmt/dynamic2darray.h>
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
	MZoneActivityState();
	MZoneActivityState(ConnectivityParams &cp, ActivityParams *actParams, int randSeed);
	MZoneActivityState(ConnectivityParams &cp, ActivityParams *actParams, std::fstream &infile);
	//MZoneActivityState(const MZoneActivityState &state);

	virtual ~MZoneActivityState();

	void writeState(ConnectivityParams &cp, ActivityParams *ap, std::fstream &outfile);

	bool state_equal(ConnectivityParams &cp, const MZoneActivityState &compState);
	bool state_unequal(ConnectivityParams &cp, const MZoneActivityState &compState);

	std::vector<float> getGRPCSynWeightLinear(ConnectivityParams &cp);
	void resetGRPCSynWeight(ConnectivityParams &cp, ActivityParams *ap);

	//basket cells
	ct_uint8_t apBC[cp.NUM_BC]();
	ct_uint32_t apBufBC[cp.NUM_BC]();
	ct_uint32_t inputPCBC[cp.NUM_BC]();
	float gPFBC[cp.NUM_BC]();
	float gPCBC[cp.NUM_BC]();
	float threshBC[cp.NUM_BC]();
	float vBC[cp.NUM_BC]();

	//purkinje cells
	ct_uint8_t apPC[cp.NUM_PC]();
	ct_uint32_t apBufPC[cp.NUM_PC]();
	ct_uint32_t inputBCPC[cp.NUM_PC]();
	ct_uint8_t inputSCPC[cp.NUM_PC][cp.NUM_P_PC_FROM_SC_TO_PC]();
	float pfSynWeightPC[cp.NUM_PC][cp.NUM_P_PC_FROM_GR_TO_PC]();
	float inputSumPFPC[cp.NUM_PC]();
	float gPFPC[cp.NUM_PC]();
	float gBCPC[cp.NUM_PC]();
	float gSCPC[cp.NUM_PC][cp.NUM_P_PC_FROM_SC_TO_PC]();
	float vPC[cp.NUM_PC]();
	float threshPC[cp.NUM_PC]();
	ct_uint32_t *histPCPopAct;

	ct_uint32_t histPCPopActSum;
	ct_uint32_t histPCPopActCurBinN;
	ct_uint32_t pcPopAct;

	//inferior olivary cells
	ct_uint8_t apIO[cp.NUM_IO]();
	ct_uint32_t apBufIO[cp.NUM_IO]();
	ct_uint8_t inputNCIO[cp.NUM_IO][cp.NUM_P_IO_FROM_NC_TO_IO]();
	float errDrive;
	float gNCIO[cp.NUM_IO][cp.NUM_P_IO_FROM_NC_TO_IO]();
	float threshIO[cp.NUM_IO]();
	float vIO[cp.NUM_IO]();
	float vCoupleIO[cp.NUM_IO]();
	ct_int32_t pfPCPlastTimerIO[cp.NUM_IO]();

	//nucleus cells
	ct_uint8_t noLTPMFNC;
	ct_uint8_t noLTDMFNC;
	ct_uint8_t apNC[cp.NUM_NC]();
	ct_uint32_t apBufNC[cp.NUM_NC]();
	ct_uint8_t inputPCNC[cp.NUM_NC][cp.NUM_P_NC_FROM_PC_TO_NC]();
	ct_uint8_t inputMFNC[cp.NUM_NC][cp.NUM_P_NC_FROM_MF_TO_NC]();
	float gPCNC[cp.NUM_NC][cp.NUM_P_NC_FROM_PC_TO_NC]();
	float gPCScaleNC[cp.NUM_NC][cp.NUM_P_NC_FROM_PC_TO_NC]();
	float mfSynWeightNC[cp.NUM_NC][cp.NUM_P_NC_FROM_MF_TO_NC]();
	float gmaxNMDAofMFtoNC[cp.NUM_NC][cp.NUM_P_NC_FROM_MF_TO_NC]();
	float gmaxAMPAofMFtoNC[cp.NUM_NC][cp.NUM_P_NC_FROM_MF_TO_NC]();
	float gMFNMDANC[cp.NUM_NC][cp.NUM_P_NC_FROM_MF_TO_NC]();
	float gMFAMPANC[cp.NUM_NC][cp.NUM_P_NC_FROM_MF_TO_NC]();
	float threshNC[cp.NUM_NC]();
	float vNC[cp.NUM_NC]();
	float synIOPReleaseNC[cp.NUM_NC]();

private:

	void allocateMemory(ActivityParams *ap);
	void initializeVals(ConnectivityParams &cp, ActivityParams *ap, int randSeed);
	void stateRW(ConnectivityParams &cp, ActivityParams *ap, bool read, std::fstream &file);
};


#endif /* MZONEACTIVITYSTATE_H_ */
