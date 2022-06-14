/*
 * mzoneactivitystate.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: consciousness
 */

#include "state/mzoneactivitystate.h"

MZoneActivityState::MZoneActivityState() {}

MZoneActivityState::MZoneActivityState(ActivityParams &ap, int randSeed)
{
	allocateMemory(ap.numPopHistBinsPC);
	initializeVals(ap, randSeed);
}

MZoneActivityState::MZoneActivityState(ActivityParams &ap, std::fstream &infile)
{
	allocateMemory(ap.numPopHistBinsPC);
	stateRW(ap.numPopHistBinsPC, true, infile);
}

MZoneActivityState::~MZoneActivityState()
{
	delete[] histPCPopAct;

}

void MZoneActivityState::writeState(ActivityParams &ap, std::fstream &outfile)
{
	stateRW(ap.numPopHistBinsPC, false, outfile);
}

void MZoneActivityState::allocateMemory(ct_uint32_t numPopHistBinsPC)
{
	histPCPopAct  = new ct_uint32_t[numPopHistBinsPC];
}

void MZoneActivityState::initializeVals(ActivityParams &ap, int randSeed)
{
	//uncomment for actual runs 	
	//CRandomSFMT0 randGen(randSeed);

	// bc
	std::fill(threshBC, threshBC + NUM_BC, ap.threshRestBC);
	std::fill(vBC, vBC + NUM_BC, ap.eLeakBC);

	// pc
	std::fill(vPC, vPC + NUM_PC, ap.eLeakPC);
	std::fill(threshPC, threshPC + NUM_PC, ap.threshRestPC);	

	std::fill(pfSynWeightPC[0], pfSynWeightPC[0]
		+ NUM_PC * NUM_P_PC_FROM_GR_TO_PC, ap.initSynWofGRtoPC);

	std::fill(histPCPopAct, histPCPopAct + ap.numPopHistBinsPC, 0);	

	histPCPopActSum		= 0;
	histPCPopActCurBinN = 0;
	pcPopAct			= 0;

	// IO
	std::fill(threshIO, threshIO + NUM_IO, ap.threshRestIO);
	std::fill(vIO, vIO + NUM_IO, ap.eLeakIO);

	errDrive = 0;
	
	// NC
	noLTPMFNC = 0;
	noLTDMFNC = 0;

	std::fill(threshNC, threshNC + NUM_NC, ap.threshRestNC);
	std::fill(vNC, vNC + NUM_NC, ap.eLeakNC);
	std::fill(gPCScaleNC[0], gPCScaleNC[0]
		+ NUM_NC * NUM_P_NC_FROM_PC_TO_NC, ap.gIncAvgPCtoNC);

	std::fill(mfSynWeightNC[0], mfSynWeightNC[0]
		+ NUM_NC * NUM_P_NC_FROM_MF_TO_NC, ap.initSynWofMFtoNC);
}

void MZoneActivityState::stateRW(ct_uint32_t numPopHistBinsPC, bool read, std::fstream &file)
{
	rawBytesRW((char *)apBC, NUM_BC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufBC, NUM_BC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCBC, NUM_BC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)gPFBC, NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)gPCBC, NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)threshBC, NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)vBC, NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)apPC, NUM_PC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufPC, NUM_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputBCPC, NUM_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputSCPC[0],
		NUM_PC * NUM_P_PC_FROM_SC_TO_PC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)pfSynWeightPC[0],
		NUM_PC * NUM_P_PC_FROM_GR_TO_PC * sizeof(float), read, file);
	rawBytesRW((char *)inputSumPFPC, NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gPFPC, NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gBCPC, NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gSCPC[0],
		NUM_PC * NUM_P_PC_FROM_SC_TO_PC * sizeof(float), read, file);
	rawBytesRW((char *)vPC, NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)threshPC, NUM_PC * sizeof(float), read, file);
	// ONE ARRAY WHYYYY
	rawBytesRW((char *)histPCPopAct, numPopHistBinsPC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActSum, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActCurBinN, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&pcPopAct, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)apIO, NUM_IO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufIO, NUM_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputNCIO[0],
		NUM_IO * NUM_P_IO_FROM_NC_TO_IO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&errDrive, sizeof(float), read, file);
	rawBytesRW((char *)gNCIO[0],
		NUM_IO * NUM_P_IO_FROM_NC_TO_IO * sizeof(float), read, file);
	rawBytesRW((char *)threshIO, NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)vIO, NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleIO, NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)pfPCPlastTimerIO, NUM_IO * sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)&noLTPMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&noLTDMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apNC, NUM_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufNC, NUM_NC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCNC[0],
		NUM_NC * NUM_P_NC_FROM_PC_TO_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)inputMFNC[0],
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)gPCNC[0],
		NUM_NC * NUM_P_NC_FROM_PC_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gPCScaleNC[0],
		NUM_NC * NUM_P_NC_FROM_PC_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)mfSynWeightNC[0],
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gmaxNMDAofMFtoNC[0],
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gmaxAMPAofMFtoNC[0],
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gMFNMDANC[0],
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gMFAMPANC[0],
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)threshNC, NUM_NC * sizeof(float), read, file);
	rawBytesRW((char *)vNC, NUM_NC * sizeof(float), read, file);
	rawBytesRW((char *)synIOPReleaseNC, NUM_NC * sizeof(float), read, file);
}

