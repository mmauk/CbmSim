/*
 * mzoneactivitystate.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: consciousness
 */

#include "state/mzoneactivitystate.h"

MZoneActivityState::MZoneActivityState(ActivityParams *ap, int randSeed)
{
	this->ap = ap;
	allocateMemory(ap);
	initializeVals(ap, randSeed);
}

MZoneActivityState::MZoneActivityState(ActivityParams *ap,
		std::fstream &infile)
{
	this->ap = ap;
	allocateMemory(ap);
	stateRW(ap, true, infile);
}

//MZoneActivityState::MZoneActivityState(const MZoneActivityState &state)
//{
//	// TODO: for all states, overload assignment op for deep copying.
//	// or overload the constructor and use in assignment
//	cp = state.cp;
//	ap = state.ap;
//
//	allocateMemory();
//
//	for (int i = 0; i < numBC; i++)
//	{
//		apBC[i]		 = state.apBC[i];
//		apBufBC[i]	 = state.apBufBC[i];
//		inputPCBC[i] = state.inputPCBC[i];
//		gPFBC[i]	 = state.gPCBC[i];
//		gPCBC[i]	 = state.gPCBC[i];
//		threshBC[i]	 = state.threshBC[i];
//		vBC[i]		 = state.vBC[i];
//	}
//
//	for (int i = 0; i < numPC; i++)
//	{
//		apPC[i]			= state.apPC[i];
//		apBufPC[i]		= state.apBufPC[i];
//		inputBCPC[i]	= state.inputBCPC[i];
//		inputSumPFPC[i]	= state.inputSumPFPC[i];
//		gPFPC[i]		= state.gPFPC[i];
//		gBCPC[i]		= state.gBCPC[i];
//		vPC[i]			= state.vPC[i];
//		threshPC[i]		= state.threshPC[i];
//
//		for (int j = 0; j < numpPCfromSCtoPC; j++)
//		{
//			inputSCPC[i][j] = state.inputSCPC[i][j];
//			gSCPC[i][j]		=state.gSCPC[i][j];
//		}
//
//		for (int j = 0; j < numpPCfromGRtoPC; j++)
//		{
//			pfSynWeightPC[i][j] = state.pfSynWeightPC[i][j];
//		}
//	}
//
//	for (int i = 0; i < ap->numPopHistBinsPC; i++) histPCPopAct[i] = state.histPCPopAct[i];
//
//	histPCPopActSum		= state.histPCPopActSum;
//	histPCPopActCurBinN = state.histPCPopActCurBinN;
//	pcPopAct			= state.pcPopAct;
//
//	for (int i = 0; i < numIO; i++)
//	{
//		apIO[i]				= state.apIO[i];
//		apBufIO[i]			= state.apBufIO[i];
//		threshIO[i] 		= state.threshIO[i];
//		vIO[i]				= state.vIO[i];
//		vCoupleIO[i]		= state.vCoupleIO[i];
//		pfPCPlastTimerIO[i] = state.pfPCPlastTimerIO[i];
//
//		for (int j = 0; j < numpIOfromNCtoIO; j++)
//		{
//			inputNCIO[i][j] = state.inputNCIO[i][j];
//			gNCIO[i][j]		= state.gNCIO[i][j];
//		}
//	}
//
//	errDrive = state.errDrive;
//
//	for (int i = 0; i < numNC; i++)
//	{
//		apNC[i]	   		   = state.apNC[i];
//		apBufNC[i] 		   = state.apBufNC[i];
//		threshNC[i]		   = state.threshNC[i];
//		vNC[i]     		   = state.vNC[i];
//		synIOPReleaseNC[i] = state.synIOPReleaseNC[i];
//
//		for (int j = 0; j < numpNCfromPCtoNC; j++)
//		{
//			inputPCNC[i][j]  = state.inputPCNC[i][j];
//			gPCNC[i][j]		 = state.gPCNC[i][j];
//			gPCScaleNC[i][j] = state.gPCScaleNC[i][j];
//		}
//
//		for (int j = 0; j < numpNCfromMFtoNC; j++)
//		{
//			inputMFNC[i][j]	   	   = state.inputMFNC[i][j];
//			mfSynWeightNC[i][j]	   = state.mfSynWeightNC[i][j];
//			gmaxNMDAofMFtoNC[i][j] = state.gmaxNMDAofMFtoNC[i][j];
//			gmaxAMPAofMFtoNC[i][j] = state.gmaxAMPAofMFtoNC[i][j];
//			gMFNMDANC[i][j]		   = state.gMFNMDANC[i][j];
//			gMFAMPANC[i][j]		   = state.gMFAMPANC[i][j];
//		}
//	}
//
//	noLTPMFNC =state.noLTPMFNC;
//	noLTDMFNC =state.noLTDMFNC;
//}

MZoneActivityState::~MZoneActivityState()
{
	delete[] histPCPopAct;

}

void MZoneActivityState::writeState(std::fstream &outfile)
{
	stateRW(ap, false, outfile);
}

bool MZoneActivityState::state_equal(const MZoneActivityState &compState)
{
	bool eq = true;

	for (int i = 0; i < NUM_BC; i++)
	{
		eq = eq && (threshBC[i] == compState.threshBC[i]) && (vBC[i] == compState.vBC[i]);
	}
	for (int i = 0; i < NUM_PC; i++)
	{
		eq = eq && (threshPC[i] == compState.threshPC[i]) && (vPC[i] == compState.vPC[i]);
	}
	for (int i = 0; i < NUM_NC; i++)
	{
		eq = eq && (vNC[i] == compState.vNC[i]) && (synIOPReleaseNC[i]==compState.synIOPReleaseNC[i]);
	}

	return eq;
}

bool MZoneActivityState::state_unequal(const MZoneActivityState &compState) 
{
	return !state_equal(compState);
}

// where do we use this function??
std::vector<float> MZoneActivityState::getGRPCSynWeightLinear()
{
	std::vector<float> retVec(NUM_GR, 0.0);
	for (int i = 0; i < NUM_PC; i++)
	{
		for (int j = 0; j < NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			retVec[i * NUM_P_PC_FROM_GR_TO_PC + j] = pfSynWeightPC[i][j];
		}
	}
	return retVec;
}

void MZoneActivityState::resetGRPCSynWeight()
{
	for (int i = 0; i < NUM_PC; i++)
	{
		for (int j = 0; j < NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			pfSynWeightPC[i][j] = ap->initSynWofGRtoPC;
		}
	}
}

void MZoneActivityState::allocateMemory()
{
	histPCPopAct  = new ct_uint32_t[ap->numPopHistBinsPC];
}

void MZoneActivityState::initializeVals(int randSeed)
{
	//uncomment for actual runs 	
	//CRandomSFMT0 randGen(randSeed);

	// bc
	std::fill(threshBC, threshBC + NUM_BC, ap->threshRestBC);
	std::fill(vBC, vBC + NUM_BC, ap->eLeakBC);

	// pc
	std::fill(vPC, vPC + NUM_PC, ap->eLeakPC);
	std::fill(threshPC, threshPC + NUM_PC, ap->threshRestPC);	

	std::fill(pfSynWeightPC[0], pfSynWeightPC[0]
		+ NUM_PC * NUM_P_PC_FROM_GR_TO_PC, ap->initSynWofGRtoPC)

	std::fill(histPCPopAct, histPCPopAct + ap->numPopHistBinsPC, 0);	

	histPCPopActSum		= 0;
	histPCPopActCurBinN = 0;
	pcPopAct			= 0;

	// IO
	std::fill(threshIO, threshIO + NUM_IO, ap->threshRestIO);
	std::fill(vIO, vIO + NUM_IO, ap->eLeakIO);

	errDrive = 0;
	
	// NC
	noLTPMFNC = 0;
	noLTDMFNC = 0;

	std::fill(threshNC, threshNC + NUM_NC, ap->threshRestNC);
	std::fill(vNC, vNC + NUM_NC, ap->eLeakNC);
	std::fill(gPCScaleNC[0], gPCScaleNC[0]
		+ NUM_NC * NUM_P_NC_FROM_PC_TO_NC, ap->gIncAvgPCtoNC);

	std::fill(mfSynWeightNC[0], mfSynWeightNC[0]
		+ NUM_NC * NUM_P_NC_FROM_MF_TO_NC, ap->initSynWofMFtoNC);
}

void MZoneActivityState::stateRW(bool read, std::fstream &file)
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
	rawBytesRW((char *)histPCPopAct, ap->numPopHistBinsPC * sizeof(ct_uint32_t), read, file);
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

