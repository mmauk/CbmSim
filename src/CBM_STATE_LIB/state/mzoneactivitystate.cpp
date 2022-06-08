/*
 * mzoneactivitystate.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: consciousness
 */

#include "state/mzoneactivitystate.h"

MZoneActivityState::MZoneActivityState(ConnectivityParams &cp, ActivityParams *ap, int randSeed)
{
	allocateMemory(ap);
	initializeVals(cp, ap, randSeed);
}

MZoneActivityState::MZoneActivityState(ConnectivityParams &cp, ActivityParams *ap,
		std::fstream &infile)
{
	allocateMemory(ap);
	stateRW(cp, ap, true, infile);
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
//	for (int i = 0; i < cp->numBC; i++)
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
//	for (int i = 0; i < cp->numPC; i++)
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
//		for (int j = 0; j < cp->numpPCfromSCtoPC; j++)
//		{
//			inputSCPC[i][j] = state.inputSCPC[i][j];
//			gSCPC[i][j]		=state.gSCPC[i][j];
//		}
//
//		for (int j = 0; j < cp->numpPCfromGRtoPC; j++)
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
//	for (int i = 0; i < cp->numIO; i++)
//	{
//		apIO[i]				= state.apIO[i];
//		apBufIO[i]			= state.apBufIO[i];
//		threshIO[i] 		= state.threshIO[i];
//		vIO[i]				= state.vIO[i];
//		vCoupleIO[i]		= state.vCoupleIO[i];
//		pfPCPlastTimerIO[i] = state.pfPCPlastTimerIO[i];
//
//		for (int j = 0; j < cp->numpIOfromNCtoIO; j++)
//		{
//			inputNCIO[i][j] = state.inputNCIO[i][j];
//			gNCIO[i][j]		= state.gNCIO[i][j];
//		}
//	}
//
//	errDrive = state.errDrive;
//
//	for (int i = 0; i < cp->numNC; i++)
//	{
//		apNC[i]	   		   = state.apNC[i];
//		apBufNC[i] 		   = state.apBufNC[i];
//		threshNC[i]		   = state.threshNC[i];
//		vNC[i]     		   = state.vNC[i];
//		synIOPReleaseNC[i] = state.synIOPReleaseNC[i];
//
//		for (int j = 0; j < cp->numpNCfromPCtoNC; j++)
//		{
//			inputPCNC[i][j]  = state.inputPCNC[i][j];
//			gPCNC[i][j]		 = state.gPCNC[i][j];
//			gPCScaleNC[i][j] = state.gPCScaleNC[i][j];
//		}
//
//		for (int j = 0; j < cp->numpNCfromMFtoNC; j++)
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

void MZoneActivityState::writeState(ConnectivityParams &cp, ActivityParams *ap, std::fstream &outfile)
{
	stateRW(cp, ap, false, outfile);
}

bool MZoneActivityState::state_equal(ConnectivityParams &cp, const MZoneActivityState &compState)
{
	bool eq = true;

	for (int i = 0; i < cp.NUM_BC; i++)
	{
		eq = eq && (threshBC[i] == compState.threshBC[i]) && (vBC[i] == compState.vBC[i]);
	}
	for (int i = 0; i < cp.NUM_PC; i++)
	{
		eq = eq && (threshPC[i] == compState.threshPC[i]) && (vPC[i] == compState.vPC[i]);
	}
	for (int i = 0; i < cp.NUM_NC; i++)
	{
		eq = eq && (vNC[i] == compState.vNC[i]) && (synIOPReleaseNC[i]==compState.synIOPReleaseNC[i]);
	}

	return eq;
}

bool MZoneActivityState::state_unequal(ConnectivityParams &cp, const MZoneActivityState &compState) 
{
	return !state_equal(cp, compState);
}

std::vector<float> MZoneActivityState::getGRPCSynWeightLinear(ConnectivityParams &cp)
{
	std::vector<float> retVec(cp.NUM_GR, 0.0);
	for (int i = 0; i < cp.NUM_PC; i++)
	{
		for (int j = 0; j < cp.NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			retVec[i * cp.NUM_P_PC_FROM_GR_TO_PC + j] = pfSynWeightPC[i][j];
		}
	}
	return retVec;
}

void MZoneActivityState::resetGRPCSynWeight(ConnectivityParams &cp, ActivityParams *ap)
{
	for (int i = 0; i < cp.NUM_PC; i++)
	{
		for (int j = 0; j < cp.NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			pfSynWeightPC[i][j] = ap->initSynWofGRtoPC;
		}
	}
}

void MZoneActivityState::allocateMemory(ActivityParams *ap)
{
	histPCPopAct  = new ct_uint32_t[ap->numPopHistBinsPC];
}

void MZoneActivityState::initializeVals(ConnectivityParams &cp, ActivityParams *ap, int randSeed)
{
	//uncomment for actual runs 	
	//CRandomSFMT0 randGen(randSeed);

	// bc
	std::fill(threshBC, threshBC + cp.NUM_BC, ap->threshRestBC);
	std::fill(vBC, vBC + cp.NUM_BC, ap->eLeakBC);

	// pc
	std::fill(vPC, vPC + cp.NUM_PC, ap->eLeakPC);
	std::fill(threshPC, threshPC + cp.NUM_PC, ap->threshRestPC);	

	std::fill(pfSynWeightPC[0], pfSynWeightPC[0]
		+ cp.NUM_PC * cp.NUM_P_PC_FROM_GR_TO_PC, ap->initSynWofGRtoPC)

	std::fill(histPCPopAct, histPCPopAct + ap->numPopHistBinsPC, 0);	

	histPCPopActSum		= 0;
	histPCPopActCurBinN = 0;
	pcPopAct			= 0;

	// IO
	std::fill(threshIO, threshIO + cp.NUM_IO, ap->threshRestIO);
	std::fill(vIO, vIO + cp.NUM_IO, ap->eLeakIO);

	errDrive = 0;
	
	// NC
	noLTPMFNC = 0;
	noLTDMFNC = 0;

	std::fill(threshNC, threshNC + cp.NUM_NC, ap->threshRestNC);
	std::fill(vNC, vNC + cp.NUM_NC, ap->eLeakNC);
	std::fill(gPCScaleNC[0], gPCScaleNC[0]
		+ cp.NUM_NC * cp.NUM_P_NC_FROM_PC_TO_NC, ap->gIncAvgPCtoNC);

	std::fill(mfSynWeightNC[0], mfSynWeightNC[0]
		+ cp.NUM_NC * cp.NUM_P_NC_FROM_MF_TO_NC, ap->initSynWofMFtoNC);
}

void MZoneActivityState::stateRW(ConnectivityParams &cp, ActivityParams *ap, bool read, std::fstream &file)
{
	rawBytesRW((char *)apBC, cp.NUM_BC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufBC, cp.NUM_BC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCBC, cp.NUM_BC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)gPFBC, cp.NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)gPCBC, cp.NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)threshBC, cp.NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)vBC, cp.NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)apPC, cp.NUM_PC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufPC, cp.NUM_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputBCPC, cp.NUM_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputSCPC[0],
		cp.NUM_PC * cp.NUM_P_PC_FROM_SC_TO_PC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)pfSynWeightPC[0],
		cp.NUM_PC * cp.NUM_P_PC_FROM_GR_TO_PC * sizeof(float), read, file);
	rawBytesRW((char *)inputSumPFPC, cp.NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gPFPC, cp.NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gBCPC, cp.NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gSCPC[0],
		cp.NUM_PC * cp.NUM_P_PC_FROM_SC_TO_PC * sizeof(float), read, file);
	rawBytesRW((char *)vPC, cp.NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)threshPC, cp.NUM_PC * sizeof(float), read, file);
	// ONE ARRAY WHYYYY
	rawBytesRW((char *)histPCPopAct, ap->numPopHistBinsPC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActSum, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActCurBinN, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&pcPopAct, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)apIO, cp.NUM_IO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufIO, cp.NUM_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputNCIO[0],
		cp.NUM_IO * cp.NUM_P_IO_FROM_NC_TO_IO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&errDrive, sizeof(float), read, file);
	rawBytesRW((char *)gNCIO[0],
		cp.NUM_IO * cp.NUM_P_IO_FROM_NC_TO_IO * sizeof(float), read, file);
	rawBytesRW((char *)threshIO, cp.NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)vIO, cp.NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleIO, cp.NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)pfPCPlastTimerIO, cp.NUM_IO * sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)&noLTPMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&noLTDMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apNC, cp.NUM_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufNC, cp.NUM_NC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCNC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_PC_TO_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)inputMFNC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_MF_TO_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)gPCNC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_PC_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gPCScaleNC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_PC_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)mfSynWeightNC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gmaxNMDAofMFtoNC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gmaxAMPAofMFtoNC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gMFNMDANC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gMFAMPANC[0],
		cp.NUM_NC * cp.NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)threshNC, cp.NUM_NC * sizeof(float), read, file);
	rawBytesRW((char *)vNC, cp.NUM_NC * sizeof(float), read, file);
	rawBytesRW((char *)synIOPReleaseNC, cp.NUM_NC * sizeof(float), read, file);
}

