/*
 * mzoneactivitystate.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: consciousness
 */

#include "state/mzoneactivitystate.h"

MZoneActivityState::MZoneActivityState(ConnectivityParams *conParams, ActivityParams *actParams, int randSeed)
{
	cp = conParams;
	ap = actParams;

	allocateMemory();
	initializeVals(randSeed);
}

MZoneActivityState::MZoneActivityState(ConnectivityParams *conParams, ActivityParams *actParams,
		std::fstream &infile)
{
	cp = conParams;
	ap = actParams;

	allocateMemory();
	stateRW(true, infile);
}

MZoneActivityState::MZoneActivityState(const MZoneActivityState &state)
{
	// TODO: for all states, overload assignment op for deep copying.
	// or overload the constructor and use in assignment
	cp = state.cp;
	ap = state.ap;

	allocateMemory();

	for (int i = 0; i < cp->numBC; i++)
	{
		apBC[i]		 = state.apBC[i];
		apBufBC[i]	 = state.apBufBC[i];
		inputPCBC[i] = state.inputPCBC[i];
		gPFBC[i]	 = state.gPCBC[i];
		gPCBC[i]	 = state.gPCBC[i];
		threshBC[i]	 = state.threshBC[i];
		vBC[i]		 = state.vBC[i];
	}

	for (int i = 0; i < cp->numPC; i++)
	{
		apPC[i]			= state.apPC[i];
		apBufPC[i]		= state.apBufPC[i];
		inputBCPC[i]	= state.inputBCPC[i];
		inputSumPFPC[i]	= state.inputSumPFPC[i];
		gPFPC[i]		= state.gPFPC[i];
		gBCPC[i]		= state.gBCPC[i];
		vPC[i]			= state.vPC[i];
		threshPC[i]		= state.threshPC[i];

		for (int j = 0; j < cp->numpPCfromSCtoPC; j++)
		{
			inputSCPC[i][j] = state.inputSCPC[i][j];
			gSCPC[i][j]		=state.gSCPC[i][j];
		}

		for (int j = 0; j < cp->numpPCfromGRtoPC; j++)
		{
			pfSynWeightPC[i][j] = state.pfSynWeightPC[i][j];
		}
	}

	for (int i = 0; i < ap->numPopHistBinsPC; i++) histPCPopAct[i] = state.histPCPopAct[i];

	histPCPopActSum		= state.histPCPopActSum;
	histPCPopActCurBinN = state.histPCPopActCurBinN;
	pcPopAct			= state.pcPopAct;

	for (int i = 0; i < cp->numIO; i++)
	{
		apIO[i]				= state.apIO[i];
		apBufIO[i]			= state.apBufIO[i];
		threshIO[i] 		= state.threshIO[i];
		vIO[i]				= state.vIO[i];
		vCoupleIO[i]		= state.vCoupleIO[i];
		pfPCPlastTimerIO[i] = state.pfPCPlastTimerIO[i];

		for (int j = 0; j < cp->numpIOfromNCtoIO; j++)
		{
			inputNCIO[i][j] = state.inputNCIO[i][j];
			gNCIO[i][j]		= state.gNCIO[i][j];
		}
	}

	errDrive = state.errDrive;

	for (int i = 0; i < cp->numNC; i++)
	{
		apNC[i]	   		   = state.apNC[i];
		apBufNC[i] 		   = state.apBufNC[i];
		threshNC[i]		   = state.threshNC[i];
		vNC[i]     		   = state.vNC[i];
		synIOPReleaseNC[i] = state.synIOPReleaseNC[i];

		for (int j = 0; j < cp->numpNCfromPCtoNC; j++)
		{
			inputPCNC[i][j]  = state.inputPCNC[i][j];
			gPCNC[i][j]		 = state.gPCNC[i][j];
			gPCScaleNC[i][j] = state.gPCScaleNC[i][j];
		}

		for (int j = 0; j < cp->numpNCfromMFtoNC; j++)
		{
			inputMFNC[i][j]	   	   = state.inputMFNC[i][j];
			mfSynWeightNC[i][j]	   = state.mfSynWeightNC[i][j];
			gmaxNMDAofMFtoNC[i][j] = state.gmaxNMDAofMFtoNC[i][j];
			gmaxAMPAofMFtoNC[i][j] = state.gmaxAMPAofMFtoNC[i][j];
			gMFNMDANC[i][j]		   = state.gMFNMDANC[i][j];
			gMFAMPANC[i][j]		   = state.gMFAMPANC[i][j];
		}
	}

	noLTPMFNC =state.noLTPMFNC;
	noLTDMFNC =state.noLTDMFNC;
}

MZoneActivityState::~MZoneActivityState()
{
	delete[] apBC;
	delete[] apBufBC;
	delete[] inputPCBC;
	delete[] gPFBC;
	delete[] gPCBC;
	delete[] threshBC;
	delete[] vBC;

	delete[] apPC;
	delete[] apBufPC;
	delete[] inputBCPC;

	delete2DArray<ct_uint8_t>(inputSCPC);
	delete2DArray<float>(pfSynWeightPC);

	delete[] inputSumPFPC;
	delete[] gPFPC;
	delete[] gBCPC;

	delete2DArray<float>(gSCPC);

	delete[] vPC;
	delete[] threshPC;
	delete[] histPCPopAct;

	delete[] apIO;
	delete[] apBufIO;

	delete2DArray<ct_uint8_t>(inputNCIO);
	delete2DArray<float>(gNCIO);

	delete[] threshIO;
	delete[] vIO;
	delete[] vCoupleIO;
	delete[] pfPCPlastTimerIO;

	delete[] apNC;
	delete[] apBufNC;

	delete2DArray<ct_uint8_t>(inputPCNC);
	delete2DArray<ct_uint8_t>(inputMFNC);

	delete2DArray<float>(gPCNC);
	delete2DArray<float>(gPCScaleNC);
	delete2DArray<float>(mfSynWeightNC);
	delete2DArray<float>(gmaxNMDAofMFtoNC);
	delete2DArray<float>(gmaxAMPAofMFtoNC);
	delete2DArray<float>(gMFNMDANC);
	delete2DArray<float>(gMFAMPANC);

	delete[] threshNC;
	delete[] vNC;
	delete[] synIOPReleaseNC;
}

void MZoneActivityState::writeState(std::fstream &outfile)
{
	stateRW(false, (std::fstream &)outfile);
}

bool MZoneActivityState::operator==(const MZoneActivityState &compState)
{
	bool eq = true;

	for (int i = 0; i < cp->numBC; i++)
	{
		eq = eq && (threshBC[i] == compState.threshBC[i]) && (vBC[i] == compState.vBC[i]);
	}
	for (int i = 0; i < cp->numPC; i++)
	{
		eq = eq && (threshPC[i] == compState.threshPC[i]) && (vPC[i] == compState.vPC[i]);
	}
	for (int i = 0; i < cp->numNC; i++)
	{
		eq = eq && (vNC[i] == compState.vNC[i]) && (synIOPReleaseNC[i]==compState.synIOPReleaseNC[i]);
	}

	return eq;
}

bool MZoneActivityState::operator!=(const MZoneActivityState &compState) 
{
	return !(this == compState);
}

std::vector<float> MZoneActivityState::getGRPCSynWeightLinear()
{
	// are we really going to return an entire-ass vector?
	std::vector<float> retVec;

	retVec.resize(cp->numGR);

	for (int i = 0; i < cp->numPC; i++)
	{
		for (int j = 0; j < cp->numpPCfromGRtoPC; j++)
		{
			retVec[i * cp->numpPCfromGRtoPC + j] = pfSynWeightPC[i][j];
		}
	}

	return retVec;
}

void MZoneActivityState::resetGRPCSynWeight()
{
	for (int i = 0; i < cp->numPC; i++)
	{
		for (int j = 0; j < cp->numpPCfromGRtoPC; j++)
		{
			pfSynWeightPC[i][j] = ap->initSynWofGRtoPC;
		}
	}
}

void MZoneActivityState::allocateMemory()
{
	apBC   	  = new ct_uint8_t[cp->numBC];
	apBufBC	  = new ct_uint32_t[cp->numBC];
	inputPCBC = new ct_uint32_t[cp->numBC];
	gPFBC	  = new float[cp->numBC];
	gPCBC	  = new float[cp->numBC];
	threshBC  = new float[cp->numBC];
	vBC		  = new float[cp->numBC];

	apPC          = new ct_uint8_t[cp->numPC];
	apBufPC       = new ct_uint32_t[cp->numPC];
	inputBCPC     = new ct_uint32_t[cp->numPC];
	inputSCPC     = allocate2DArray<ct_uint8_t>(cp->numPC, cp->numpPCfromSCtoPC);
	pfSynWeightPC = allocate2DArray<float>(cp->numPC, cp->numpPCfromGRtoPC);
	inputSumPFPC  = new float[cp->numPC];
	gPFPC		  = new float[cp->numPC];
	gBCPC		  = new float[cp->numPC];
	gSCPC		  = allocate2DArray<float>(cp->numPC, cp->numpPCfromSCtoPC);
	vPC     	  = new float[cp->numPC];
	threshPC	  = new float[cp->numPC];
	histPCPopAct  = new ct_uint32_t[ap->numPopHistBinsPC];

	apIO     		 = new ct_uint8_t[cp->numIO];
	apBufIO  		 = new ct_uint32_t[cp->numIO];
	inputNCIO		 = allocate2DArray<ct_uint8_t>(cp->numIO, cp->numpIOfromNCtoIO);
	gNCIO			 = allocate2DArray<float>(cp->numIO, cp->numpIOfromNCtoIO);
	threshIO		 = new float[cp->numIO];
	vIO				 = new float[cp->numIO];
	vCoupleIO		 = new float[cp->numIO];
	pfPCPlastTimerIO = new ct_int32_t[cp->numIO];

	apNC	     	 = new ct_uint8_t[cp->numNC];
	apBufNC      	 = new ct_uint32_t[cp->numNC];
	inputPCNC    	 = allocate2DArray<ct_uint8_t>(cp->numNC, cp->numpNCfromPCtoNC);
	inputMFNC    	 = allocate2DArray<ct_uint8_t>(cp->numNC, cp->numpNCfromMFtoNC);
	gPCNC	     	 = allocate2DArray<float>(cp->numNC, cp->numpNCfromPCtoNC);
	gPCScaleNC   	 = allocate2DArray<float>(cp->numNC, cp->numpNCfromPCtoNC);
	mfSynWeightNC	 = allocate2DArray<float>(cp->numNC, cp->numpNCfromMFtoNC);
	gmaxNMDAofMFtoNC = allocate2DArray<float>(cp->numNC, cp->numpNCfromMFtoNC);
	gmaxAMPAofMFtoNC = allocate2DArray<float>(cp->numNC, cp->numpNCfromMFtoNC);
	gMFNMDANC		 = allocate2DArray<float>(cp->numNC, cp->numpNCfromMFtoNC);
	gMFAMPANC		 = allocate2DArray<float>(cp->numNC, cp->numpNCfromMFtoNC);
	threshNC		 = new float[cp->numNC];
	vNC				 = new float[cp->numNC];
	synIOPReleaseNC	 = new float[cp->numNC];
}

void MZoneActivityState::stateRW(bool read, fstream &file)
{
	rawBytesRW((char *)apBC, cp->numBC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufBC, cp->numBC*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCBC, cp->numBC*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)gPFBC, cp->numBC*sizeof(float), read, file);
	rawBytesRW((char *)gPCBC, cp->numBC*sizeof(float), read, file);
	rawBytesRW((char *)threshBC, cp->numBC*sizeof(float), read, file);
	rawBytesRW((char *)vBC, cp->numBC*sizeof(float), read, file);
	rawBytesRW((char *)apPC, cp->numPC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufPC, cp->numPC*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputBCPC, cp->numPC*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputSCPC[0], cp->numPC*cp->numpPCfromSCtoPC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)pfSynWeightPC[0], cp->numPC*cp->numpPCfromGRtoPC*sizeof(float), read, file);
	rawBytesRW((char *)inputSumPFPC, cp->numPC*sizeof(float), read, file);
	rawBytesRW((char *)gPFPC, cp->numPC*sizeof(float), read, file);
	rawBytesRW((char *)gBCPC, cp->numPC*sizeof(float), read, file);
	rawBytesRW((char *)gSCPC[0], cp->numPC*cp->numpPCfromSCtoPC*sizeof(float), read, file);
	rawBytesRW((char *)vPC, cp->numPC*sizeof(float), read, file);
	rawBytesRW((char *)threshPC, cp->numPC*sizeof(float), read, file);
	rawBytesRW((char *)histPCPopAct, ap->numPopHistBinsPC*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActSum, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActCurBinN, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&pcPopAct, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)apIO, cp->numIO*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufIO, cp->numIO*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputNCIO[0], cp->numIO*cp->numpIOfromNCtoIO*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&errDrive, sizeof(float), read, file);
	rawBytesRW((char *)gNCIO[0], cp->numIO*cp->numpIOfromNCtoIO*sizeof(float), read, file);
	rawBytesRW((char *)threshIO, cp->numIO*sizeof(float), read, file);
	rawBytesRW((char *)vIO, cp->numIO*sizeof(float), read, file);
	rawBytesRW((char *)vCoupleIO, cp->numIO*sizeof(float), read, file);
	rawBytesRW((char *)pfPCPlastTimerIO, cp->numIO*sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)&noLTPMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&noLTDMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apNC, cp->numNC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufNC, cp->numNC*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCNC[0], cp->numNC*cp->numpNCfromPCtoNC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)inputMFNC[0], cp->numNC*cp->numpNCfromMFtoNC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)gPCNC[0], cp->numNC*cp->numpNCfromPCtoNC*sizeof(float), read, file);
	rawBytesRW((char *)gPCScaleNC[0], cp->numNC*cp->numpNCfromPCtoNC*sizeof(float), read, file);
	rawBytesRW((char *)mfSynWeightNC[0], cp->numNC*cp->numpNCfromMFtoNC*sizeof(float), read, file);
	rawBytesRW((char *)gmaxNMDAofMFtoNC[0], cp->numNC*cp->numpNCfromMFtoNC*sizeof(float), read, file);
	rawBytesRW((char *)gmaxAMPAofMFtoNC[0], cp->numNC*cp->numpNCfromMFtoNC*sizeof(float), read, file);
	rawBytesRW((char *)gMFNMDANC[0], cp->numNC*cp->numpNCfromMFtoNC*sizeof(float), read, file);
	rawBytesRW((char *)gMFAMPANC[0], cp->numNC*cp->numpNCfromMFtoNC*sizeof(float), read, file);
	rawBytesRW((char *)threshNC, cp->numNC*sizeof(float), read, file);
	rawBytesRW((char *)vNC, cp->numNC*sizeof(float), read, file);
	rawBytesRW((char *)synIOPReleaseNC, cp->numNC*sizeof(float), read, file);
}

void MZoneActivityState::initializeVals(int randSeed)
{
	//uncomment for actual runs 	
	//CRandomSFMT0 randGen(randSeed);

	for (int i = 0; i < cp->numBC; i++)
	{
		apBC[i]	    = 0;
		apBufBC[i]  = 0;
		inputPCBC[i]= 0;
		gPFBC[i]	= 0;
		gPCBC[i]	= 0;
		threshBC[i] = ap->threshRestBC;
		vBC[i]		= ap->eLeakBC;
	}

	for (int i = 0; i < cp->numPC; i++)
	{
		apPC[i]        = 0;
		apBufPC[i]     = 0;
		inputBCPC[i]   = 0;
		inputSumPFPC[i]= 0;
		gPFPC[i]	   = 0;
		gBCPC[i]	   = 0;
		vPC[i]		   = ap->eLeakPC;
		threshPC[i]	   = ap->threshRestPC;

		for (int j = 0; j < cp->numpPCfromSCtoPC; j++)
		{
			inputSCPC[i][j] = 0;
			gSCPC[i][j]		= 0;
		}

		for (int j = 0; j < cp->numpPCfromGRtoPC; j++)
		{
			pfSynWeightPC[i][j] = ap->initSynWofGRtoPC;
		}
	}

	for (int i = 0; i < ap->numPopHistBinsPC; i++) histPCPopAct[i] = 0;

	histPCPopActSum		= 0;
	histPCPopActCurBinN = 0;
	pcPopAct			= 0;

	for (int i = 0; i < cp->numIO; i++)
	{
		apIO[i]	    	    = 0;
		apBufIO[i]  	    = 0;
		threshIO[i] 	    = ap->threshRestIO;
		vIO[i]	    	    = ap->eLeakIO;
		vCoupleIO[i]	    = 0;
		pfPCPlastTimerIO[i] = 0;

		for (int j = 0; j < cp->numpIOfromNCtoIO; j++)
		{
			inputNCIO[i][j] = 0;
			gNCIO[i][j]     = 0;
		}
	}

	errDrive = 0;

	for (int i = 0; i < cp->numNC; i++)
	{
		apNC[i]    		   = 0;
		apBufNC[i] 		   = 0;
		threshNC[i]		   = ap->threshRestNC;
		vNC[i]			   = ap->eLeakNC;
		synIOPReleaseNC[i] = 0;

		for (int j = 0; j < cp->numpNCfromPCtoNC; j++)
		{
			inputPCNC[i][j]  = 0;
			gPCNC[i][j]		 = 0;
			gPCScaleNC[i][j] = ap->gIncAvgPCtoNC;
		}

		for (int j = 0; j < cp->numpNCfromMFtoNC; j++)
		{
			inputMFNC[i][j]		   = 0;
			mfSynWeightNC[i][j]    = ap->initSynWofMFtoNC;
			gmaxNMDAofMFtoNC[i][j] = 0;
			gmaxAMPAofMFtoNC[i][j] = 0;
			gMFNMDANC[i][j]		   = 0;
			gMFAMPANC[i][j]		   = 0;
		}
	}

	noLTPMFNC = 0;
	noLTDMFNC = 0;
}
