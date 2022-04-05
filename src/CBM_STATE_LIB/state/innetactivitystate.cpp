/*
 * innetactivitystate.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: consciousness
 *  Updated on: March, 2022
 * 		Author: the gallogly
 */ 

#include "state/innetactivitystate.h"

InNetActivityState::InNetActivityState
	(ConnectivityParams *conParams, ActivityParams *actParams)
{
	cp = conParams;

	allocateMemory();
	initializeVals(actParams);
}

InNetActivityState::InNetActivityState(ConnectivityParams *conParams, std::fstream &infile)
{
	cp = conParams;

	allocateMemory();

	stateRW(true, infile);
}

InNetActivityState::InNetActivityState(const InNetActivityState &state)
{
	cp = state.cp;

	allocateMemory();
	goTimeStep = 0;
	for (int i = 0; i < cp->numMF; i++)
	{
		histMF[i] = state.histMF[i];
		apBufMF[i] = state.apBufMF[i];
		depAmpMFGO[i] = state.depAmpMFGO[i];
		gi_MFtoGO[i] = state.gi_MFtoGO[i];
		
		depAmpMFGR[i] = state.depAmpMFGR[i];
		gi_MFtoGR[i] = state.gi_MFtoGR[i];
		depAmpMFUBC[i] = state.depAmpMFUBC[i];
	}

	for(int i = 0; i < cp->numGO; i++)
	{
		apGO[i] = state.apGO[i];
		spkGO[i] = state.spkGO[i];
		vGO[i] = state.vGO[i];
		vCoupleGO[i] = state.vCoupleGO[i];
		threshCurGO[i] = state.threshCurGO[i];
		inputMFGO[i] = state.inputMFGO[i];
		inputUBCGO[i] = state.inputUBCGO[i];	
		gSum_MFGO[i] = state.gSum_MFGO[i];
		inputGOGO[i] = state.inputGOGO[i];

		gi_GOtoGO[i] = state.gi_GOtoGO[i];
		depAmpGOGO[i] = state.depAmpGOGO[i];
		gSum_GOGO[i] = state.gSum_GOGO[i];

		depAmpGOGR[i] = state.depAmpGOGR[i];
		dynamicAmpGOGR[i] = state.dynamicAmpGOGR[i];

		gSum_UBCtoGO[i] = state.gSum_UBCtoGO[i];

		gMFGO[i] = state.gMFGO[i];
		gNMDAMFGO[i] = state.gNMDAMFGO[i];
		gNMDAUBCGO[i] = state.gNMDAUBCGO[i];
		gNMDAIncMFGO[i] = state.gNMDAIncMFGO[i];
		gGRGO[i] = state.gGRGO[i];
		gGRGO_NMDA[i] = state.gGRGO_NMDA[i];
		gGOGO[i] = state.gGOGO[i];
		gMGluRGO[i] = state.gMGluRGO[i];
		gMGluRIncGO[i] = state.gMGluRIncGO[i];
		mGluRGO[i] = state.mGluRGO[i];
		gluGO[i] = state.gluGO[i];
	}

	for (int i = 0; i < cp->numGR; i++)
	{
		apGR[i] = state.apGR[i];
		apBufGR[i] = state.apBufGR[i];
		
		gSum_MFGR[i] = state.gSum_MFGR[i];
		
		for (int j = 0; j < cp->maxnumpGRfromMFtoGR; j++)
		{
			gMFGR[i][j] = state.gMFGR[i][j];
			gUBCGR[i][j] = state.gUBCGR[i][j];
		}

		gMFSumGR[i] = 0;
		gMFDirectGR[i] = 0;
		gMFSpilloverGR[i] = 0;
		gGODirectGR[i] = 0;
		gGOSpilloverGR[i] = 0;
		apMFtoGR[i] = 0;
		apUBCtoGR[i] = 0;
		gUBCSumGR[i] = 0;
		gUBCDirectGR[i] = 0;
		gUBCSpilloverGR[i] = 0;	
		gNMDAGR[i] = 0;
		gNMDAIncGR[i] = 0;
		gLeakGR[i] = 0.11;
		depAmpMFtoGR[i] = 1;
		depAmpUBCtoGR[i] = 1;
		depAmpGOtoGR[i] = 1;
		dynamicAmpGOtoGR[i] = 1;
		
		for(int j = 0; j < cp->maxnumpGRfromGOtoGR; j++)
		{
			gGOGR[i][j] = state.gGOGR[i][j];
		}

		gGOSumGR[i] = state.gGOSumGR[i];
		threshGR[i] = state.threshGR[i];
		vGR[i] = state.vGR[i];
		gKCaGR[i] = state.gKCaGR[i];
		historyGR[i] = state.historyGR[i];
	}

	for (int i = 0; i < cp->numSC; i++)
	{
		apSC[i] = state.apSC[i];
		apBufSC[i] = state.apBufSC[i];
		gPFSC[i] = state.gPFSC[i];
		threshSC[i] = state.threshSC[i];
		vSC[i] = state.vSC[i];
		inputSumPFSC[i] = state.inputSumPFSC[i];
	}

	for (int i = 0; i < cp->numUBC; i++)
	{
		gRise_MFtoUBC[i] = state.gRise_MFtoUBC[i];
		gDecay_MFtoUBC[i] = state.gDecay_MFtoUBC[i];
		gSum_MFtoUBC[i] = state.gSum_MFtoUBC[i];
		depAmpUBCtoUBC[i] = state.depAmpUBCtoUBC[i];

		gRise_UBCNMDA[i] = state.gRise_UBCNMDA[i];
		gDecay_UBCNMDA[i] = state.gDecay_UBCNMDA[i];
		gSum_UBCNMDA[i] = state.gSum_UBCNMDA[i];
		gK_UBC[i] = state.gK_UBC[i];

		inputMFUBC[i] = state.inputMFUBC[i];
		
		gRise_UBCtoUBC[i] = state.gRise_UBCtoUBC[i];
		gDecay_UBCtoUBC[i] = state.gDecay_UBCtoUBC[i];
		gSumOutUBCtoUBC[i] = state.gSumOutUBCtoUBC[i];
		gSumInUBCtoUBC[i] = state.gSumInUBCtoUBC[i];
		
		inputGOUBC[i] = state.inputGOUBC[i];
		gSum_GOtoUBC[i] = state.gSum_GOtoUBC[i];
		vUBC[i] = state.vUBC[i];
	
		threshUBC[i] = state.threshUBC[i];
		apUBC[i] = state.apUBC[i];
		inputUBCtoUBC[i] = state.inputUBCtoUBC[i];
	
		gi_UBCtoGO[i] = state.gi_UBCtoGO[i];
		depAmpUBCGO[i] = state.depAmpUBCGO[i];
		depAmpUBCGR[i] = state.depAmpUBCGR[i];
	}
}

InNetActivityState::~InNetActivityState()
{
	delete[] histMF;
	delete[] apBufMF;

	delete[] apGO;
	delete[] spkGO;
	delete[] apBufGO;
	delete[] vGO;
	delete[] exGOInput;
	delete[] inhGOInput;
	delete[] vCoupleGO;
	delete[] threshCurGO;
	delete[] inputMFGO;
	delete[] inputUBCGO;
	delete[] depAmpMFGO;
	delete[] gi_MFtoGO;
	delete[] gSum_MFGO;
	delete[] inputGOGO;
	
	delete[] gi_GOtoGO;
	delete[] depAmpGOGO;
	delete[] gSum_GOGO;
	delete[] depAmpGOGR;
	delete[] dynamicAmpGOGR;
	delete[] gSum_UBCtoGO;

	delete[] vSum_GOGO;
	delete[] vSum_GRGO;
	delete[] vSum_MFGO;

	delete[] gMFGO;
	delete[] gNMDAMFGO;
	delete[] gNMDAUBCGO;
	delete[] gNMDAIncMFGO;
	delete[] gGRGO;
	delete[] gGRGO_NMDA;
	delete[] gGOGO;
	delete[] gMGluRGO;
	delete[] gMGluRIncGO;
	delete[] mGluRGO;
	delete[] gluGO;

	delete[] apGR;
	delete[] apBufGR;
	
	delete[] depAmpMFGR;
	delete[] depAmpMFUBC;
	delete[] gi_MFtoGR;
	delete[] gSum_MFGR;
	
	delete2DArray<float>(gMFGR);
	delete2DArray<float>(gUBCGR);
	delete[] gMFSumGR;
	delete[] gMFDirectGR;
	delete[] gMFSpilloverGR;
	delete[] gGODirectGR;
	delete[] gGOSpilloverGR;
	delete[] apMFtoGR;
	delete[] apUBCtoGR;
	delete[] gUBCSumGR;
	delete[] gUBCDirectGR;
	delete[] gUBCSpilloverGR;
	delete[] gNMDAGR;
	delete[] gNMDAIncGR;
	delete[] gLeakGR;
	delete[] depAmpMFtoGR;
	delete[] depAmpUBCtoGR;
	delete[] depAmpGOtoGR;
	delete[] dynamicAmpGOtoGR;
	delete2DArray<float>(gGOGR);
	delete[] gGOSumGR;
	delete[] threshGR;
	delete[] vGR;
	delete[] gKCaGR;
	delete[] historyGR;

	delete[] apSC;
	delete[] apBufSC;
	delete[] gPFSC;
	delete[] threshSC;
	delete[] vSC;
	delete[] inputSumPFSC;

	delete[] gRise_MFtoUBC;
	delete[] gDecay_MFtoUBC;
	delete[] gSum_MFtoUBC;
	delete[] depAmpUBCtoUBC;
	delete[] gRise_UBCNMDA;
	delete[] gDecay_UBCNMDA;
	delete[] gSum_UBCNMDA;
	delete[] gK_UBC;
	delete[] gRise_UBCtoUBC;
	delete[] gDecay_UBCtoUBC;
	delete[] gSumOutUBCtoUBC;
	delete[] gSumInUBCtoUBC;
	delete[] inputMFUBC;
	delete[] inputGOUBC;
	delete[] gSum_GOtoUBC;
	delete[] vUBC;
	delete[] apUBC;
	delete[] threshUBC;
	delete[] inputUBCtoUBC;	
	delete[] gi_UBCtoGO;
	delete[] depAmpUBCGO;
	delete[] depAmpUBCGR;
}

void InNetActivityState::writeState(std::fstream &outfile)
{
	stateRW(false, (std::fstream &)outfile);
}

bool InNetActivityState::operator==(const InNetActivityState &compState)
{
	bool equal = true;

	for (int i = 0; i < cp->numMF; i++)
	{
		equal = equal && (histMF[i] == compState.histMF[i]);
		equal = equal && (apBufMF[i] == compState.apBufMF[i]);
	}

	for (int i = 0; i < cp->numGO; i++)
	{
		equal = equal && (apGO[i] == compState.apGO[i]);
		equal = equal && (spkGO[i] == compState.spkGO[i]);
		equal = equal && (apBufGO[i] == compState.apBufGO[i]);
		equal = equal && (vGO[i] == compState.vGO[i]);
		equal = equal && (threshCurGO[i] == compState.threshCurGO[i]);
	}

	for (int i = 0; i < cp->numSC; i++)
	{
		equal = equal && (vSC[i] == compState.vSC[i]);
	}

	return equal;
}

bool InNetActivityState::operator!=(const InNetActivityState &compState)
{
	return !(*this == compState);
}

bool InNetActivityState::validateState()
{
	bool valid = true;

	valid = valid && validateFloatArray(vGO, cp->numGO);
	valid = valid && validateFloatArray(gMFGO, cp->numGO);
	valid = valid && validateFloatArray(gNMDAMFGO, cp->numGO);
	valid = valid && validateFloatArray(gNMDAUBCGO, cp->numGO);
	valid = valid && validateFloatArray(gNMDAIncMFGO, cp->numGO);
	valid = valid && validateFloatArray(gGRGO, cp->numGO);

	valid = valid && validate2DfloatArray(gMFGR, cp->numGR*cp->maxnumpGRfromMFtoGR);
	valid = valid && validate2DfloatArray(gUBCGR, cp->numGR*cp->maxnumpGRfromMFtoGR);
	valid = valid && validate2DfloatArray(gGOGR, cp->numGR*cp->maxnumpGRfromGOtoGR);
	valid = valid && validateFloatArray(vGR, cp->numGR);
	valid = valid && validateFloatArray(gKCaGR, cp->numGR);

	valid = valid && validateFloatArray(gPFSC, cp->numSC);
	valid = valid && validateFloatArray(vSC, cp->numSC);

	return valid;
}

void InNetActivityState::resetState(ActivityParams *ap)
{
	initializeVals(ap);
}

void InNetActivityState::allocateMemory()
{	
	int goTimeStep;
	histMF = new ct_uint8_t[cp->numMF];
	apBufMF = new ct_uint32_t[cp->numMF];

	spkGO  = new int[cp->numGO];
	goFR_HP = new float[cp->numGO];
	goSpkSumHP = new int[cp->numGO];
	synWscalerGRtoGO = new float[cp->numGO];
	synWscalerGOtoGO = new float[cp->numGO];
	apGO = new ct_uint8_t[cp->numGO];
	apBufGO = new ct_uint32_t[cp->numGO];
	vGO = new float[cp->numGO];
	exGOInput = new float[cp->numGO];
	inhGOInput = new float[cp->numGO];
	vCoupleGO = new float[cp->numGO];
	threshCurGO = new float[cp->numGO];
	inputMFGO = new ct_uint32_t[cp->numGO];
	inputUBCGO = new ct_uint32_t[cp->numGO];
	depAmpMFGO = new float[cp->numMF];
	gi_MFtoGO = new float[cp->numMF];
	gSum_MFGO = new float[cp->numGO];
	inputGOGO = new ct_uint32_t[cp->numGO];

	gi_GOtoGO = new float[cp->numGO];
	depAmpGOGO = new float[cp->numGO];
	gSum_GOGO = new float[cp->numGO];
	depAmpGOGR = new float[cp->numGO];
	dynamicAmpGOGR = new float[cp->numGO];
	
	gSum_UBCtoGO = new float[cp->numGO];

	vSum_GOGO = new float[cp->numGO];
	vSum_GRGO = new float[cp->numGO];
	vSum_MFGO = new float[cp->numGO];
	
	//todo: synaptic depression test
	inputGOGABASynDepGO = new float[cp->numGO];
	goGABAOutSynScaleGOGO = new float[cp->numGO];

	gMFGO = new float[cp->numGO];
	gNMDAMFGO = new float[cp->numGO];
	gNMDAUBCGO = new float[cp->numGO];
	gNMDAIncMFGO = new float[cp->numGO];
	gGRGO = new float[cp->numGO];
	gGRGO_NMDA = new float[cp->numGO];
	gGOGO = new float[cp->numGO];
	gMGluRGO = new float[cp->numGO];
	gMGluRIncGO = new float[cp->numGO];
	mGluRGO = new float[cp->numGO];
	gluGO = new float[cp->numGO];

	//New GR stuff
	depAmpMFGR = new float[cp->numMF];
	depAmpMFUBC = new float[cp->numMF];
	gi_MFtoGR = new float[cp->numMF];
	gSum_MFGR = new float[cp->numGR];
	
	apGR = new ct_uint8_t[cp->numGR];
	apBufGR =  new ct_uint32_t[cp->numGR];
	gMFGR = allocate2DArray<float>(cp->numGR, cp->maxnumpGRfromMFtoGR);
	gUBCGR = allocate2DArray<float>(cp->numGR, cp->maxnumpGRfromMFtoGR);
	gMFSumGR = new float[cp->numGR];
	gMFDirectGR = new float[cp->numGR];
	gMFSpilloverGR = new float[cp->numGR];	
	gGODirectGR = new float[cp->numGR];
	gGOSpilloverGR = new float[cp->numGR];	
	apMFtoGR = new int[cp->numGR];
	apUBCtoGR = new int[cp->numGR];
	gUBCSumGR = new float[cp->numGR];
	gUBCDirectGR = new float[cp->numGR];
	gUBCSpilloverGR = new float[cp->numGR];	
	gNMDAGR = new float[cp->numGR];
	gNMDAIncGR = new float[cp->numGR];
	gLeakGR = new float[cp->numGR];
	depAmpMFtoGR = new float[cp->numGR];
	depAmpUBCtoGR = new float[cp->numGR];
	depAmpGOtoGR = new float[cp->numGR];
	dynamicAmpGOtoGR = new float[cp->numGR];	
	gGOGR = allocate2DArray<float>(cp->numGR, cp->maxnumpGRfromGOtoGR);
	gGOSumGR = new float[cp->numGR];
	threshGR = new float[cp->numGR];
	vGR = new float[cp->numGR];
	gKCaGR = new float[cp->numGR];
	historyGR = new ct_uint64_t[cp->numGR];

	apSC = new ct_uint8_t[cp->numSC];
	apBufSC = new ct_uint32_t[cp->numSC];
	gPFSC = new float[cp->numSC];
	threshSC = new float[cp->numSC];
	vSC = new float[cp->numSC];
	inputSumPFSC = new ct_uint32_t[cp->numSC];

	//UBC
	gRise_MFtoUBC = new float[cp->numUBC];
	gDecay_MFtoUBC = new float[cp->numUBC];
	gSum_MFtoUBC = new float[cp->numUBC];
	depAmpUBCtoUBC = new float[cp->numUBC];
	gRise_UBCNMDA = new float[cp->numUBC];
	gDecay_UBCNMDA = new float[cp->numUBC];
	gSum_UBCNMDA = new float[cp->numUBC];
	gK_UBC = new float[cp->numUBC];
	
	gRise_UBCtoUBC = new float[cp->numUBC];
	gDecay_UBCtoUBC = new float[cp->numUBC];
	gSumOutUBCtoUBC = new float[cp->numUBC];
	gSumInUBCtoUBC = new float[cp->numUBC];
	
	inputMFUBC = new int[cp->numUBC];
	inputGOUBC = new int[cp->numUBC];
	gSum_GOtoUBC = new float[cp->numUBC];	
	vUBC = new float[cp->numUBC];	
	apUBC = new ct_uint8_t[cp->numUBC];	
	threshUBC = new float[cp->numUBC];	
	inputUBCtoUBC = new int[cp->numUBC];	

	gi_UBCtoGO = new float[cp->numUBC];
	depAmpUBCGO = new float[cp->numUBC];
	depAmpUBCGR = new float[cp->numUBC];
}

void InNetActivityState::stateRW(bool read, std::fstream &file)
{
	rawBytesRW((char *)histMF, cp->numMF*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufMF, cp->numMF*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)spkGO, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)apGO, cp->numGO*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGO, cp->numGO*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)vGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)vCoupleGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)threshCurGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)inputMFGO, cp->numGO*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputUBCGO, cp->numGO*sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)depAmpMFGO, cp->numMF*sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGO, cp->numMF*sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)inputGOGO, cp->numGO*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gi_GOtoGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gSum_GOGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGR, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOGR, cp->numGO*sizeof(float), read, file);
	
	rawBytesRW((char *)gSum_UBCtoGO, cp->numGO*sizeof(float), read, file);

	rawBytesRW((char *)depAmpMFGR, cp->numMF*sizeof(float), read, file);
	rawBytesRW((char *)depAmpMFUBC, cp->numMF*sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGR, cp->numMF*sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGR, cp->numGR*sizeof(float), read, file);

	rawBytesRW((char *)inputGOGABASynDepGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)goGABAOutSynScaleGOGO, cp->numGO*sizeof(float), read, file);

	rawBytesRW((char *)gMFGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gNMDAUBCGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gNMDAMFGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncMFGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gGRGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gGRGO_NMDA, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gGOGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gMGluRGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gMGluRIncGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)mGluRGO, cp->numGO*sizeof(float), read, file);
	rawBytesRW((char *)gluGO, cp->numGO*sizeof(float), read, file);

	rawBytesRW((char *)apGR, cp->numGR*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGR, cp->numGR*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gMFGR[0], cp->numGR*cp->maxnumpGRfromMFtoGR*sizeof(float), read, file);
	rawBytesRW((char *)gUBCGR[0], cp->numGR*cp->maxnumpGRfromMFtoGR*sizeof(float), read, file);
	rawBytesRW((char *)gMFSumGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gMFDirectGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gMFSpilloverGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gGODirectGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gGOSpilloverGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)apMFtoGR, cp->numGR*sizeof(int), read, file);
	rawBytesRW((char *)apUBCtoGR, cp->numGR*sizeof(int), read, file);
	rawBytesRW((char *)gUBCSumGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gUBCDirectGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gUBCSpilloverGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gNMDAGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gLeakGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)depAmpMFtoGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCtoGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOtoGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOtoGR, cp->numGR*sizeof(float), read, file);

	rawBytesRW((char *)gGOGR[0], cp->numGR*cp->maxnumpGRfromGOtoGR*sizeof(float), read, file);
	rawBytesRW((char *)gGOSumGR, cp->numGR*sizeof(float), read, file);

	rawBytesRW((char *)threshGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)vGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)gKCaGR, cp->numGR*sizeof(float), read, file);
	rawBytesRW((char *)historyGR, cp->numGR*sizeof(ct_uint64_t), read, file);

	rawBytesRW((char *)apSC, cp->numSC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufSC, cp->numSC*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gPFSC, cp->numSC*sizeof(float), read, file);
	rawBytesRW((char *)threshSC, cp->numSC*sizeof(float), read, file);
	rawBytesRW((char *)vSC, cp->numSC*sizeof(float), read, file);

	rawBytesRW((char *)inputSumPFSC, cp->numSC*sizeof(ct_uint32_t), read, file);

	//UBCs
	rawBytesRW((char *)gRise_MFtoUBC, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)gDecay_MFtoUBC, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFtoUBC, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCtoUBC, cp->numUBC*sizeof(float), read, file);
	
	rawBytesRW((char *)gRise_UBCNMDA, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)gDecay_UBCNMDA, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)gSum_UBCNMDA, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)gK_UBC, cp->numUBC*sizeof(float), read, file);
	
	rawBytesRW((char *)gRise_UBCtoUBC, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)gDecay_UBCtoUBC, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)gSumOutUBCtoUBC, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)gSumInUBCtoUBC, cp->numUBC*sizeof(float), read, file);
	
	rawBytesRW((char *)inputMFUBC, cp->numUBC*sizeof(int), read, file);
	rawBytesRW((char *)inputGOUBC, cp->numUBC*sizeof(int), read, file);
	rawBytesRW((char *)gSum_GOtoUBC, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)vUBC, cp->numUBC*sizeof(float), read, file);
	
	rawBytesRW((char *)apUBC, cp->numUBC*sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)threshUBC, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)inputUBCtoUBC, cp->numUBC*sizeof(int), read, file);
	
	rawBytesRW((char *)gi_UBCtoGO, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCGO, cp->numUBC*sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCGR, cp->numUBC*sizeof(float), read, file);
}

void InNetActivityState::initializeVals(ActivityParams *ap)
{
	// TODO: rewrite so that we arent looping here: leverage the STL!	
	goTimeStep = 0;
	for (int i = 0; i < cp->numMF; i++)
	{
		histMF[i] = false;
		apBufMF[i] = 0;
		depAmpMFGO[i] = 1;
		gi_MFtoGO[i] = 0;
		
		depAmpMFGR[i] = 1;
		depAmpMFUBC[i] = 1;	
		gi_MFtoGR[i] = 0;
	
	}

	for (int i = 0; i < cp->numGO; i++)
	{
		goSpkSumHP[i] = 0;
		synWscalerGRtoGO[i] = 1;
		synWscalerGOtoGO[i] = 1;
		goFR_HP[i] = 0;
		apGO[i] = false;
		spkGO[i] = 0;
		apBufGO[i] = 0;
		vGO[i] = ap->eLeakGO;
		exGOInput[i] = 0;
		inhGOInput[i] = 0;
		vCoupleGO[i] = 0;
		threshCurGO[i] = ap->threshRestGO;
		inputMFGO[i] = 0;
		inputUBCGO[i] = 0;
		gSum_MFGO[i] = 0;	
		inputGOGO[i] = 0;

		gi_GOtoGO[i] = 0;
		depAmpGOGO[i] = 0;
		gSum_GOGO[i] = 0;
		depAmpGOGR[i] = 0;
		dynamicAmpGOGR[i] = 0;

		gSum_UBCtoGO[i] = 0;
		
		vSum_GOGO[i] = ap->eLeakGO;
		vSum_GRGO[i] = ap->eLeakGO;
		vSum_MFGO[i] = ap->eLeakGO;
		
		inputGOGABASynDepGO[i] = 0;
		goGABAOutSynScaleGOGO[i] = 1;

		gMFGO[i] = 0;
		gNMDAUBCGO[i] = 0;	
		gNMDAMFGO[i] = 0;	
		gNMDAIncMFGO[i] = 0;
		gGRGO[i] = 0;
		gGRGO_NMDA[i] = 0;
		gGOGO[i] = 0;
		gMGluRGO[i] = 0;
		gMGluRIncGO[i] = 0;
		mGluRGO[i] = 0;
		gluGO[i] = 0;
	}

	for (int i = 0; i < cp->numGR; i++)
	{
		apGR[i] = false;
		apBufGR[i] = 0;
		
		gSum_MFGR[i] = 0;	
		
		for (int j = 0; j < cp->maxnumpGRfromMFtoGR; j++)
		{
			gMFGR[i][j] = 0;
			gUBCGR[i][j] = 0;
		}

		gMFSumGR[i] = 0;
		gMFDirectGR[i] = 0;
		gMFSpilloverGR[i] = 0;
		gGODirectGR[i] = 0;
		gGOSpilloverGR[i] = 0;
		apMFtoGR[i] = 0;
		apUBCtoGR[i] = 0;
		gUBCSumGR[i] = 0;
		gUBCDirectGR[i] = 0;
		gUBCSpilloverGR[i] = 0;
		gNMDAGR[i] = 0;
		gNMDAIncGR[i] = 0;
		gLeakGR[i]  =  0.11;	
		depAmpMFtoGR[i]  =  1;	
		depAmpUBCtoGR[i]  =  1;	
		depAmpGOtoGR[i]  =  1;	
		dynamicAmpGOtoGR[i]  =  0;

		for (int j = 0; j < cp->maxnumpGRfromGOtoGR; j++)
		{
			gGOGR[i][j] = 0;
		}	

		gGOSumGR[i] = 0;
		threshGR[i] = ap->threshRestGR;
		vGR[i] = ap->eLeakGR;
		gKCaGR[i] = 0;
		historyGR[i] = 0;
	}


	for (int i = 0; i < cp->numSC; i++)
	{
		apSC[i] = false;
		apBufSC[i] = 0;
		gPFSC[i] = 0;
		threshSC[i] = ap->threshRestSC;
		vSC[i] = ap->eLeakSC;
		inputSumPFSC[i] = 0;
	}

	for (int i = 0; i < cp->numUBC; i++)
	{
		gRise_MFtoUBC[i] = 0;
		gDecay_MFtoUBC[i] = 0;
		
		gRise_UBCNMDA[i] = 0;
		gDecay_UBCNMDA[i] = 0;
		gSum_UBCNMDA[i] = 0;
		gK_UBC[i] = 0;
		
		gRise_UBCtoUBC[i] = 0;
		gDecay_UBCtoUBC[i] = 0;
		gSumOutUBCtoUBC[i] = 0;
		gSumInUBCtoUBC[i] = 0;
		
		inputMFUBC[i] = 0;
		inputGOUBC[i] = 0;
		gSum_MFtoUBC[i] = 0;
		depAmpUBCtoUBC[i] = 1;
		vUBC[i] = -70;
	
		threshUBC[i] = ap->threshRestGO;
		apUBC[i] = false;
		inputUBCtoUBC[i] = false;

		gi_UBCtoGO[i] = 0;
		depAmpUBCGO[i] = 1;
		depAmpUBCGR[i] = 1;
	}
}

