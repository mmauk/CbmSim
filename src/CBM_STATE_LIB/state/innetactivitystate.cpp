/*
 * innetactivitystate.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: consciousness
 *  Updated on: March, 2022
 * 		Author: the gallogly
 */ 

#include "state/innetactivitystate.h"

InNetActivityState::InNetActivityState(ActivityParams *ap)
{
	initializeVals(ap);
}

InNetActivityState::InNetActivityState(std::fstream &infile)
{
	stateRW(true, infile);
}

//InNetActivityState::InNetActivityState(const InNetActivityState &state)
//{
//	cp = state.cp;
//
//	allocateMemory();
//	goTimeStep = 0;
//	for (int i = 0; i < numMF; i++)
//	{
//		histMF[i] = state.histMF[i];
//		apBufMF[i] = state.apBufMF[i];
//		depAmpMFGO[i] = state.depAmpMFGO[i];
//		gi_MFtoGO[i] = state.gi_MFtoGO[i];
//		
//		depAmpMFGR[i] = state.depAmpMFGR[i];
//		gi_MFtoGR[i] = state.gi_MFtoGR[i];
//		depAmpMFUBC[i] = state.depAmpMFUBC[i];
//	}
//
//	for(int i = 0; i < numGO; i++)
//	{
//		apGO[i] = state.apGO[i];
//		spkGO[i] = state.spkGO[i];
//		vGO[i] = state.vGO[i];
//		vCoupleGO[i] = state.vCoupleGO[i];
//		threshCurGO[i] = state.threshCurGO[i];
//		inputMFGO[i] = state.inputMFGO[i];
//		inputUBCGO[i] = state.inputUBCGO[i];	
//		gSum_MFGO[i] = state.gSum_MFGO[i];
//		inputGOGO[i] = state.inputGOGO[i];
//
//		gi_GOtoGO[i] = state.gi_GOtoGO[i];
//		depAmpGOGO[i] = state.depAmpGOGO[i];
//		gSum_GOGO[i] = state.gSum_GOGO[i];
//
//		depAmpGOGR[i] = state.depAmpGOGR[i];
//		dynamicAmpGOGR[i] = state.dynamicAmpGOGR[i];
//
//		gSum_UBCtoGO[i] = state.gSum_UBCtoGO[i];
//
//		gMFGO[i] = state.gMFGO[i];
//		gNMDAMFGO[i] = state.gNMDAMFGO[i];
//		gNMDAUBCGO[i] = state.gNMDAUBCGO[i];
//		gNMDAIncMFGO[i] = state.gNMDAIncMFGO[i];
//		gGRGO[i] = state.gGRGO[i];
//		gGRGO_NMDA[i] = state.gGRGO_NMDA[i];
//		gGOGO[i] = state.gGOGO[i];
//		gMGluRGO[i] = state.gMGluRGO[i];
//		gMGluRIncGO[i] = state.gMGluRIncGO[i];
//		mGluRGO[i] = state.mGluRGO[i];
//		gluGO[i] = state.gluGO[i];
//	}
//
//	for (int i = 0; i < numGR; i++)
//	{
//		apGR[i] = state.apGR[i];
//		apBufGR[i] = state.apBufGR[i];
//		
//		gSum_MFGR[i] = state.gSum_MFGR[i];
//		
//		for (int j = 0; j < maxnumpGRfromMFtoGR; j++)
//		{
//			gMFGR[i][j] = state.gMFGR[i][j];
//			gUBCGR[i][j] = state.gUBCGR[i][j];
//		}
//
//		gMFSumGR[i] = 0;
//		gMFDirectGR[i] = 0;
//		gMFSpilloverGR[i] = 0;
//		gGODirectGR[i] = 0;
//		gGOSpilloverGR[i] = 0;
//		apMFtoGR[i] = 0;
//		apUBCtoGR[i] = 0;
//		gUBCSumGR[i] = 0;
//		gUBCDirectGR[i] = 0;
//		gUBCSpilloverGR[i] = 0;	
//		gNMDAGR[i] = 0;
//		gNMDAIncGR[i] = 0;
//		gLeakGR[i] = 0.11;
//		depAmpMFtoGR[i] = 1;
//		depAmpUBCtoGR[i] = 1;
//		depAmpGOtoGR[i] = 1;
//		dynamicAmpGOtoGR[i] = 1;
//		
//		for(int j = 0; j < maxnumpGRfromGOtoGR; j++)
//		{
//			gGOGR[i][j] = state.gGOGR[i][j];
//		}
//
//		gGOSumGR[i] = state.gGOSumGR[i];
//		threshGR[i] = state.threshGR[i];
//		vGR[i] = state.vGR[i];
//		gKCaGR[i] = state.gKCaGR[i];
//		historyGR[i] = state.historyGR[i];
//	}
//
//	for (int i = 0; i < numSC; i++)
//	{
//		apSC[i] = state.apSC[i];
//		apBufSC[i] = state.apBufSC[i];
//		gPFSC[i] = state.gPFSC[i];
//		threshSC[i] = state.threshSC[i];
//		vSC[i] = state.vSC[i];
//		inputSumPFSC[i] = state.inputSumPFSC[i];
//	}
//
//	for (int i = 0; i < numUBC; i++)
//	{
//		gRise_MFtoUBC[i] = state.gRise_MFtoUBC[i];
//		gDecay_MFtoUBC[i] = state.gDecay_MFtoUBC[i];
//		gSum_MFtoUBC[i] = state.gSum_MFtoUBC[i];
//		depAmpUBCtoUBC[i] = state.depAmpUBCtoUBC[i];
//
//		gRise_UBCNMDA[i] = state.gRise_UBCNMDA[i];
//		gDecay_UBCNMDA[i] = state.gDecay_UBCNMDA[i];
//		gSum_UBCNMDA[i] = state.gSum_UBCNMDA[i];
//		gK_UBC[i] = state.gK_UBC[i];
//
//		inputMFUBC[i] = state.inputMFUBC[i];
//		
//		gRise_UBCtoUBC[i] = state.gRise_UBCtoUBC[i];
//		gDecay_UBCtoUBC[i] = state.gDecay_UBCtoUBC[i];
//		gSumOutUBCtoUBC[i] = state.gSumOutUBCtoUBC[i];
//		gSumInUBCtoUBC[i] = state.gSumInUBCtoUBC[i];
//		
//		inputGOUBC[i] = state.inputGOUBC[i];
//		gSum_GOtoUBC[i] = state.gSum_GOtoUBC[i];
//		vUBC[i] = state.vUBC[i];
//	
//		threshUBC[i] = state.threshUBC[i];
//		apUBC[i] = state.apUBC[i];
//		inputUBCtoUBC[i] = state.inputUBCtoUBC[i];
//	
//		gi_UBCtoGO[i] = state.gi_UBCtoGO[i];
//		depAmpUBCGO[i] = state.depAmpUBCGO[i];
//		depAmpUBCGR[i] = state.depAmpUBCGR[i];
//	}
//}

InNetActivityState::~InNetActivityState() {}

void InNetActivityState::writeState(std::fstream &outfile)
{
	stateRW(false, outfile);
}

bool InNetActivityState::state_equal(const InNetActivityState &compState)
{
	bool equal = true;

	for (int i = 0; i < NUM_MF; i++)
	{
		equal = equal && (histMF[i] == compState.histMF[i]);
		equal = equal && (apBufMF[i] == compState.apBufMF[i]);
	}

	for (int i = 0; i < NUM_GO; i++)
	{
		equal = equal && (apGO[i] == compState.apGO[i]);
		equal = equal && (spkGO[i] == compState.spkGO[i]);
		equal = equal && (apBufGO[i] == compState.apBufGO[i]);
		equal = equal && (vGO[i] == compState.vGO[i]);
		equal = equal && (threshCurGO[i] == compState.threshCurGO[i]);
	}

	for (int i = 0; i < NUM_SC; i++)
	{
		equal = equal && (vSC[i] == compState.vSC[i]);
	}

	return equal;
}

bool InNetActivityState::state_unequal(const InNetActivityState &compState)
{
	return !state_equal(compState);
}

//bool InNetActivityState::validateState()
//{
//	bool valid = true;
//
//	valid = valid && validateFloatArray(vGO, numGO);
//	valid = valid && validateFloatArray(gMFGO, numGO);
//	valid = valid && validateFloatArray(gNMDAMFGO, numGO);
//	valid = valid && validateFloatArray(gNMDAUBCGO, numGO);
//	valid = valid && validateFloatArray(gNMDAIncMFGO, numGO);
//	valid = valid && validateFloatArray(gGRGO, numGO);
//
//	valid = valid && validate2DfloatArray(gMFGR, numGR*maxnumpGRfromMFtoGR);
//	valid = valid && validate2DfloatArray(gUBCGR, numGR*maxnumpGRfromMFtoGR);
//	valid = valid && validate2DfloatArray(gGOGR, numGR*maxnumpGRfromGOtoGR);
//	valid = valid && validateFloatArray(vGR, numGR);
//	valid = valid && validateFloatArray(gKCaGR, numGR);
//
//	valid = valid && validateFloatArray(gPFSC, numSC);
//	valid = valid && validateFloatArray(vSC, numSC);
//
//	return valid;
//}

void InNetActivityState::resetState(ActivityParams *ap)
{
	initializeVals(ap);
}

//void InNetActivityState::allocateMemory()
//{	
//	//histMF = new ct_uint8_t[numMF];
//	//apBufMF = new ct_uint32_t[numMF];
//
//	//spkGO  = new int[numGO];
//	//goFR_HP = new float[numGO];
//	//goSpkSumHP = new int[numGO];
//	//synWscalerGRtoGO = new float[numGO];
//	//synWscalerGOtoGO = new float[numGO];
//	//apGO = new ct_uint8_t[numGO];
//	//apBufGO = new ct_uint32_t[numGO];
//	//vGO = new float[numGO];
//	//exGOInput = new float[numGO];
//	//inhGOInput = new float[numGO];
//	//vCoupleGO = new float[numGO];
//	//threshCurGO = new float[numGO];
//	//inputMFGO = new ct_uint32_t[numGO];
//	//inputUBCGO = new ct_uint32_t[numGO];
//	//depAmpMFGO = new float[numMF];
//	//gi_MFtoGO = new float[numMF];
//	//gSum_MFGO = new float[numGO];
//	//inputGOGO = new ct_uint32_t[numGO];
//
//	//gi_GOtoGO = new float[numGO];
//	//depAmpGOGO = new float[numGO];
//	//gSum_GOGO = new float[numGO];
//	//depAmpGOGR = new float[numGO];
//	//dynamicAmpGOGR = new float[numGO];
//	//
//	//gSum_UBCtoGO = new float[numGO];
//
//	//vSum_GOGO = new float[numGO];
//	//vSum_GRGO = new float[numGO];
//	//vSum_MFGO = new float[numGO];
//	
//	//todo: synaptic depression test
//	//inputGOGABASynDepGO = new float[numGO];
//	//goGABAOutSynScaleGOGO = new float[numGO];
//
//	//gMFGO = new float[numGO];
//	//gNMDAMFGO = new float[numGO];
//	//gNMDAUBCGO = new float[numGO];
//	//gNMDAIncMFGO = new float[numGO];
//	//gGRGO = new float[numGO];
//	//gGRGO_NMDA = new float[numGO];
//	//gGOGO = new float[numGO];
//	//gMGluRGO = new float[numGO];
//	//gMGluRIncGO = new float[numGO];
//	//mGluRGO = new float[numGO];
//	//gluGO = new float[numGO];
//
//	//New GR stuff
//	//depAmpMFGR = new float[numMF];
//	////depAmpMFUBC = new float[numMF];
//	//gi_MFtoGR = new float[numMF];
//	//gSum_MFGR = new float[numGR];
//
//	//apGR = new ct_uint8_t[numGR];
//	//apBufGR =  new ct_uint32_t[numGR];
//	//gMFGR = allocate2DArray<float>(numGR, maxnumpGRfromMFtoGR);
//	//gUBCGR = allocate2DArray<float>(numGR, maxnumpGRfromMFtoGR);
//	//gMFSumGR = new float[numGR];
//	//gMFDirectGR = new float[numGR];
//	//gMFSpilloverGR = new float[numGR];	
//	//gGODirectGR = new float[numGR];
//	//gGOSpilloverGR = new float[numGR];	
//	//apMFtoGR = new int[numGR];
//	//apUBCtoGR = new int[numGR];
//	//gUBCSumGR = new float[numGR];
//	//gUBCDirectGR = new float[numGR];
//	//gUBCSpilloverGR = new float[numGR];	
//	//gNMDAGR = new float[numGR];
//	//gNMDAIncGR = new float[numGR];
//	//gLeakGR = new float[numGR];
//	//depAmpMFtoGR = new float[numGR];
//	//depAmpUBCtoGR = new float[numGR];
//	//depAmpGOtoGR = new float[numGR];
//	//dynamicAmpGOtoGR = new float[numGR];	
//	//gGOGR = allocate2DArray<float>(numGR, maxnumpGRfromGOtoGR);
//	//gGOSumGR = new float[numGR];
//	//threshGR = new float[numGR];
//	//vGR = new float[numGR];
//	//gKCaGR = new float[numGR];
//	//historyGR = new ct_uint64_t[numGR];
//
//	// sc
//	//apSC = new ct_uint8_t[numSC];
//	//apBufSC = new ct_uint32_t[numSC];
//	//gPFSC = new float[numSC];
//	//threshSC = new float[numSC];
//	//vSC = new float[numSC];
//	//inputSumPFSC = new ct_uint32_t[numSC];
//
//	//UBC
//	//gRise_MFtoUBC = new float[numUBC];
//	//gDecay_MFtoUBC = new float[numUBC];
//	//gSum_MFtoUBC = new float[numUBC];
//	//depAmpUBCtoUBC = new float[numUBC];
//	//gRise_UBCNMDA = new float[numUBC];
//	//gDecay_UBCNMDA = new float[numUBC];
//	//gSum_UBCNMDA = new float[numUBC];
//	//gK_UBC = new float[numUBC];
//	//
//	//gRise_UBCtoUBC = new float[numUBC];
//	//gDecay_UBCtoUBC = new float[numUBC];
//	//gSumOutUBCtoUBC = new float[numUBC];
//	//gSumInUBCtoUBC = new float[numUBC];
//	//
//	//inputMFUBC = new int[numUBC];
//	//inputGOUBC = new int[numUBC];
//	//gSum_GOtoUBC = new float[numUBC];	
//	//vUBC = new float[numUBC];	
//	//apUBC = new ct_uint8_t[numUBC];	
//	//threshUBC = new float[numUBC];	
//	//inputUBCtoUBC = new int[numUBC];	
//
//	//gi_UBCtoGO = new float[numUBC];
//	//depAmpUBCGO = new float[numUBC];
//	//depAmpUBCGR = new float[numUBC];
//}

void InNetActivityState::stateRW(bool read, std::fstream &file)
{
	rawBytesRW((char *)histMF, NUM_MF * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufMF, NUM_MF * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)spkGO, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)apGO, NUM_GO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGO, NUM_GO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)vGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)threshCurGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)inputMFGO, NUM_GO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputUBCGO, NUM_GO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)depAmpMFGO, NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGO, NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)inputGOGO, NUM_GO * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gi_GOtoGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gSum_GOGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGR, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOGR, NUM_GO * sizeof(float), read, file);
	
	rawBytesRW((char *)gSum_UBCtoGO, NUM_GO * sizeof(float), read, file);

	rawBytesRW((char *)depAmpMFGR, NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)depAmpMFUBC, NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGR, NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGR, NUM_GR * sizeof(float), read, file);

	rawBytesRW((char *)inputGOGABASynDepGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)goGABAOutSynScaleGOGO, NUM_GO * sizeof(float), read, file);

	rawBytesRW((char *)gMFGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAUBCGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAMFGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncMFGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO_NMDA, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGOGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gMGluRGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gMGluRIncGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)mGluRGO, NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gluGO, NUM_GO * sizeof(float), read, file);

	rawBytesRW((char *)apGR, NUM_GR * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGR, NUM_GR * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gMFGR[0], NUM_GR * MAX_NUM_P_GR_FROM_MF_TO_GR * sizeof(float), read, file);
	rawBytesRW((char *)gUBCGR[0], NUM_GR * MAX_NUM_P_GR_FROM_MF_TO_GR * sizeof(float), read, file);
	rawBytesRW((char *)gMFSumGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gMFDirectGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gMFSpilloverGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gGODirectGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gGOSpilloverGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)apMFtoGR, NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)apUBCtoGR, NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)gUBCSumGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gUBCDirectGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gUBCSpilloverGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gLeakGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)depAmpMFtoGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCtoGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOtoGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOtoGR, NUM_GR * sizeof(float), read, file);

	rawBytesRW((char *)gGOGR[0], NUM_GR * MAX_NUM_P_GR_FROM_GO_TO_GR * sizeof(float), read, file);
	rawBytesRW((char *)gGOSumGR, NUM_GR * sizeof(float), read, file);

	rawBytesRW((char *)threshGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)vGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gKCaGR, NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)historyGR, NUM_GR * sizeof(ct_uint64_t), read, file);

	rawBytesRW((char *)apSC, NUM_SC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufSC, NUM_SC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gPFSC, NUM_SC * sizeof(float), read, file);
	rawBytesRW((char *)threshSC, NUM_SC * sizeof(float), read, file);
	rawBytesRW((char *)vSC, NUM_SC * sizeof(float), read, file);

	rawBytesRW((char *)inputSumPFSC, NUM_SC * sizeof(ct_uint32_t), read, file);

	//UBCs
	rawBytesRW((char *)gRise_MFtoUBC, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gDecay_MFtoUBC, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFtoUBC, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCtoUBC, NUM_UBC * sizeof(float), read, file);
	
	rawBytesRW((char *)gRise_UBCNMDA, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gDecay_UBCNMDA, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gSum_UBCNMDA, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gK_UBC, NUM_UBC * sizeof(float), read, file);
	
	rawBytesRW((char *)gRise_UBCtoUBC, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gDecay_UBCtoUBC, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gSumOutUBCtoUBC, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gSumInUBCtoUBC, NUM_UBC * sizeof(float), read, file);
	
	rawBytesRW((char *)inputMFUBC, NUM_UBC * sizeof(int), read, file);
	rawBytesRW((char *)inputGOUBC, NUM_UBC * sizeof(int), read, file);
	rawBytesRW((char *)gSum_GOtoUBC, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)vUBC, NUM_UBC * sizeof(float), read, file);
	
	rawBytesRW((char *)apUBC, NUM_UBC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)threshUBC, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)inputUBCtoUBC, NUM_UBC * sizeof(int), read, file);
	
	rawBytesRW((char *)gi_UBCtoGO, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCGO, NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCGR, NUM_UBC * sizeof(float), read, file);
}

void InNetActivityState::initializeVals(ActivityParams *ap)
{
	// only actively initializing those arrays whose initial values we want
	// differ from the default initilizer value
	goTimeStep = 0;

	// mf
	std::fill(histMF, histMF + NUM_MF, false);
	std::fill(depAmpMFGO, depAmpMFGO + NUM_MF, 1.0);
	std::fill(depAmpMFGR, depAmpMFGR + NUM_MF, 1.0);
	std::fill(depAmpMFUBC, depAmpMFUBC + NUM_MF, 1.0);

	// go	
	std::fill(synWscalerGRtoGO, synWscalerGRtoGO + NUM_GO, 1.0);
	std::fill(synWscalerGOtoGO, synWscalerGOtoGO + NUM_GO, 1.0);
	std::fill(apGO, apGO + NUM_GO, false);
	std::fill(vGO, vGO + NUM_GO, ap->eLeakGO);
	std::fill(threshCurGO, threshCurGO + NUM_GO, ap->threshRestGO);
	std::fill(vSum_GOGO, vSum_GOGO + NUM_GO, ap->eLeakGO);
	std::fill(vSum_GRGO, vSum_GRGO + NUM_GO, ap->eLeakGO);
	std::fill(vSum_MFGO, vSum_MFGO + NUM_GO, ap->eLeakGO);
	std::fill(goGABAOutSynScaleGOGO, goGABAOutSynScaleGOGO + NUM_GO, 1.0);

	// gr
	std::fill(apGR, apGR + NUM_GR, false);
	std::fill(gLeakGR, gLeakGR + NUM_GR, 0.11);
	std::fill(depAmpMFtoGR, depAmpMFtoGR + NUM_GR, 1.0);
	std::fill(depAmpUBCtoGR, depAmpUBCtoGR + NUM_GR, 1.0);
	std::fill(depAmpGOtoGR, depAmpGOtoGR + NUM_GR, 1.0);
	std::fill(threshGR, threshGR + NUM_GR, ap->threshRestGR);
	std::fill(vGR, vGR + NUM_GR, ap->eLeakGR);

	// sc
	std::fill(apSC, apSC + NUM_SC, false);	
	std::fill(threshSC, threshSC + NUM_SC, ap->threshRestSC);
	std::fill(apSC, apSC + NUM_SC, false);
	std::fill(threshSC, threshSC + NUM_SC, ap->threshRestSC);
	std::fill(vSC, vSC + NUM_SC, ap->eLeakSC);

	// ubc
	std::fill(depAmpUBCtoUBC, depAmpUBCtoUBC + NUM_UBC, 1.0);
	std::fill(vUBC, vUBC + NUM_UBC, -70.0);
	std::fill(threshUBC, threshUBC + NUM_UBC, ap->threshRestGO);
	std::fill(apUBC, apUBC + NUM_UBC, false);
	std::fill(inputUBCtoUBC, inputUBCtoUBC + NUM_UBC, false);
	std::fill(depAmpUBCGO, depAmpUBCGO + NUM_UBC, 1.0);
	std::fill(depAmpUBCGR, depAmpUBCGR + NUM_UBC, 1.0);
}

