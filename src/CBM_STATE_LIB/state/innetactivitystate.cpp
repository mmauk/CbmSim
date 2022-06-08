/*
 * innetactivitystate.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: consciousness
 *  Updated on: March, 2022
 * 		Author: the gallogly
 */ 

#include "state/innetactivitystate.h"

InNetActivityState::InNetActivityState(ConnectivityParams &cp, ActivityParams *ap)
{
	initializeVals(cp, ap);
}

InNetActivityState::InNetActivityState(ConnectivityParams &cp, std::fstream &infile)
{
	stateRW(cp, true, infile);
}

//InNetActivityState::InNetActivityState(const InNetActivityState &state)
//{
//	cp = state.cp;
//
//	allocateMemory();
//	goTimeStep = 0;
//	for (int i = 0; i < cp->numMF; i++)
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
//	for(int i = 0; i < cp->numGO; i++)
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
//	for (int i = 0; i < cp->numGR; i++)
//	{
//		apGR[i] = state.apGR[i];
//		apBufGR[i] = state.apBufGR[i];
//		
//		gSum_MFGR[i] = state.gSum_MFGR[i];
//		
//		for (int j = 0; j < cp->maxnumpGRfromMFtoGR; j++)
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
//		for(int j = 0; j < cp->maxnumpGRfromGOtoGR; j++)
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
//	for (int i = 0; i < cp->numSC; i++)
//	{
//		apSC[i] = state.apSC[i];
//		apBufSC[i] = state.apBufSC[i];
//		gPFSC[i] = state.gPFSC[i];
//		threshSC[i] = state.threshSC[i];
//		vSC[i] = state.vSC[i];
//		inputSumPFSC[i] = state.inputSumPFSC[i];
//	}
//
//	for (int i = 0; i < cp->numUBC; i++)
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

void InNetActivityState::writeState(ConnectivityParams &cp, std::fstream &outfile)
{
	stateRW(cp, false, outfile);
}

bool InNetActivityState::state_equal(ConnectivityParams &cp, const InNetActivityState &compState)
{
	bool equal = true;

	for (int i = 0; i < cp.NUM_MF; i++)
	{
		equal = equal && (histMF[i] == compState.histMF[i]);
		equal = equal && (apBufMF[i] == compState.apBufMF[i]);
	}

	for (int i = 0; i < cp.NUM_GO; i++)
	{
		equal = equal && (apGO[i] == compState.apGO[i]);
		equal = equal && (spkGO[i] == compState.spkGO[i]);
		equal = equal && (apBufGO[i] == compState.apBufGO[i]);
		equal = equal && (vGO[i] == compState.vGO[i]);
		equal = equal && (threshCurGO[i] == compState.threshCurGO[i]);
	}

	for (int i = 0; i < cp.NUM_SC; i++)
	{
		equal = equal && (vSC[i] == compState.vSC[i]);
	}

	return equal;
}

bool InNetActivityState::state_unequal(ConnectivityParams &cp, const InNetActivityState &compState)
{
	return !state_equal(cp, compstate);
}

//bool InNetActivityState::validateState()
//{
//	bool valid = true;
//
//	valid = valid && validateFloatArray(vGO, cp->numGO);
//	valid = valid && validateFloatArray(gMFGO, cp->numGO);
//	valid = valid && validateFloatArray(gNMDAMFGO, cp->numGO);
//	valid = valid && validateFloatArray(gNMDAUBCGO, cp->numGO);
//	valid = valid && validateFloatArray(gNMDAIncMFGO, cp->numGO);
//	valid = valid && validateFloatArray(gGRGO, cp->numGO);
//
//	valid = valid && validate2DfloatArray(gMFGR, cp->numGR*cp->maxnumpGRfromMFtoGR);
//	valid = valid && validate2DfloatArray(gUBCGR, cp->numGR*cp->maxnumpGRfromMFtoGR);
//	valid = valid && validate2DfloatArray(gGOGR, cp->numGR*cp->maxnumpGRfromGOtoGR);
//	valid = valid && validateFloatArray(vGR, cp->numGR);
//	valid = valid && validateFloatArray(gKCaGR, cp->numGR);
//
//	valid = valid && validateFloatArray(gPFSC, cp->numSC);
//	valid = valid && validateFloatArray(vSC, cp->numSC);
//
//	return valid;
//}

void InNetActivityState::resetState(ConnectivityParams &cp, ActivityParams *ap)
{
	initializeVals(cp, ap);
}

//void InNetActivityState::allocateMemory()
//{	
//	//histMF = new ct_uint8_t[cp->numMF];
//	//apBufMF = new ct_uint32_t[cp->numMF];
//
//	//spkGO  = new int[cp->numGO];
//	//goFR_HP = new float[cp->numGO];
//	//goSpkSumHP = new int[cp->numGO];
//	//synWscalerGRtoGO = new float[cp->numGO];
//	//synWscalerGOtoGO = new float[cp->numGO];
//	//apGO = new ct_uint8_t[cp->numGO];
//	//apBufGO = new ct_uint32_t[cp->numGO];
//	//vGO = new float[cp->numGO];
//	//exGOInput = new float[cp->numGO];
//	//inhGOInput = new float[cp->numGO];
//	//vCoupleGO = new float[cp->numGO];
//	//threshCurGO = new float[cp->numGO];
//	//inputMFGO = new ct_uint32_t[cp->numGO];
//	//inputUBCGO = new ct_uint32_t[cp->numGO];
//	//depAmpMFGO = new float[cp->numMF];
//	//gi_MFtoGO = new float[cp->numMF];
//	//gSum_MFGO = new float[cp->numGO];
//	//inputGOGO = new ct_uint32_t[cp->numGO];
//
//	//gi_GOtoGO = new float[cp->numGO];
//	//depAmpGOGO = new float[cp->numGO];
//	//gSum_GOGO = new float[cp->numGO];
//	//depAmpGOGR = new float[cp->numGO];
//	//dynamicAmpGOGR = new float[cp->numGO];
//	//
//	//gSum_UBCtoGO = new float[cp->numGO];
//
//	//vSum_GOGO = new float[cp->numGO];
//	//vSum_GRGO = new float[cp->numGO];
//	//vSum_MFGO = new float[cp->numGO];
//	
//	//todo: synaptic depression test
//	//inputGOGABASynDepGO = new float[cp->numGO];
//	//goGABAOutSynScaleGOGO = new float[cp->numGO];
//
//	//gMFGO = new float[cp->numGO];
//	//gNMDAMFGO = new float[cp->numGO];
//	//gNMDAUBCGO = new float[cp->numGO];
//	//gNMDAIncMFGO = new float[cp->numGO];
//	//gGRGO = new float[cp->numGO];
//	//gGRGO_NMDA = new float[cp->numGO];
//	//gGOGO = new float[cp->numGO];
//	//gMGluRGO = new float[cp->numGO];
//	//gMGluRIncGO = new float[cp->numGO];
//	//mGluRGO = new float[cp->numGO];
//	//gluGO = new float[cp->numGO];
//
//	//New GR stuff
//	//depAmpMFGR = new float[cp->numMF];
//	////depAmpMFUBC = new float[cp->numMF];
//	//gi_MFtoGR = new float[cp->numMF];
//	//gSum_MFGR = new float[cp->numGR];
//
//	//apGR = new ct_uint8_t[cp->numGR];
//	//apBufGR =  new ct_uint32_t[cp->numGR];
//	//gMFGR = allocate2DArray<float>(cp->numGR, cp->maxnumpGRfromMFtoGR);
//	//gUBCGR = allocate2DArray<float>(cp->numGR, cp->maxnumpGRfromMFtoGR);
//	//gMFSumGR = new float[cp->numGR];
//	//gMFDirectGR = new float[cp->numGR];
//	//gMFSpilloverGR = new float[cp->numGR];	
//	//gGODirectGR = new float[cp->numGR];
//	//gGOSpilloverGR = new float[cp->numGR];	
//	//apMFtoGR = new int[cp->numGR];
//	//apUBCtoGR = new int[cp->numGR];
//	//gUBCSumGR = new float[cp->numGR];
//	//gUBCDirectGR = new float[cp->numGR];
//	//gUBCSpilloverGR = new float[cp->numGR];	
//	//gNMDAGR = new float[cp->numGR];
//	//gNMDAIncGR = new float[cp->numGR];
//	//gLeakGR = new float[cp->numGR];
//	//depAmpMFtoGR = new float[cp->numGR];
//	//depAmpUBCtoGR = new float[cp->numGR];
//	//depAmpGOtoGR = new float[cp->numGR];
//	//dynamicAmpGOtoGR = new float[cp->numGR];	
//	//gGOGR = allocate2DArray<float>(cp->numGR, cp->maxnumpGRfromGOtoGR);
//	//gGOSumGR = new float[cp->numGR];
//	//threshGR = new float[cp->numGR];
//	//vGR = new float[cp->numGR];
//	//gKCaGR = new float[cp->numGR];
//	//historyGR = new ct_uint64_t[cp->numGR];
//
//	// sc
//	//apSC = new ct_uint8_t[cp->numSC];
//	//apBufSC = new ct_uint32_t[cp->numSC];
//	//gPFSC = new float[cp->numSC];
//	//threshSC = new float[cp->numSC];
//	//vSC = new float[cp->numSC];
//	//inputSumPFSC = new ct_uint32_t[cp->numSC];
//
//	//UBC
//	//gRise_MFtoUBC = new float[cp->numUBC];
//	//gDecay_MFtoUBC = new float[cp->numUBC];
//	//gSum_MFtoUBC = new float[cp->numUBC];
//	//depAmpUBCtoUBC = new float[cp->numUBC];
//	//gRise_UBCNMDA = new float[cp->numUBC];
//	//gDecay_UBCNMDA = new float[cp->numUBC];
//	//gSum_UBCNMDA = new float[cp->numUBC];
//	//gK_UBC = new float[cp->numUBC];
//	//
//	//gRise_UBCtoUBC = new float[cp->numUBC];
//	//gDecay_UBCtoUBC = new float[cp->numUBC];
//	//gSumOutUBCtoUBC = new float[cp->numUBC];
//	//gSumInUBCtoUBC = new float[cp->numUBC];
//	//
//	//inputMFUBC = new int[cp->numUBC];
//	//inputGOUBC = new int[cp->numUBC];
//	//gSum_GOtoUBC = new float[cp->numUBC];	
//	//vUBC = new float[cp->numUBC];	
//	//apUBC = new ct_uint8_t[cp->numUBC];	
//	//threshUBC = new float[cp->numUBC];	
//	//inputUBCtoUBC = new int[cp->numUBC];	
//
//	//gi_UBCtoGO = new float[cp->numUBC];
//	//depAmpUBCGO = new float[cp->numUBC];
//	//depAmpUBCGR = new float[cp->numUBC];
//}

void InNetActivityState::stateRW(ConnectivityParams &cp, bool read, std::fstream &file)
{
	rawBytesRW((char *)histMF, cp.NUM_MF * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufMF, cp.NUM_MF * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)spkGO, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)apGO, cp.NUM_GO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGO, cp.NUM_GO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)vGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)threshCurGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)inputMFGO, cp.NUM_GO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputUBCGO, cp.NUM_GO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)depAmpMFGO, cp.NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGO, cp.NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)inputGOGO, cp.NUM_GO * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gi_GOtoGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gSum_GOGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGR, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOGR, cp.NUM_GO * sizeof(float), read, file);
	
	rawBytesRW((char *)gSum_UBCtoGO, cp.NUM_GO * sizeof(float), read, file);

	rawBytesRW((char *)depAmpMFGR, cp.NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)depAmpMFUBC, cp.NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGR, cp.NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGR, cp.NUM_GR * sizeof(float), read, file);

	rawBytesRW((char *)inputGOGABASynDepGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)goGABAOutSynScaleGOGO, cp.NUM_GO * sizeof(float), read, file);

	rawBytesRW((char *)gMFGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAUBCGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAMFGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncMFGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO_NMDA, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGOGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gMGluRGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gMGluRIncGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)mGluRGO, cp.NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gluGO, cp.NUM_GO * sizeof(float), read, file);

	rawBytesRW((char *)apGR, cp.NUM_GR * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGR, cp.NUM_GR * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gMFGR[0], cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_MF_TO_GR * sizeof(float), read, file);
	rawBytesRW((char *)gUBCGR[0], cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_MF_TO_GR * sizeof(float), read, file);
	rawBytesRW((char *)gMFSumGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gMFDirectGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gMFSpilloverGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gGODirectGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gGOSpilloverGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)apMFtoGR, cp.NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)apUBCtoGR, cp.NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)gUBCSumGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gUBCDirectGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gUBCSpilloverGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gLeakGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)depAmpMFtoGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCtoGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOtoGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOtoGR, cp.NUM_GR * sizeof(float), read, file);

	rawBytesRW((char *)gGOGR[0], cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GO_TO_GR * sizeof(float), read, file);
	rawBytesRW((char *)gGOSumGR, cp.NUM_GR * sizeof(float), read, file);

	rawBytesRW((char *)threshGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)vGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gKCaGR, cp.NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)historyGR, cp.NUM_GR * sizeof(ct_uint64_t), read, file);

	rawBytesRW((char *)apSC, cp.NUM_SC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufSC, cp.NUM_SC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gPFSC, cp.NUM_SC * sizeof(float), read, file);
	rawBytesRW((char *)threshSC, cp.NUM_SC * sizeof(float), read, file);
	rawBytesRW((char *)vSC, cp.NUM_SC * sizeof(float), read, file);

	rawBytesRW((char *)inputSumPFSC, cp.NUM_SC * sizeof(ct_uint32_t), read, file);

	//UBCs
	rawBytesRW((char *)gRise_MFtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gDecay_MFtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	
	rawBytesRW((char *)gRise_UBCNMDA, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gDecay_UBCNMDA, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gSum_UBCNMDA, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gK_UBC, cp.NUM_UBC * sizeof(float), read, file);
	
	rawBytesRW((char *)gRise_UBCtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gDecay_UBCtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gSumOutUBCtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)gSumInUBCtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	
	rawBytesRW((char *)inputMFUBC, cp.NUM_UBC * sizeof(int), read, file);
	rawBytesRW((char *)inputGOUBC, cp.NUM_UBC * sizeof(int), read, file);
	rawBytesRW((char *)gSum_GOtoUBC, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)vUBC, cp.NUM_UBC * sizeof(float), read, file);
	
	rawBytesRW((char *)apUBC, cp.NUM_UBC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)threshUBC, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)inputUBCtoUBC, cp.NUM_UBC * sizeof(int), read, file);
	
	rawBytesRW((char *)gi_UBCtoGO, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCGO, cp.NUM_UBC * sizeof(float), read, file);
	rawBytesRW((char *)depAmpUBCGR, cp.NUM_UBC * sizeof(float), read, file);
}

void InNetActivityState::initializeVals(ConnectivityParams &cp, ActivityParams *ap)
{
	// only actively initializing those arrays whose initial values we want
	// differ from the default initilizer value
	goTimeStep = 0;

	// mf
	std::fill(histMF, histMF + cp.NUM_MF, false);
	std::fill(depAmpMFGO, depAmpMFGO + cp.NUM_MF, 1.0);
	std::fill(depAmpMFGR, depAmpMFGR + cp.NUM_MF, 1.0);
	std::fill(depAmpMFUBC, depAmpMFUBC + cp.NUM_MF, 1.0);

	// go	
	std::fill(synWscalerGRtoGO, synWscalerGRtoGO + cp.NUM_GO, 1.0);
	std::fill(synWscalerGOtoGO, synWscalerGOtoGO + cp.NUM_GO, 1.0);
	std::fill(apGO, apGO + cp.NUM_GO, false);
	std::fill(vGO, vGO + cp.NUM_GO, ap->eLeakGO);
	std::fill(threshCurGO, threshCurGO + cp.NUM_GO, ap->threshRestGO);
	std::fill(vSum_GOGO, vSum_GOGO + cp.NUM_GO, ap->eLeakGO);
	std::fill(vSum_GRGO, vSum_GRGO + cp.NUM_GO, ap->eLeakGO);
	std::fill(vSum_MFGO, vSum_MFGO + cp.NUM_GO, ap->eLeakGO);
	std::fill(goGABAOutSynScaleGOGO, goGABAOutSynScaleGOGO + cp.NUM_GO, 1.0);

	// gr
	std::fill(apGR, apGR + cp.NUM_GR, false);
	std::fill(gLeakGR, gLeakGR + cp.NUM_GR, 0.11);
	std::fill(depAmpMFtoGR, depAmpMFtoGR + cp.NUM_GR, 1.0);
	std::fill(depAmpUBCtoGR, depAmpUBCtoGR + cp.NUM_GR, 1.0);
	std::fill(depAmpGOtoGR, depAmpGOtoGR + cp.NUM_GR, 1.0);
	std::fill(threshGR, threshGR + cp.NUM_GR, ap->threshRestGR);
	std::fill(vGR, vGR + cp.NUM_GR, ap->eLeakGR);

	// sc
	std::fill(apSC, apSC + cp.NUM_SC, false);	
	std::fill(threshSC, threshSC + cp.NUM_SC, ap->threshRestSC);
	std::fill(apSC, apSC + cp.NUM_SC, false);
	std::fill(threshSC, threshSC + cp.NUM_SC, ap->threshRestSC);
	std::fill(vSC, vSC + cp.NUM_SC, ap->eLeakSC);

	// ubc
	std::fill(depAmpUBCtoUBC, depAmpUBCtoUBC + cp.NUM_UBC, 1.0);
	std::fill(vUBC, vUBC + cp.NUM_UBC, -70.0);
	std::fill(threshUBC, threshUBC + cp.NUM_UBC, ap->threshRestGO);
	std::fill(apUBC, apUBC + cp.NUM_UBC, false);
	std::fill(inputUBCtoUBC, inputUBCtoUBC + cp.NUM_UBC, false);
	std::fill(depAmpUBCGO, depAmpUBCGO + cp.NUM_UBC, 1.0);
	std::fill(depAmpUBCGR, depAmpUBCGR + cp.NUM_UBC, 1.0);
}

