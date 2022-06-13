/*
 * innetactivitystate.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: consciousness
 *  Updated on: March, 2022
 * 		Author: the gallogly
 */ 

#include "state/innetactivitystate.h"

InNetActivityState::InNetActivityState(ActivityParams &ap)
{
	initializeVals(ap);
}

InNetActivityState::InNetActivityState(std::fstream &infile)
{
	stateRW(true, infile);
}

InNetActivityState::~InNetActivityState() {}

void InNetActivityState::writeState(std::fstream &outfile)
{
	stateRW(false, outfile);
}

void InNetActivityState::resetState(ActivityParams &ap)
{
	initializeVals(ap);
}

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

void InNetActivityState::initializeVals(ActivityParams &ap)
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
	std::fill(vGO, vGO + NUM_GO, ap.eLeakGO);
	std::fill(threshCurGO, threshCurGO + NUM_GO, ap.threshRestGO);
	std::fill(vSum_GOGO, vSum_GOGO + NUM_GO, ap.eLeakGO);
	std::fill(vSum_GRGO, vSum_GRGO + NUM_GO, ap.eLeakGO);
	std::fill(vSum_MFGO, vSum_MFGO + NUM_GO, ap.eLeakGO);
	std::fill(goGABAOutSynScaleGOGO, goGABAOutSynScaleGOGO + NUM_GO, 1.0);

	// gr
	std::fill(apGR, apGR + NUM_GR, false);
	std::fill(gLeakGR, gLeakGR + NUM_GR, 0.11);
	std::fill(depAmpMFtoGR, depAmpMFtoGR + NUM_GR, 1.0);
	std::fill(depAmpUBCtoGR, depAmpUBCtoGR + NUM_GR, 1.0);
	std::fill(depAmpGOtoGR, depAmpGOtoGR + NUM_GR, 1.0);
	std::fill(threshGR, threshGR + NUM_GR, ap.threshRestGR);
	std::fill(vGR, vGR + NUM_GR, ap.eLeakGR);

	// sc
	std::fill(apSC, apSC + NUM_SC, false);	
	std::fill(threshSC, threshSC + NUM_SC, ap.threshRestSC);
	std::fill(apSC, apSC + NUM_SC, false);
	std::fill(threshSC, threshSC + NUM_SC, ap.threshRestSC);
	std::fill(vSC, vSC + NUM_SC, ap.eLeakSC);

	// ubc
	std::fill(depAmpUBCtoUBC, depAmpUBCtoUBC + NUM_UBC, 1.0);
	std::fill(vUBC, vUBC + NUM_UBC, -70.0);
	std::fill(threshUBC, threshUBC + NUM_UBC, ap.threshRestGO);
	std::fill(apUBC, apUBC + NUM_UBC, false);
	std::fill(inputUBCtoUBC, inputUBCtoUBC + NUM_UBC, false);
	std::fill(depAmpUBCGO, depAmpUBCGO + NUM_UBC, 1.0);
	std::fill(depAmpUBCGR, depAmpUBCGR + NUM_UBC, 1.0);
}

