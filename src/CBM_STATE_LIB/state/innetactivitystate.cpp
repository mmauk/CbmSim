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
	allocateArrMem(ap);
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
	// TODO: implement better function for handling underlying pointer
	rawBytesRW((char *)histMF.get(), NUM_MF * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufMF.get(), NUM_MF * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)apGO.get(), NUM_GO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGO.get(), NUM_GO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)vGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)threshCurGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)inputMFGO.get(), NUM_GO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)depAmpMFGO.get(), NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGO.get(), NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)inputGOGO.get(), NUM_GO * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gi_GOtoGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gSum_GOGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGR.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOGR.get(), NUM_GO * sizeof(float), read, file);
	
	rawBytesRW((char *)depAmpMFGR, NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGR, NUM_MF * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGR, NUM_GR * sizeof(float), read, file);

	rawBytesRW((char *)gNMDAMFGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncMFGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO_NMDA.get(), NUM_GO * sizeof(float), read, file);

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
}

void InNetActivityState::allocateArrMem(ActivityParams &ap)
{
	// mf
	histMF    = std::make_unique<ct_uint8_t[]>(NUM_MF);
	apBufMF   = std::make_unique<ct_uint32_t[]>(NUM_MF);

	// go
	synWscalerGRtoGO = std::make_unique<float[]>(NUM_GO);
	apGO 			 = std::make_unique<ct_uint8_t[]>(NUM_GO);
	apBufGO 		 = std::make_unique<ct_uint32_t[]>(NUM_GO);
	vGO				 = std::make_unique<float[]>(NUM_GO);
	vCoupleGO		 = std::make_unique<float[]>(NUM_GO);
	threshCurGO		 = std::make_unique<float[]>(NUM_GO);

	inputMFGO  = std::make_unique<ct_uint32_t[]>(NUM_GO);
	depAmpMFGO = std::make_unique<float[]>(NUM_GO);
	gi_MFtoGO  = std::make_unique<float[]>(NUM_MF);
	gSum_MFGO  = std::make_unique<float[]>(NUM_GO);
	
	inputGOGO  = std::make_unique<ct_uint32_t[]>(NUM_GO);
	gi_GOtoGO  = std::make_unique<float[]>(NUM_GO);
	depAmpGOGO = std::make_unique<float[]>(NUM_GO);
	gSum_GOGO  = std::make_unique<float[]>(NUM_GO);
	depAmpGOGR = std::make_unique<float[]>(NUM_GO);

	dynamicAmpGOGR = std::make_unique<float[]>(NUM_GO);
	gNMDAMFGO	   = std::make_unique<float[]>(NUM_GO);
	gNMDAIncMFGO   = std::make_unique<float[]>(NUM_GO);
	gGRGO		   = std::make_unique<float[]>(NUM_GO);
	gGRGO_NMDA	   = std::make_unique<float[]>(NUM_GO);

}

void InNetActivityState::initializeVals(ActivityParams &ap)
{
	// only actively initializing those arrays whose initial values we want
	// differ from the default initilizer value
	goTimeStep = 0;

	// mf
	std::fill(depAmpMFGO.get(), depAmpMFGO.get() + NUM_MF, 1.0);
	std::fill(depAmpMFGR, depAmpMFGR + NUM_MF, 1.0);

	// go	
	std::fill(synWscalerGRtoGO.get(), synWscalerGRtoGO.get() + NUM_GO, 1.0);
	std::fill(vGO.get(), vGO.get() + NUM_GO, ap.eLeakGO);
	std::fill(threshCurGO.get(), threshCurGO.get() + NUM_GO, ap.threshRestGO);

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
}

