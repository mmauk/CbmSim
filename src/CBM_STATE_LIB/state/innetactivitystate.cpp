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
	
	rawBytesRW((char *)depAmpMFGR.get(), NUM_MF * sizeof(float), read, file);

	rawBytesRW((char *)gNMDAMFGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncMFGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO.get(), NUM_GO * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO_NMDA.get(), NUM_GO * sizeof(float), read, file);

	rawBytesRW((char *)apGR.get(), NUM_GR * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGR.get(), NUM_GR * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gMFGR.get(), NUM_GR * MAX_NUM_P_GR_FROM_MF_TO_GR * sizeof(float), read, file);
	rawBytesRW((char *)gMFSumGR.get(), NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)apMFtoGR.get(), NUM_GR * sizeof(int), read, file);

	rawBytesRW((char *)gGOGR.get(), NUM_GR * MAX_NUM_P_GR_FROM_GO_TO_GR * sizeof(float), read, file);
	rawBytesRW((char *)gGOSumGR.get(), NUM_GR * sizeof(float), read, file);

	rawBytesRW((char *)threshGR.get(), NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)vGR.get(), NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)gKCaGR.get(), NUM_GR * sizeof(float), read, file);
	rawBytesRW((char *)historyGR.get(), NUM_GR * sizeof(ct_uint64_t), read, file);

	rawBytesRW((char *)apSC.get(), NUM_SC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufSC.get(), NUM_SC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gPFSC.get(), NUM_SC * sizeof(float), read, file);
	rawBytesRW((char *)threshSC.get(), NUM_SC * sizeof(float), read, file);
	rawBytesRW((char *)vSC.get(), NUM_SC * sizeof(float), read, file);

	//rawBytesRW((char *)inputSumPFSC.get(), NUM_SC * sizeof(ct_uint32_t), read, file);
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
	gGOGO		   = std::make_unique<float[]>(NUM_GO);

	depAmpMFGR	   = std::make_unique<float[]>(NUM_MF);
	apGR		   = std::make_unique<ct_uint8_t[]>(NUM_GR);	
	apBufGR		   = std::make_unique<ct_uint32_t[]>(NUM_GR);
	gMFGR		   = std::make_unique<float[]>(NUM_GR * MAX_NUM_P_GR_FROM_MF_TO_GR);
	gMFSumGR	   = std::make_unique<float[]>(NUM_GR);
	apMFtoGR	   = std::make_unique<float[]>(NUM_GR);
	gGOGR		   = std::make_unique<float[]>(NUM_GR * MAX_NUM_P_GR_FROM_GO_TO_GR);
	gGOSumGR	   = std::make_unique<float[]>(NUM_GR);
	threshGR	   = std::make_unique<float[]>(NUM_GR);
	vGR			   = std::make_unique<float[]>(NUM_GR);
	gKCaGR		   = std::make_unique<float[]>(NUM_GR);
	historyGR	   = std::make_unique<ct_uint64_t[]>(NUM_GR);

	apSC		   = std::make_unique<ct_uint8_t[]>(NUM_SC);
	apBufSC		   = std::make_unique<ct_uint32_t[]>(NUM_SC);
	gPFSC		   = std::make_unique<float[]>(NUM_SC);
	threshSC	   = std::make_unique<float[]>(NUM_SC);
	vSC		       = std::make_unique<float[]>(NUM_SC);
	// Not used.
	// inputSumPFSC   = std::make_unique<ct_uint32_t[]>(NUM_SC);
}

void InNetActivityState::initializeVals(ActivityParams &ap)
{
	// only actively initializing those arrays whose initial values we want
	// differ from the default initilizer value

	// mf
	std::fill(depAmpMFGO.get(), depAmpMFGO.get() + NUM_MF, 1.0);
	std::fill(depAmpMFGR.get(), depAmpMFGR.get() + NUM_MF, 1.0);

	// go	
	std::fill(synWscalerGRtoGO.get(), synWscalerGRtoGO.get() + NUM_GO, 1.0);
	std::fill(vGO.get(), vGO.get() + NUM_GO, ap.eLeakGO);
	std::fill(threshCurGO.get(), threshCurGO.get() + NUM_GO, ap.threshRestGO);

	// gr
	std::fill(threshGR.get(), threshGR.get() + NUM_GR, ap.threshRestGR);
	std::fill(vGR.get(), vGR.get() + NUM_GR, ap.eLeakGR);

	// sc
	// NOTE: no need to explicitly init apSC: false is the default value made by make_unique
	std::fill(threshSC.get(), threshSC.get() + NUM_SC, ap.threshRestSC);
	std::fill(vSC.get(), vSC.get() + NUM_SC, ap.eLeakSC);
}

