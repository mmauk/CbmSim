/*
 * innetactivitystate.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: consciousness
 *  Updated on: March, 2022
 *      Author: the gallogly
 */ 
#include "logger.h"
#include "innetactivitystate.h"

InNetActivityState::InNetActivityState()
{
	LOG_DEBUG("Allocating and initializing innet activity state...");
	allocateMemory();
	initializeVals();
	LOG_DEBUG("Finished allocating and initializing innet activity state.");
}

InNetActivityState::InNetActivityState(std::fstream &infile)
{
	allocateMemory();
	stateRW(true, infile);
}

InNetActivityState::~InNetActivityState() {}

void InNetActivityState::readState(std::fstream &infile)
{
	stateRW(true, infile);
}

void InNetActivityState::writeState(std::fstream &outfile)
{
	stateRW(false, outfile);
}

void InNetActivityState::resetState()
{
	initializeVals();
}

void InNetActivityState::stateRW(bool read, std::fstream &file)
{
	// TODO: implement better function for handling underlying pointer
	rawBytesRW((char *)histMF.get(), num_mf * sizeof(uint8_t), read, file);
	rawBytesRW((char *)apBufMF.get(), num_mf * sizeof(uint32_t), read, file);

	rawBytesRW((char *)synWscalerGOtoGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)synWscalerGRtoGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)apGO.get(), num_go * sizeof(uint8_t), read, file);
	rawBytesRW((char *)apBufGO.get(), num_go * sizeof(uint32_t), read, file);
	rawBytesRW((char *)vGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)threshCurGO.get(), num_go * sizeof(float), read, file);

	rawBytesRW((char *)inputMFGO.get(), num_go * sizeof(uint32_t), read, file);
	rawBytesRW((char *)gSum_MFGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)inputGOGO.get(), num_go * sizeof(float), read, file);

	rawBytesRW((char *)depAmpGOGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)gSum_GOGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGR.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOGR.get(), num_go * sizeof(float), read, file);
	
	rawBytesRW((char *)gNMDAMFGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncMFGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO_NMDA.get(), num_go * sizeof(float), read, file);

	rawBytesRW((char *)depAmpMFGR.get(), num_mf * sizeof(float), read, file);
	rawBytesRW((char *)apGR.get(), num_gr * sizeof(uint8_t), read, file);
	rawBytesRW((char *)apBufGR.get(), num_gr * sizeof(uint32_t), read, file);

	rawBytesRW((char *)gMFGR.get(), num_gr * max_num_p_gr_from_mf_to_gr * sizeof(float), read, file);
	rawBytesRW((char *)gMFSumGR.get(), num_gr * sizeof(float), read, file);
	rawBytesRW((char *)apMFtoGR.get(), num_gr * sizeof(float), read, file);

	rawBytesRW((char *)gGOGR.get(), num_gr * max_num_p_gr_from_go_to_gr * sizeof(float), read, file);
	rawBytesRW((char *)gGOSumGR.get(), num_gr * sizeof(float), read, file);
	rawBytesRW((char *)threshGR.get(), num_gr * sizeof(float), read, file);
	rawBytesRW((char *)vGR.get(), num_gr * sizeof(float), read, file);
	rawBytesRW((char *)gKCaGR.get(), num_gr * sizeof(float), read, file);
	rawBytesRW((char *)historyGR.get(), num_gr * sizeof(uint64_t), read, file);
}

void InNetActivityState::allocateMemory()
{
	// mf
	histMF    = std::make_unique<uint8_t[]>(num_mf);
	apBufMF   = std::make_unique<uint32_t[]>(num_mf);

	// go
	synWscalerGOtoGO = std::make_unique<float[]>(num_go);
	synWscalerGRtoGO = std::make_unique<float[]>(num_go);
	apGO             = std::make_unique<uint8_t[]>(num_go);
	apBufGO          = std::make_unique<uint32_t[]>(num_go);
	vGO              = std::make_unique<float[]>(num_go);
	vCoupleGO        = std::make_unique<float[]>(num_go);
	threshCurGO      = std::make_unique<float[]>(num_go);

	inputMFGO  = std::make_unique<uint32_t[]>(num_go);
	gSum_MFGO  = std::make_unique<float[]>(num_go);
	inputGOGO  = std::make_unique<float[]>(num_go);

	depAmpGOGO = std::make_unique<float[]>(num_go);
	gSum_GOGO  = std::make_unique<float[]>(num_go);
	depAmpGOGR = std::make_unique<float[]>(num_go);
	dynamicAmpGOGR = std::make_unique<float[]>(num_go);

	gNMDAMFGO      = std::make_unique<float[]>(num_go);
	gNMDAIncMFGO   = std::make_unique<float[]>(num_go);
	gGRGO          = std::make_unique<float[]>(num_go);
	gGRGO_NMDA     = std::make_unique<float[]>(num_go);

	depAmpMFGR     = std::make_unique<float[]>(num_mf);
	apGR           = std::make_unique<uint8_t[]>(num_gr);
	apBufGR        = std::make_unique<uint32_t[]>(num_gr);

	gMFGR          = std::make_unique<float[]>(num_gr * max_num_p_gr_from_mf_to_gr);
	gMFSumGR       = std::make_unique<float[]>(num_gr);
	apMFtoGR       = std::make_unique<float[]>(num_gr);

	gGOGR          = std::make_unique<float[]>(num_gr * max_num_p_gr_from_go_to_gr);
	gGOSumGR       = std::make_unique<float[]>(num_gr);
	threshGR       = std::make_unique<float[]>(num_gr);
	vGR            = std::make_unique<float[]>(num_gr);
	gKCaGR         = std::make_unique<float[]>(num_gr);
	historyGR      = std::make_unique<uint64_t[]>(num_gr);
}

void InNetActivityState::initializeVals()
{
	// only actively initializing those arrays whose initial values we want
	// differ from the default initilizer value

	// mf
	std::fill(depAmpMFGR.get(), depAmpMFGR.get() + num_mf, 1.0);

	// go
	std::fill(synWscalerGOtoGO.get(), synWscalerGOtoGO.get() + num_go, 1.0);
	std::fill(synWscalerGRtoGO.get(), synWscalerGRtoGO.get() + num_go, 1.0);

	std::fill(vGO.get(), vGO.get() + num_go, eLeakGO);
	std::fill(threshCurGO.get(), threshCurGO.get() + num_go, threshRestGO);

	// gr
	std::fill(vGR.get(), vGR.get() + num_gr, eLeakGR);
	std::fill(threshGR.get(), threshGR.get() + num_gr, threshRestGR);
}

