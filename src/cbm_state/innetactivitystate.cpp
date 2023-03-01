/*
 * innetactivitystate.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: consciousness
 *  Updated on: March, 2022
 *      Author: the gallogly
 */ 
#include "assert.h"
#include "array_util.h"
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

//TODO: rewrite: must include zero-ing out all arrays not invovled in initializeVals
void InNetActivityState::resetState()
{
	initializeVals();
}

// TODO: finish
bool InNetActivityState::inInitialState()
{
	// mf
	assert(arr_filled_with_int_t<uint8_t>(histMF.get(), num_mf, 0),
		   "ERROR: histMF not all zero", __func__);
	assert(arr_filled_with_int_t<uint32_t>(apBufMF.get(), num_mf, 0),
		   "ERROR: apBufMF not all zero", __func__);

	// go
	assert(arr_filled_with_int_t<uint8_t>(apGO.get(), num_go, 0),
		   "ERROR: apGO not all zero", __func__);
	assert(arr_filled_with_int_t<uint32_t>(apBufGO.get(), num_go, 0),
		   "ERROR: apBufGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(vCoupleGO.get(), num_go, 0.0),
		   "ERROR: vCoupleGO not all zero", __func__);
	assert(arr_filled_with_int_t<uint32_t>(inputMFGO.get(), num_go, 0),
		   "ERROR: inputMFGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gi_MFtoGO.get(), num_mf, 0.0),
		   "ERROR: gi_MFtoGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gSum_MFGO.get(), num_go, 0.0),
		   "ERROR: gSum_MFGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(inputGOGO.get(), num_go, 0.0),
		   "ERROR: inputGOGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gi_GOtoGO.get(), num_go, 0.0),
		   "ERROR: gi_GOtoGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(depAmpGOGO.get(), num_go, 0.0),
		   "ERROR: depAmpGOGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gSum_GOGO.get(), num_go, 0.0),
		   "ERROR: gSum_GOGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(depAmpGOGR.get(), num_go, 0.0),
		   "ERROR: depAmpGOGR not all zero", __func__);
	assert(arr_filled_with_float_t<float>(dynamicAmpGOGR.get(), num_go, 0.0),
		   "ERROR: dynamicAmpGOGR not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gNMDAMFGO.get(), num_go, 0.0),
		   "ERROR: gNMDAMFGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gNMDAIncMFGO.get(), num_go, 0.0),
		   "ERROR: gNMDAIncMFGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gGRGO.get(), num_go, 0.0),
		   "ERROR: gGRGO not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gGRGO_NMDA.get(), num_go, 0.0),
		   "ERROR: gGRGO_NMDA not all zero", __func__);

	assert(arr_filled_with_int_t<uint8_t>(apGR.get(), num_gr, 0),
		   "ERROR: apGR not all zero", __func__);
	assert(arr_filled_with_int_t<uint32_t>(apBufGR.get(), num_gr, 0),
		   "ERROR: apBufGR not all zero", __func__);

	assert(arr_filled_with_float_t<float>(gMFGR.get(), num_gr * max_num_p_gr_from_mf_to_gr, 0.0),
		   "ERROR: gMFGR not all zero", __func__);

	assert(arr_filled_with_float_t<float>(gMFSumGR.get(), num_gr, 0.0),
		   "ERROR: gMFSumGR not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gMFSumGR.get(), num_gr, 0.0),
		   "ERROR: apMFtoGR not all zero", __func__);

	assert(arr_filled_with_float_t<float>(gGOGR.get(), num_gr * max_num_p_gr_from_go_to_gr, 0.0),
		   "ERROR: gGOGR not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gGOSumGR.get(), num_gr, 0.0),
		   "ERROR: gGOSumGR not all zero", __func__);
	assert(arr_filled_with_float_t<float>(gKCaGR.get(), num_gr, 0.0),
		   "ERROR: gKCaGR not all zero", __func__);
	assert(arr_filled_with_int_t<uint64_t>(historyGR.get(), num_gr, 0),
		   "ERROR: historyGR not all zero", __func__);

	// only actively initializing those arrays whose initial values we want
	// differ from the default initilizer value

	// mf
	assert(arr_filled_with_float_t<float>(depAmpMFGO.get(), num_mf, 1.0),
		   "ERROR: depAmpMFGO not all one", __func__);
	assert(arr_filled_with_float_t<float>(depAmpMFGR.get(), num_mf, 1.0),
		   "ERROR: depAmpMFGR not all one", __func__);

	// go
	assert(arr_filled_with_float_t<float>(synWscalerGOtoGO.get(), num_go, 1.0),
		   "ERROR: synWscalerGOtoGO not all one", __func__);
	assert(arr_filled_with_float_t<float>(synWscalerGRtoGO.get(), num_go, 1.0),
		   "ERROR: synWscalerGRtoGO not all one", __func__);

	assert(arr_filled_with_float_t<float>(vGO.get(), num_go, eLeakGO),
		   "ERROR: vGO not all eLeakGO", __func__);
	assert(arr_filled_with_float_t<float>(threshCurGO.get(), num_go, threshRestGO),
		   "ERROR: threshCurGO not all threshRestGO", __func__);

	assert(arr_filled_with_float_t<float>(vGR.get(), num_gr, eLeakGR),
		   "ERROR: vGR not all eLeakGR", __func__);
	assert(arr_filled_with_float_t<float>(threshGR.get(), num_gr, threshRestGR),
		   "ERROR: threshGR not all threshRestGR", __func__);
  return true;
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
	rawBytesRW((char *)depAmpMFGO.get(), num_mf * sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGO.get(), num_mf * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGO.get(), num_go * sizeof(float), read, file);
	rawBytesRW((char *)inputGOGO.get(), num_go * sizeof(float), read, file);

	rawBytesRW((char *)gi_GOtoGO.get(), num_go * sizeof(float), read, file);
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
	depAmpMFGO = std::make_unique<float[]>(num_mf);
	gi_MFtoGO  = std::make_unique<float[]>(num_mf);
	gSum_MFGO  = std::make_unique<float[]>(num_go);
	inputGOGO  = std::make_unique<float[]>(num_go);

	gi_GOtoGO  = std::make_unique<float[]>(num_go);
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
	std::fill(depAmpMFGO.get(), depAmpMFGO.get() + num_mf, 1.0);
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

