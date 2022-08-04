/*
 * innetactivitystate.cpp
 *
 *  Created on: Nov 16, 2012
 *      Author: consciousness
 *  Updated on: March, 2022
 * 		Author: the gallogly
 */ 

#include "state/innetactivitystate.h"

InNetActivityState::InNetActivityState(ConnectivityParams *cp)
{
	std::cout << "[INFO]: Allocating and initializing innet activity state..." << std::endl;
	allocateMemory(cp);
	initializeVals(cp);
	std::cout << "[INFO]: Finished allocating and initializing innet activity state." << std::endl;
}

InNetActivityState::InNetActivityState(ConnectivityParams *cp, std::fstream &infile)
{
	allocateMemory(cp);
	stateRW(cp, true, infile);
}

InNetActivityState::~InNetActivityState() {}

void InNetActivityState::readState(ConnectivityParams *cp, std::fstream &infile)
{
	stateRW(cp, true, infile);
}

void InNetActivityState::writeState(ConnectivityParams *cp, std::fstream &outfile)
{
	stateRW(cp, false, outfile);
}

void InNetActivityState::resetState(ConnectivityParams *cp)
{
	initializeVals(cp);
}

void InNetActivityState::stateRW(ConnectivityParams *cp, bool read, std::fstream &file)
{
	// TODO: implement better function for handling underlying pointer
	rawBytesRW((char *)histMF.get(), cp->int_params["num_mf"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufMF.get(), cp->int_params["num_mf"] * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)synWscalerGRtoGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)apGO.get(), cp->int_params["num_go"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGO.get(), cp->int_params["num_go"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)vGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)threshCurGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);

	rawBytesRW((char *)inputMFGO.get(), cp->int_params["num_go"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)depAmpMFGO.get(), cp->int_params["num_mf"] * sizeof(float), read, file);
	rawBytesRW((char *)gi_MFtoGO.get(), cp->int_params["num_mf"] * sizeof(float), read, file);
	rawBytesRW((char *)gSum_MFGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)inputGOGO.get(), cp->int_params["num_go"] * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gi_GOtoGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)gSum_GOGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)depAmpGOGR.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)dynamicAmpGOGR.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	
	rawBytesRW((char *)gNMDAMFGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)gNMDAIncMFGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO.get(), cp->int_params["num_go"] * sizeof(float), read, file);
	rawBytesRW((char *)gGRGO_NMDA.get(), cp->int_params["num_go"] * sizeof(float), read, file);

	rawBytesRW((char *)depAmpMFGR.get(), cp->int_params["num_mf"] * sizeof(float), read, file);
	rawBytesRW((char *)apGR.get(), cp->int_params["num_gr"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufGR.get(), cp->int_params["num_gr"] * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)gMFGR.get(), cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_mf_to_gr"] * sizeof(float), read, file);
	rawBytesRW((char *)gMFSumGR.get(), cp->int_params["num_gr"] * sizeof(float), read, file);
	rawBytesRW((char *)apMFtoGR.get(), cp->int_params["num_gr"] * sizeof(float), read, file);

	rawBytesRW((char *)gGOGR.get(), cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_go_to_gr"] * sizeof(float), read, file);
	rawBytesRW((char *)gGOSumGR.get(), cp->int_params["num_gr"] * sizeof(float), read, file);
	rawBytesRW((char *)threshGR.get(), cp->int_params["num_gr"] * sizeof(float), read, file);
	rawBytesRW((char *)vGR.get(), cp->int_params["num_gr"] * sizeof(float), read, file);
	rawBytesRW((char *)gKCaGR.get(), cp->int_params["num_gr"] * sizeof(float), read, file);
	rawBytesRW((char *)historyGR.get(), cp->int_params["num_gr"] * sizeof(ct_uint64_t), read, file);

	rawBytesRW((char *)apSC.get(), cp->int_params["num_sc"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufSC.get(), cp->int_params["num_sc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)gPFSC.get(), cp->int_params["num_sc"] * sizeof(float), read, file);
	rawBytesRW((char *)threshSC.get(), cp->int_params["num_sc"] * sizeof(float), read, file);
	rawBytesRW((char *)vSC.get(), cp->int_params["num_sc"] * sizeof(float), read, file);

	//rawBytesRW((char *)inputSumPFSC.get(), cp->int_params["num_sc"] * sizeof(ct_uint32_t), read, file);
}

void InNetActivityState::allocateMemory(ConnectivityParams *cp)
{
	// mf
	histMF    = std::make_unique<ct_uint8_t[]>(cp->int_params["num_mf"]);
	apBufMF   = std::make_unique<ct_uint32_t[]>(cp->int_params["num_mf"]);

	// go
	synWscalerGRtoGO = std::make_unique<float[]>(cp->int_params["num_go"]);
	apGO             = std::make_unique<ct_uint8_t[]>(cp->int_params["num_go"]);
	apBufGO          = std::make_unique<ct_uint32_t[]>(cp->int_params["num_go"]);
	vGO              = std::make_unique<float[]>(cp->int_params["num_go"]);
	vCoupleGO        = std::make_unique<float[]>(cp->int_params["num_go"]);
	threshCurGO      = std::make_unique<float[]>(cp->int_params["num_go"]);

	inputMFGO  = std::make_unique<ct_uint32_t[]>(cp->int_params["num_go"]);
	depAmpMFGO = std::make_unique<float[]>(cp->int_params["num_mf"]);
	gi_MFtoGO  = std::make_unique<float[]>(cp->int_params["num_mf"]);
	gSum_MFGO  = std::make_unique<float[]>(cp->int_params["num_go"]);
	inputGOGO  = std::make_unique<ct_uint32_t[]>(cp->int_params["num_go"]);

	gi_GOtoGO  = std::make_unique<float[]>(cp->int_params["num_go"]);
	depAmpGOGO = std::make_unique<float[]>(cp->int_params["num_go"]);
	gSum_GOGO  = std::make_unique<float[]>(cp->int_params["num_go"]);
	depAmpGOGR = std::make_unique<float[]>(cp->int_params["num_go"]);
	dynamicAmpGOGR = std::make_unique<float[]>(cp->int_params["num_go"]);

	gNMDAMFGO      = std::make_unique<float[]>(cp->int_params["num_go"]);
	gNMDAIncMFGO   = std::make_unique<float[]>(cp->int_params["num_go"]);
	gGRGO          = std::make_unique<float[]>(cp->int_params["num_go"]);
	gGRGO_NMDA     = std::make_unique<float[]>(cp->int_params["num_go"]);

	depAmpMFGR     = std::make_unique<float[]>(cp->int_params["num_mf"]);
	apGR           = std::make_unique<ct_uint8_t[]>(cp->int_params["num_gr"]);
	apBufGR        = std::make_unique<ct_uint32_t[]>(cp->int_params["num_gr"]);

	gMFGR          = std::make_unique<float[]>(cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_mf_to_gr"]);
	gMFSumGR       = std::make_unique<float[]>(cp->int_params["num_gr"]);
	apMFtoGR       = std::make_unique<float[]>(cp->int_params["num_gr"]);

	gGOGR          = std::make_unique<float[]>(cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_go_to_gr"]);
	gGOSumGR       = std::make_unique<float[]>(cp->int_params["num_gr"]);
	threshGR       = std::make_unique<float[]>(cp->int_params["num_gr"]);
	vGR            = std::make_unique<float[]>(cp->int_params["num_gr"]);
	gKCaGR         = std::make_unique<float[]>(cp->int_params["num_gr"]);
	historyGR      = std::make_unique<ct_uint64_t[]>(cp->int_params["num_gr"]);

	apSC           = std::make_unique<ct_uint8_t[]>(cp->int_params["num_sc"]);
	apBufSC        = std::make_unique<ct_uint32_t[]>(cp->int_params["num_sc"]);
	gPFSC          = std::make_unique<float[]>(cp->int_params["num_sc"]);
	threshSC       = std::make_unique<float[]>(cp->int_params["num_sc"]);
	vSC            = std::make_unique<float[]>(cp->int_params["num_sc"]);
	// Not used.
	// inputSumPFSC   = std::make_unique<ct_uint32_t[]>(cp->int_params["num_sc"]);
}

void InNetActivityState::initializeVals(ConnectivityParams *cp)
{
	// only actively initializing those arrays whose initial values we want
	// differ from the default initilizer value

	// mf
	std::fill(depAmpMFGO.get(), depAmpMFGO.get() + cp->int_params["num_mf"], 1.0);
	std::fill(depAmpMFGR.get(), depAmpMFGR.get() + cp->int_params["num_mf"], 1.0);

	// go	
	std::fill(synWscalerGRtoGO.get(), synWscalerGRtoGO.get() + cp->int_params["num_go"], 1.0);
	std::fill(vGO.get(), vGO.get() + cp->int_params["num_go"], act_params[eLeakGO]);
	std::fill(threshCurGO.get(), threshCurGO.get() + cp->int_params["num_go"], act_params[threshRestGO]);

	// gr
	std::fill(threshGR.get(), threshGR.get() + cp->int_params["num_gr"], act_params[threshRestGR]);
	std::fill(vGR.get(), vGR.get() + cp->int_params["num_gr"], act_params[eLeakGR]);

	// sc
	// NOTE: no need to explicitly init apSC: false is the default value made by make_unique
	std::fill(threshSC.get(), threshSC.get() + cp->int_params["num_sc"], act_params[threshRestSC]);
	std::fill(vSC.get(), vSC.get() + cp->int_params["num_sc"], act_params[eLeakSC]);
}

