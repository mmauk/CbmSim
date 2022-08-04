/*
 * mzoneactivitystate.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: consciousness
 */

#include "state/mzoneactivitystate.h"

MZoneActivityState::MZoneActivityState() {}

MZoneActivityState::MZoneActivityState(ConnectivityParams *cp, int randSeed)
{
	allocateMemory(cp);
	initializeVals(cp, randSeed);
}

MZoneActivityState::MZoneActivityState(ConnectivityParams *cp, std::fstream &infile)
{
	allocateMemory(cp);
	stateRW(cp, true, infile);
}

MZoneActivityState::~MZoneActivityState() {}

void MZoneActivityState::readState(ConnectivityParams *cp, std::fstream &infile)
{
	stateRW(cp, true, infile);
}

void MZoneActivityState::writeState(ConnectivityParams *cp, std::fstream &outfile)
{
	stateRW(cp, false, outfile);
}

void MZoneActivityState::allocateMemory(ConnectivityParams *cp)
{
	// basket cells
	apBC      = std::make_unique<ct_uint8_t[]>(cp->int_params["num_bc"]);
	apBufBC   = std::make_unique<ct_uint32_t[]>(cp->int_params["num_bc"]);
	inputPCBC = std::make_unique<ct_uint32_t[]>(cp->int_params["num_bc"]);
	gPFBC     = std::make_unique<float[]>(cp->int_params["num_bc"]);
	gPCBC     = std::make_unique<float[]>(cp->int_params["num_bc"]);
	vBC       = std::make_unique<float[]>(cp->int_params["num_bc"]);
	threshBC  = std::make_unique<float[]>(cp->int_params["num_bc"]);

	// purkinje cells
	apPC          = std::make_unique<ct_uint8_t[]>(cp->int_params["num_pc"]);
	apBufPC       = std::make_unique<ct_uint32_t[]>(cp->int_params["num_pc"]);
	inputBCPC     = std::make_unique<ct_uint32_t[]>(cp->int_params["num_pc"]);
	inputSCPC     = std::make_unique<ct_uint8_t[]>(cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_sc_to_pc"]);
	pfSynWeightPC = std::make_unique<float[]>(cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_gr_to_pc"]);
	inputSumPFPC  = std::make_unique<float[]>(cp->int_params["num_pc"]);
	gPFPC         = std::make_unique<float[]>(cp->int_params["num_pc"]);
	gBCPC         = std::make_unique<float[]>(cp->int_params["num_pc"]);
	gSCPC         = std::make_unique<float[]>(cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_sc_to_pc"]);
	vPC           = std::make_unique<float[]>(cp->int_params["num_pc"]);
	threshPC      = std::make_unique<float[]>(cp->int_params["num_pc"]);
	histPCPopAct = std::make_unique<ct_uint32_t[]>(numPopHistBinsPC);

	// inferior olivary cells
	apIO      = std::make_unique<ct_uint8_t[]>(cp->int_params["num_io"]);
	apBufIO   = std::make_unique<ct_uint8_t[]>(cp->int_params["num_io"]);
	inputNCIO = std::make_unique<ct_uint8_t[]>(cp->int_params["num_io"] * cp->int_params["num_p_io_from_nc_to_io"]);
	gNCIO     = std::make_unique<float[]>(cp->int_params["num_io"] * cp->int_params["num_p_io_from_nc_to_io"]);
	threshIO  = std::make_unique<float[]>(cp->int_params["num_io"]);
	vIO       = std::make_unique<float[]>(cp->int_params["num_io"]);
	vCoupleIO = std::make_unique<float[]>(cp->int_params["num_io"]);
	pfPCPlastTimerIO = std::make_unique<ct_int32_t[]>(cp->int_params["num_io"]);

	// nucleus cells
	apNC          = std::make_unique<ct_uint8_t[]>(cp->int_params["num_nc"]);
	apBufNC       = std::make_unique<ct_uint32_t[]>(cp->int_params["num_nc"]);
	inputPCNC     = std::make_unique<ct_uint8_t[]>(cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_pc_to_nc"]);
	inputMFNC     = std::make_unique<ct_uint8_t[]>(cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"]);
	gPCNC         = std::make_unique<float[]>(cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_pc_to_nc"]);
	mfSynWeightNC = std::make_unique<float[]>(cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"]);
	gMFAMPANC     = std::make_unique<float[]>(cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"]);
	threshNC      = std::make_unique<float[]>(cp->int_params["num_nc"]);
	vNC           = std::make_unique<float[]>(cp->int_params["num_nc"]);
	synIOPReleaseNC = std::make_unique<float[]>(cp->int_params["num_nc"]);
}

void MZoneActivityState::initializeVals(ConnectivityParams *cp, int randSeed)
{
	//uncomment for actual runs 
	//CRandomSFMT0 randGen(randSeed);

	/* NOTE: I only fill those arrays whose initial values are intended to be nonzero, as
	 * std::make_unique performs value initialization on Type[] template arguments. - S.G.
	 */

	// bc
	std::fill(threshBC.get(), threshBC.get() + cp->int_params["num_bc"], act_params[threshRestBC]);
	std::fill(vBC.get(), vBC.get() + cp->int_params["num_bc"], act_params[eLeakBC]);

	// pc
	std::fill(vPC.get(), vPC.get() + cp->int_params["num_pc"], act_params[eLeakPC]);
	std::fill(threshPC.get(), threshPC.get() + cp->int_params["num_pc"], act_params[threshRestPC]);

	std::fill(pfSynWeightPC.get(), pfSynWeightPC.get()
		+ cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_gr_to_pc"], act_params[initSynWofGRtoPC]);

	std::fill(histPCPopAct.get(), histPCPopAct.get() + (int)derived_act_params[numPopHistBinsPC], 0);

	histPCPopActSum     = 0;
	histPCPopActCurBinN = 0;
	pcPopAct            = 0;

	// IO
	std::fill(threshIO.get(), threshIO.get() + cp->int_params["num_io"], act_params[threshRestIO]);
	std::fill(vIO.get(), vIO.get() + cp->int_params["num_io"], act_params[eLeakIO]);

	errDrive = 0;
	
	// NC
	noLTPMFNC = 0;
	noLTDMFNC = 0;

	std::fill(threshNC.get(), threshNC.get() + cp->int_params["num_nc"], act_params[threshRestNC]);
	std::fill(vNC.get(), vNC.get() + cp->int_params["num_nc"], act_params[eLeakNC]);

	std::fill(mfSynWeightNC.get(), mfSynWeightNC.get()
		+ cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"], act_params[initSynWofMFtoNC]);
}

void MZoneActivityState::stateRW(ConnectivityParams *cp, bool read, std::fstream &file)
{
	// basket cells
	rawBytesRW((char *)apBC.get(), cp->int_params["num_bc"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufBC.get(), cp->int_params["num_bc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCBC.get(), cp->int_params["num_bc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)gPFBC.get(), cp->int_params["num_bc"] * sizeof(float), read, file);
	rawBytesRW((char *)gPCBC.get(), cp->int_params["num_bc"] * sizeof(float), read, file);
	rawBytesRW((char *)vBC.get(), cp->int_params["num_bc"] * sizeof(float), read, file);
	rawBytesRW((char *)threshBC.get(), cp->int_params["num_bc"] * sizeof(float), read, file);

	// purkinje cells
	rawBytesRW((char *)apPC.get(), cp->int_params["num_pc"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufPC.get(), cp->int_params["num_pc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputBCPC.get(), cp->int_params["num_pc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputSCPC.get(),
		cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_sc_to_pc"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)pfSynWeightPC.get(),
		cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_gr_to_pc"] * sizeof(float), read, file);
	rawBytesRW((char *)inputSumPFPC.get(), cp->int_params["num_pc"] * sizeof(float), read, file);
	rawBytesRW((char *)gPFPC.get(), cp->int_params["num_pc"] * sizeof(float), read, file);
	rawBytesRW((char *)gBCPC.get(), cp->int_params["num_pc"] * sizeof(float), read, file);
	rawBytesRW((char *)gSCPC.get(),
		cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_sc_to_pc"] * sizeof(float), read, file);
	rawBytesRW((char *)vPC.get(), cp->int_params["num_pc"] * sizeof(float), read, file);
	rawBytesRW((char *)threshPC.get(), cp->int_params["num_pc"] * sizeof(float), read, file);
	rawBytesRW((char *)histPCPopAct.get(), numPopHistBinsPC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)&histPCPopActSum, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActCurBinN, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&pcPopAct, sizeof(ct_uint32_t), read, file);
	
	// inferior olivary cells
	rawBytesRW((char *)apIO.get(), cp->int_params["num_io"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufIO.get(), cp->int_params["num_io"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)inputNCIO.get(),
		cp->int_params["num_io"] * cp->int_params["num_p_io_from_nc_to_io"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)gNCIO.get(),
		cp->int_params["num_io"] * cp->int_params["num_p_io_from_nc_to_io"] * sizeof(float), read, file);
	rawBytesRW((char *)threshIO.get(), cp->int_params["num_io"] * sizeof(float), read, file);
	rawBytesRW((char *)vIO.get(), cp->int_params["num_io"] * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleIO.get(), cp->int_params["num_io"] * sizeof(float), read, file);
	rawBytesRW((char *)pfPCPlastTimerIO.get(), cp->int_params["num_io"] * sizeof(ct_int32_t), read, file);

	rawBytesRW((char *)&errDrive, sizeof(float), read, file);

	// nucleus cells
	rawBytesRW((char *)apNC.get(), cp->int_params["num_nc"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufNC.get(), cp->int_params["num_nc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCNC.get(),
		cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_pc_to_nc"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)inputMFNC.get(),
		cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"] * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)gPCNC.get(),
		cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_pc_to_nc"] * sizeof(float), read, file);
	rawBytesRW((char *)mfSynWeightNC.get(),
		cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"] * sizeof(float), read, file);
	rawBytesRW((char *)gMFAMPANC.get(),
		cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"] * sizeof(float), read, file);
	rawBytesRW((char *)threshNC.get(), cp->int_params["num_nc"] * sizeof(float), read, file);
	rawBytesRW((char *)vNC.get(), cp->int_params["num_nc"] * sizeof(float), read, file);
	rawBytesRW((char *)synIOPReleaseNC.get(), cp->int_params["num_nc"] * sizeof(float), read, file);

	rawBytesRW((char *)&noLTPMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&noLTDMFNC, sizeof(ct_uint8_t), read, file);
}

