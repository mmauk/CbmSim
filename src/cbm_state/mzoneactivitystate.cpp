/*
 * mzoneactivitystate.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: consciousness
 */

#include <iostream>
#include <algorithm> /* std::fill */

#include "assert.h"
#include "array_util.h"
#include "file_utility.h"
#include "sfmt.h"
#include "connectivityparams.h"
#include "activityparams.h"
#include "mzoneactivitystate.h"

MZoneActivityState::MZoneActivityState() {}

MZoneActivityState::MZoneActivityState(int randSeed)
{
	LOG_DEBUG("Allocating and initializing mzone activity state...");
	allocateMemory();
	initializeVals(randSeed);
	LOG_DEBUG("Finished allocating and initializing mzone activity state.");
}

MZoneActivityState::MZoneActivityState(std::fstream &infile)
{
	allocateMemory();
	stateRW(true, infile);
}

MZoneActivityState::~MZoneActivityState() {}

void MZoneActivityState::readState(std::fstream &infile)
{
	stateRW(true, infile);
}

void MZoneActivityState::writeState(std::fstream &outfile)
{
	stateRW(false, outfile);
}

bool MZoneActivityState::inInitialState()
{
  // stellate cells
  assert(arr_filled_with_int_t<uint32_t>(apBufSC.get(), num_sc, 0),
        "ERROR: apBufSC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gPFSC.get(), num_sc, 0),
        "ERROR: gPFSC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(vSC.get(), num_sc, 0),
        "ERROR: vSC not all zero", __func__);

  // basket cells
  assert(arr_filled_with_int_t<uint8_t>(apBC.get(), num_bc, 0),
        "ERROR: apBC not all zero", __func__);
  assert(arr_filled_with_int_t<uint32_t>(apBufBC.get(), num_bc, 0),
        "ERROR: apBufBC not all zero", __func__);
  assert(arr_filled_with_int_t<uint32_t>(inputPCBC.get(), num_bc, 0),
        "ERROR: inputPCBC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gPFBC.get(), num_bc, 0),
        "ERROR: gPFBC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gPCBC.get(), num_bc, 0),
        "ERROR: gPCBC not all zero", __func__);

  // purkinje cells
  assert(arr_filled_with_int_t<uint8_t>(apPC.get(), num_pc, 0),
        "ERROR: apPC not all zero", __func__);
  assert(arr_filled_with_int_t<uint32_t>(apBufPC.get(), num_pc, 0),
        "ERROR: apBufPC not all zero", __func__);
  assert(arr_filled_with_int_t<uint32_t>(inputBCPC.get(), num_pc, 0),
        "ERROR: inputBCPC not all zero", __func__);
  assert(arr_filled_with_int_t<uint32_t>(inputSCPC.get(), num_pc, 0),
        "ERROR: inputSCPC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(inputSumPFPC.get(), num_pc, 0),
        "ERROR: inputSumPFPC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gPFPC.get(), num_pc, 0),
        "ERROR: gPFPC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gBCPC.get(), num_pc, 0),
        "ERROR: gBCPC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gSCPC.get(), num_pc, 0),
        "ERROR: gSCPC not all zero", __func__);

  // inferior olive cells
  assert(arr_filled_with_int_t<uint8_t>(apIO.get(), num_io, 0),
        "ERROR: apIO not all zero", __func__);
  assert(arr_filled_with_int_t<uint8_t>(apBufIO.get(), num_io, 0),
        "ERROR: apBufIO not all zero", __func__);
  assert(arr_filled_with_int_t<uint8_t>(inputNCIO.get(), num_io * num_p_io_from_nc_to_io, 0),
        "ERROR: inputNCIO not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gNCIO.get(), num_io * num_p_io_from_nc_to_io, 0),
        "ERROR: gNCIO not all zero", __func__);
  assert(arr_filled_with_float_t<float>(vCoupleIO.get(), num_io, 0),
        "ERROR: vCoupleIO not all zero", __func__);
  assert(arr_filled_with_int_t<int32_t>(pfPCPlastTimerIO.get(), num_io, 0),
        "ERROR: pfPCPlastTimerIO not all zero", __func__);

  // nucleus cells
  assert(arr_filled_with_int_t<uint8_t>(apNC.get(), num_nc, 0),
        "ERROR: apNC not all zero", __func__);
  assert(arr_filled_with_int_t<uint32_t>(apBufNC.get(), num_nc, 0),
        "ERROR: apBufNC not all zero", __func__);
  assert(arr_filled_with_int_t<uint8_t>(inputPCNC.get(), num_nc * num_p_nc_from_pc_to_nc, 0),
        "ERROR: inputPCNC not all zero", __func__);
  assert(arr_filled_with_int_t<uint8_t>(inputMFNC.get(), num_nc * num_p_nc_from_mf_to_nc, 0),
        "ERROR: inputMFNC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gPCNC.get(), num_nc * num_p_nc_from_pc_to_nc, 0),
        "ERROR: gPCNC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(gMFAMPANC.get(), num_nc * num_p_nc_from_mf_to_nc, 0),
        "ERROR: gMFAMPANC not all zero", __func__);
  assert(arr_filled_with_float_t<float>(synIOPReleaseNC.get(), num_nc, 0),
        "ERROR: synIOPReleaseNC not all zero", __func__);

  // validate non-zero initialized values

  assert(arr_filled_with_int_t<uint8_t>(apSC.get(), num_sc, eLeakSC),
        "ERROR: apSC not all eLeakSC", __func__);
  assert(arr_filled_with_float_t<float>(threshSC.get(), num_sc, threshRestSC),
        "ERROR: threshRestSC not all zero", __func__);

  assert(arr_filled_with_int_t<uint8_t>(apBC.get(), num_bc, eLeakBC),
        "ERROR: apBC not all eLeakBC", __func__);
  assert(arr_filled_with_float_t<float>(threshBC.get(), num_bc, threshRestBC),
        "ERROR: threshRestBC not all zero", __func__);

  assert(arr_filled_with_float_t<float>(vPC.get(), num_pc, eLeakPC),
        "ERROR: vPC not all eLeakPC", __func__);
  assert(arr_filled_with_float_t<float>(threshPC.get(), num_pc, threshRestPC),
        "ERROR: threshPC not all threshRestPC", __func__);

  // below is tricky due to dichotomoy between graded and rest about how to initialize weights
  assert(arr_filled_with_float_t<float>(pfSynWeightPC.get(), num_pc * num_p_pc_from_gr_to_pc, initSynWofGRtoPC),
        "ERROR: pfSynWeightPC not all initSynWofGRtoPC", __func__);

  assert(histPCPopActSum == 0, "ERROR: histPCPopActSum not set to zero", __func__);
  assert(histPCPopActCurBinN == 0, "ERROR: histPCPopActCurBinN not set to zero", __func__);
  assert(pcPopAct == 0, "ERROR: pcPopAct not set to zero", __func__);

  assert(arr_filled_with_float_t<float>(vIO.get(), num_io, eLeakIO),
        "ERROR: vIO not all eLeakIO", __func__);
  assert(arr_filled_with_float_t<float>(threshIO.get(), num_io, threshRestIO),
        "ERROR: threshIO not all threshRestIO", __func__);

  assert(errDrive == 0, "ERROR: errDrive not set to zero", __func__);
  assert(noLTPMFNC == 0, "ERROR: noLTPMFNC not set to zero", __func__);
  assert(noLTDMFNC == 0, "ERROR: noLTDMFNC not set to zero", __func__);

  assert(arr_filled_with_float_t<float>(vNC.get(), num_nc, eLeakNC),
        "ERROR: vNC not all eLeakNC", __func__);
  assert(arr_filled_with_float_t<float>(threshNC.get(), num_nc, threshRestNC),
        "ERROR: threshNC not all threshRestNC", __func__);
  assert(arr_filled_with_float_t<float>(mfSynWeightNC.get(), num_nc * num_p_nc_from_mf_to_nc, initSynWofMFtoNC),
        "ERROR: mfSynWeightNC not all initSynWofMFtoNC", __func__);

  return true;
}

void MZoneActivityState::allocateMemory()
{
	// stellate cells
	apSC           = std::make_unique<uint8_t[]>(num_sc);
	apBufSC        = std::make_unique<uint32_t[]>(num_sc);
	gPFSC          = std::make_unique<float[]>(num_sc);
	threshSC       = std::make_unique<float[]>(num_sc);
	vSC            = std::make_unique<float[]>(num_sc);

	// basket cells
	apBC      = std::make_unique<uint8_t[]>(num_bc);
	apBufBC   = std::make_unique<uint32_t[]>(num_bc);
	inputPCBC = std::make_unique<uint32_t[]>(num_bc);
	gPFBC     = std::make_unique<float[]>(num_bc);
	gPCBC     = std::make_unique<float[]>(num_bc);
	vBC       = std::make_unique<float[]>(num_bc);
	threshBC  = std::make_unique<float[]>(num_bc);

	// purkinje cells
	apPC          = std::make_unique<uint8_t[]>(num_pc);
	apBufPC       = std::make_unique<uint32_t[]>(num_pc);
	inputBCPC     = std::make_unique<uint32_t[]>(num_pc);
	inputSCPC     = std::make_unique<uint32_t[]>(num_pc);
	pfSynWeightPC = std::make_unique<float[]>(num_pc * num_p_pc_from_gr_to_pc);
	inputSumPFPC  = std::make_unique<float[]>(num_pc);
	gPFPC         = std::make_unique<float[]>(num_pc);
	gBCPC         = std::make_unique<float[]>(num_pc);
	gSCPC         = std::make_unique<float[]>(num_pc);
	vPC           = std::make_unique<float[]>(num_pc);
	threshPC      = std::make_unique<float[]>(num_pc);
	histPCPopAct = std::make_unique<uint32_t[]>(numPopHistBinsPC);

	// inferior olivary cells
	apIO      = std::make_unique<uint8_t[]>(num_io);
	apBufIO   = std::make_unique<uint8_t[]>(num_io); // TODO: make 32 type
	inputNCIO = std::make_unique<uint8_t[]>(num_io * num_p_io_from_nc_to_io);
	gNCIO     = std::make_unique<float[]>(num_io * num_p_io_from_nc_to_io);
	threshIO  = std::make_unique<float[]>(num_io);
	vIO       = std::make_unique<float[]>(num_io);
	vCoupleIO = std::make_unique<float[]>(num_io);
	pfPCPlastTimerIO = std::make_unique<int32_t[]>(num_io);

	// nucleus cells
	apNC          = std::make_unique<uint8_t[]>(num_nc);
	apBufNC       = std::make_unique<uint32_t[]>(num_nc);
	inputPCNC     = std::make_unique<uint8_t[]>(num_nc * num_p_nc_from_pc_to_nc);
	inputMFNC     = std::make_unique<uint8_t[]>(num_nc * num_p_nc_from_mf_to_nc);
	gPCNC         = std::make_unique<float[]>(num_nc * num_p_nc_from_pc_to_nc);
	mfSynWeightNC = std::make_unique<float[]>(num_nc * num_p_nc_from_mf_to_nc);
	gMFAMPANC     = std::make_unique<float[]>(num_nc * num_p_nc_from_mf_to_nc);
	threshNC      = std::make_unique<float[]>(num_nc);
	vNC           = std::make_unique<float[]>(num_nc);
	synIOPReleaseNC = std::make_unique<float[]>(num_nc);
}

void MZoneActivityState::initializeVals(int randSeed)
{
	// only actively initializing those arrays whose initial values we want
	// differ from the default initilizer value

	// sc
	std::fill(vSC.get(), vSC.get() + num_sc, eLeakSC);
	std::fill(threshSC.get(), threshSC.get() + num_sc, threshRestSC);

	// bc
	std::fill(vBC.get(), vBC.get() + num_bc, eLeakBC);
	std::fill(threshBC.get(), threshBC.get() + num_bc, threshRestBC);

	// pc
	std::fill(vPC.get(), vPC.get() + num_pc, eLeakPC);
	std::fill(threshPC.get(), threshPC.get() + num_pc, threshRestPC);

	std::fill(pfSynWeightPC.get(), pfSynWeightPC.get()
		+ num_pc * num_p_pc_from_gr_to_pc, initSynWofGRtoPC);

  // TODO: remove
	std::fill(histPCPopAct.get(), histPCPopAct.get() + (int)numPopHistBinsPC, 0);

	histPCPopActSum     = 0;
	histPCPopActCurBinN = 0;
	pcPopAct            = 0;

	// IO
	std::fill(vIO.get(), vIO.get() + num_io, eLeakIO);
	std::fill(threshIO.get(), threshIO.get() + num_io, threshRestIO);

	errDrive = 0;
	
	// NC
	noLTPMFNC = 0;
	noLTDMFNC = 0;

	std::fill(vNC.get(), vNC.get() + num_nc, eLeakNC);
	std::fill(threshNC.get(), threshNC.get() + num_nc, threshRestNC);

	std::fill(mfSynWeightNC.get(), mfSynWeightNC.get()
		+ num_nc * num_p_nc_from_mf_to_nc, initSynWofMFtoNC);
}

void MZoneActivityState::stateRW(bool read, std::fstream &file)
{
	// stellate cells
	rawBytesRW((char *)apSC.get(), num_sc * sizeof(uint8_t), read, file);
	rawBytesRW((char *)apBufSC.get(), num_sc * sizeof(uint32_t), read, file);
	rawBytesRW((char *)gPFSC.get(), num_sc * sizeof(float), read, file);
	rawBytesRW((char *)threshSC.get(), num_sc * sizeof(float), read, file);
	rawBytesRW((char *)vSC.get(), num_sc * sizeof(float), read, file);

	// basket cells
	rawBytesRW((char *)apBC.get(), num_bc * sizeof(uint8_t), read, file);
	rawBytesRW((char *)apBufBC.get(), num_bc * sizeof(uint32_t), read, file);
	rawBytesRW((char *)inputPCBC.get(), num_bc * sizeof(uint32_t), read, file);
	rawBytesRW((char *)gPFBC.get(), num_bc * sizeof(float), read, file);
	rawBytesRW((char *)gPCBC.get(), num_bc * sizeof(float), read, file);
	rawBytesRW((char *)vBC.get(), num_bc * sizeof(float), read, file);
	rawBytesRW((char *)threshBC.get(), num_bc * sizeof(float), read, file);

	// purkinje cells
	rawBytesRW((char *)apPC.get(), num_pc * sizeof(uint8_t), read, file);
	rawBytesRW((char *)apBufPC.get(), num_pc * sizeof(uint32_t), read, file);
	rawBytesRW((char *)inputBCPC.get(), num_pc * sizeof(uint32_t), read, file);
	rawBytesRW((char *)inputSCPC.get(), num_pc * sizeof(uint32_t), read, file);
	rawBytesRW((char *)pfSynWeightPC.get(),
		num_pc * num_p_pc_from_gr_to_pc * sizeof(float), read, file);
	rawBytesRW((char *)inputSumPFPC.get(), num_pc * sizeof(float), read, file);
	rawBytesRW((char *)gPFPC.get(), num_pc * sizeof(float), read, file);
	rawBytesRW((char *)gBCPC.get(), num_pc * sizeof(float), read, file);
	rawBytesRW((char *)gSCPC.get(), num_pc * sizeof(float), read, file);
	rawBytesRW((char *)vPC.get(), num_pc * sizeof(float), read, file);
	rawBytesRW((char *)threshPC.get(), num_pc * sizeof(float), read, file);
	rawBytesRW((char *)histPCPopAct.get(), numPopHistBinsPC * sizeof(uint32_t), read, file);

	rawBytesRW((char *)&histPCPopActSum, sizeof(uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActCurBinN, sizeof(uint32_t), read, file);
	rawBytesRW((char *)&pcPopAct, sizeof(uint32_t), read, file);
	
	// inferior olivary cells
	rawBytesRW((char *)apIO.get(), num_io * sizeof(uint8_t), read, file);
	rawBytesRW((char *)apBufIO.get(), num_io * sizeof(uint8_t), read, file);
	rawBytesRW((char *)inputNCIO.get(),
		num_io * num_p_io_from_nc_to_io * sizeof(uint8_t), read, file);
	rawBytesRW((char *)gNCIO.get(),
		num_io * num_p_io_from_nc_to_io * sizeof(float), read, file);
	rawBytesRW((char *)threshIO.get(), num_io * sizeof(float), read, file);
	rawBytesRW((char *)vIO.get(), num_io * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleIO.get(), num_io * sizeof(float), read, file);
	rawBytesRW((char *)pfPCPlastTimerIO.get(), num_io * sizeof(int32_t), read, file);

	rawBytesRW((char *)&errDrive, sizeof(float), read, file);

	// nucleus cells
	rawBytesRW((char *)apNC.get(), num_nc * sizeof(uint8_t), read, file);
	rawBytesRW((char *)apBufNC.get(), num_nc * sizeof(uint32_t), read, file);
	rawBytesRW((char *)inputPCNC.get(),
		num_nc * num_p_nc_from_pc_to_nc * sizeof(uint8_t), read, file);
	rawBytesRW((char *)inputMFNC.get(),
		num_nc * num_p_nc_from_mf_to_nc * sizeof(uint8_t), read, file);
	rawBytesRW((char *)gPCNC.get(),
		num_nc * num_p_nc_from_pc_to_nc * sizeof(float), read, file);
	rawBytesRW((char *)mfSynWeightNC.get(),
		num_nc * num_p_nc_from_mf_to_nc * sizeof(float), read, file);
	rawBytesRW((char *)gMFAMPANC.get(),
		num_nc * num_p_nc_from_mf_to_nc * sizeof(float), read, file);
	rawBytesRW((char *)threshNC.get(), num_nc * sizeof(float), read, file);
	rawBytesRW((char *)vNC.get(), num_nc * sizeof(float), read, file);
	rawBytesRW((char *)synIOPReleaseNC.get(), num_nc * sizeof(float), read, file);

	rawBytesRW((char *)&noLTPMFNC, sizeof(uint8_t), read, file);
	rawBytesRW((char *)&noLTDMFNC, sizeof(uint8_t), read, file);
}

