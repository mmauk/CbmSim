/*
 * mzoneactivitystate.cpp
 *
 *  Created on: Nov 27, 2012
 *      Author: consciousness
 */

#include "state/mzoneactivitystate.h"

MZoneActivityState::MZoneActivityState() {}

MZoneActivityState::MZoneActivityState(ActivityParams &ap, int randSeed)
{
	std::cout << "[INFO]: Initializing mzone activity state..." << std::endl;
	allocateMemory(ap.numPopHistBinsPC);
	initializeVals(ap, randSeed);
	std::cout << "[INFO]: Finished initializing mzone activity state." << std::endl;
}

MZoneActivityState::MZoneActivityState(ActivityParams &ap, std::fstream &infile)
{
	allocateMemory(ap.numPopHistBinsPC);
	stateRW(ap.numPopHistBinsPC, true, infile);
}

MZoneActivityState::~MZoneActivityState() {}

void MZoneActivityState::writeState(ActivityParams &ap, std::fstream &outfile)
{
	stateRW(ap.numPopHistBinsPC, false, outfile);
}

void MZoneActivityState::allocateMemory(ct_uint32_t numPopHistBinsPC)
{
	apBC      = std::make_unique<ct_uint8_t[]>(NUM_BC);
	apBufBC   = std::make_unique<ct_uint32_t[]>(NUM_BC);
	inputPCBC = std::make_unique<ct_uint32_t[]>(NUM_BC);
	gPFBC     = std::make_unique<float[]>(NUM_BC);
	gPCBC     = std::make_unique<float[]>(NUM_BC);
	vBC		  = std::make_unique<float[]>(NUM_BC);
	threshBC  = std::make_unique<float[]>(NUM_BC);

	apPC	  	  = std::make_unique<ct_uint8_t[]>(NUM_PC);
	apBufPC	  	  = std::make_unique<ct_uint32_t[]>(NUM_PC);
	inputBCPC 	  = std::make_unique<ct_uint32_t[]>(NUM_PC);
	inputSCPC 	  = std::make_unique<ct_uint8_t[]>(NUM_PC * NUM_P_PC_FROM_SC_TO_PC);
	pfSynWeightPC = std::make_unique<float[]>(NUM_PC * NUM_P_PC_FROM_GR_TO_PC);
	inputSumPFPC  = std::make_unique<float[]>(NUM_PC);
	gPFPC  		  = std::make_unique<float[]>(NUM_PC);
	gBCPC  		  = std::make_unique<float[]>(NUM_PC);
	gSCPC 	  	  = std::make_unique<float[]>(NUM_PC * NUM_P_PC_FROM_SC_TO_PC);
	vPC		  	  = std::make_unique<float[]>(NUM_PC);
	threshPC      = std::make_unique<float[]>(NUM_PC);

	histPCPopAct = std::make_unique<ct_uint32_t[]>(numPopHistBinsPC);

	apIO 	  = std::make_unique<ct_uint8_t[]>(NUM_IO);
	apBufIO   = std::make_unique<ct_uint8_t[]>(NUM_IO);
	inputNCIO = std::make_unique<ct_uint8_t[]>(NUM_IO * NUM_P_IO_FROM_NC_TO_IO);
	gNCIO 	  = std::make_unique<float[]>(NUM_IO * NUM_P_IO_FROM_NC_TO_IO);
	threshIO  = std::make_unique<float[]>(NUM_IO);
	vIO		  = std::make_unique<float[]>(NUM_IO);
	vCoupleIO = std::make_unique<float[]>(NUM_IO);

	pfPCPlastTimerIO = std::make_unique<ct_int32_t[]>(NUM_IO);
	
	apNC 	      = std::make_unique<ct_uint8_t[]>(NUM_NC);
	apBufNC       = std::make_unique<ct_uint32_t[]>(NUM_NC);
	inputPCNC 	  = std::make_unique<ct_uint8_t[]>(NUM_NC * NUM_P_NC_FROM_PC_TO_NC);
	inputMFNC 	  = std::make_unique<ct_uint8_t[]>(NUM_NC * NUM_P_NC_FROM_MF_TO_NC);
	gPCNC		  = std::make_unique<float[]>(NUM_NC * NUM_P_NC_FROM_PC_TO_NC);
	mfSynWeightNC = std::make_unique<float[]>(NUM_NC * NUM_P_NC_FROM_MF_TO_NC);
	gMFAMPANC 	  = std::make_unique<float[]>(NUM_NC * NUM_P_NC_FROM_MF_TO_NC);
	threshNC  	  = std::make_unique<float[]>(NUM_NC);
	vNC		  	  = std::make_unique<float[]>(NUM_NC);

	synIOPReleaseNC = std::make_unique<float[]>(NUM_NC);
}

void MZoneActivityState::initializeVals(ActivityParams &ap, int randSeed)
{
	//uncomment for actual runs 	
	//CRandomSFMT0 randGen(randSeed);

	// bc
	std::fill(threshBC.get(), threshBC.get() + NUM_BC, ap.threshRestBC);
	std::fill(vBC.get(), vBC.get() + NUM_BC, ap.eLeakBC);

	// pc
	std::fill(vPC.get(), vPC.get() + NUM_PC, ap.eLeakPC);
	std::fill(threshPC.get(), threshPC.get() + NUM_PC, ap.threshRestPC);	

	std::fill(pfSynWeightPC.get(), pfSynWeightPC.get()
		+ NUM_PC * NUM_P_PC_FROM_GR_TO_PC, ap.initSynWofGRtoPC);

	std::fill(histPCPopAct.get(), histPCPopAct.get() + ap.numPopHistBinsPC, 0);	

	histPCPopActSum		= 0;
	histPCPopActCurBinN = 0;
	pcPopAct			= 0;

	// IO
	std::fill(threshIO.get(), threshIO.get() + NUM_IO, ap.threshRestIO);
	std::fill(vIO.get(), vIO.get() + NUM_IO, ap.eLeakIO);

	errDrive = 0;
	
	// NC
	noLTPMFNC = 0;
	noLTDMFNC = 0;

	std::fill(threshNC.get(), threshNC.get() + NUM_NC, ap.threshRestNC);
	std::fill(vNC.get(), vNC.get() + NUM_NC, ap.eLeakNC);

	std::fill(mfSynWeightNC.get(), mfSynWeightNC.get()
		+ NUM_NC * NUM_P_NC_FROM_MF_TO_NC, ap.initSynWofMFtoNC);
}

void MZoneActivityState::stateRW(ct_uint32_t numPopHistBinsPC, bool read, std::fstream &file)
{
	rawBytesRW((char *)apBC.get(), NUM_BC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufBC.get(), NUM_BC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCBC.get(), NUM_BC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)gPFBC.get(), NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)gPCBC.get(), NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)threshBC.get(), NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)vBC.get(), NUM_BC * sizeof(float), read, file);
	rawBytesRW((char *)apPC.get(), NUM_PC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufPC.get(), NUM_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputBCPC.get(), NUM_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputSCPC.get(),
		NUM_PC * NUM_P_PC_FROM_SC_TO_PC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)pfSynWeightPC.get(),
		NUM_PC * NUM_P_PC_FROM_GR_TO_PC * sizeof(float), read, file);
	rawBytesRW((char *)inputSumPFPC.get(), NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gPFPC.get(), NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gBCPC.get(), NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)gSCPC.get(),
		NUM_PC * NUM_P_PC_FROM_SC_TO_PC * sizeof(float), read, file);
	rawBytesRW((char *)vPC.get(), NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)threshPC.get(), NUM_PC * sizeof(float), read, file);
	rawBytesRW((char *)histPCPopAct.get(), numPopHistBinsPC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActSum, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&histPCPopActCurBinN, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)&pcPopAct, sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)apIO.get(), NUM_IO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufIO.get(), NUM_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputNCIO.get(),
		NUM_IO * NUM_P_IO_FROM_NC_TO_IO * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&errDrive, sizeof(float), read, file);
	rawBytesRW((char *)gNCIO.get(),
		NUM_IO * NUM_P_IO_FROM_NC_TO_IO * sizeof(float), read, file);
	rawBytesRW((char *)threshIO.get(), NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)vIO.get(), NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)vCoupleIO.get(), NUM_IO * sizeof(float), read, file);
	rawBytesRW((char *)pfPCPlastTimerIO.get(), NUM_IO * sizeof(ct_int32_t), read, file);
	rawBytesRW((char *)&noLTPMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)&noLTDMFNC, sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apNC.get(), NUM_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)apBufNC.get(), NUM_NC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)inputPCNC.get(),
		NUM_NC * NUM_P_NC_FROM_PC_TO_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)inputMFNC.get(),
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(ct_uint8_t), read, file);
	rawBytesRW((char *)gPCNC.get(),
		NUM_NC * NUM_P_NC_FROM_PC_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)mfSynWeightNC.get(),
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)gMFAMPANC.get(),
		NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(float), read, file);
	rawBytesRW((char *)threshNC.get(), NUM_NC * sizeof(float), read, file);
	rawBytesRW((char *)vNC.get(), NUM_NC * sizeof(float), read, file);
	rawBytesRW((char *)synIOPReleaseNC.get(), NUM_NC * sizeof(float), read, file);
}

