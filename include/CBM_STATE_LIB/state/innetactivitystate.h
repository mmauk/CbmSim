/*
 * innetactivitystate.h
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#ifndef INNETACTIVITYSTATE_H_
#define INNETACTIVITYSTATE_H_

#include <memory>
#include <fstream>
#include <fileIO/rawbytesrw.h>
#include <stdDefinitions/pstdint.h>
#include "params/connectivityparams.h"
#include "params/activityparams.h"


class InNetActivityState
{
public:
	InNetActivityState();
	InNetActivityState(ActivityParams &ap);
	InNetActivityState(std::fstream &infile);

	virtual ~InNetActivityState();

	void writeState(std::fstream &outfile);
	void resetState(ActivityParams &ap);

	//mossy fiber
	std::unique_ptr<ct_uint8_t[]> histMF{nullptr};
	std::unique_ptr<ct_uint32_t[]> apBufMF{nullptr};

	//golgi cells
	int goTimeStep;
	std::unique_ptr<float[]> synWscalerGRtoGO{nullptr};
	std::unique_ptr<ct_uint8_t[]> apGO{nullptr};
	std::unique_ptr<ct_uint32_t[]> apBufGO{nullptr};
	std::unique_ptr<float[]> vGO{nullptr};
	std::unique_ptr<float[]> vCoupleGO{nullptr};
	std::unique_ptr<float[]> threshCurGO{nullptr};
	std::unique_ptr<ct_uint32_t[]> inputMFGO{nullptr};
	ct_uint32_t inputUBCGO[NUM_GO] = {0};
	float depAmpMFGO[NUM_MF] = {0.0};
	float gi_MFtoGO[NUM_MF] = {0.0};
	float gSum_MFGO[NUM_GO] = {0.0};
	ct_uint32_t inputGOGO[NUM_GO] = {0};
	
	float gi_GOtoGO[NUM_GO] = {0.0};
	float depAmpGOGO[NUM_GO] = {0.0};
	float gSum_GOGO[NUM_GO] = {0.0};
	float depAmpGOGR[NUM_GO] = {0.0};
	float dynamicAmpGOGR[NUM_GO] = {0.0};

	float gSum_UBCtoGO[NUM_GO] = {0.0};

	float vSum_GOGO[NUM_GO] = {0.0};
	float vSum_GRGO[NUM_GO] = {0.0};
	float vSum_MFGO[NUM_GO] = {0.0};


	//todo: synaptic depression test
	float inputGOGABASynDepGO[NUM_GO] = {0.0};
	float goGABAOutSynScaleGOGO[NUM_GO] = {0.0};

	//NOTE: removed NMDA UBC GO conductance 06/15/2022
	float gMFGO[NUM_GO] = {0.0};
	float gNMDAMFGO[NUM_GO] = {0.0};
	float gNMDAIncMFGO[NUM_GO] = {0.0};
	float gGRGO[NUM_GO] = {0.0};
	float gGRGO_NMDA[NUM_GO] = {0.0};
	float gGOGO[NUM_GO] = {0.0};
	float gMGluRGO[NUM_GO] = {0.0};
	float gMGluRIncGO[NUM_GO] = {0.0};
	float mGluRGO[NUM_GO] = {0.0};
	float gluGO[NUM_GO] = {0.0};

	//granule cells
	float depAmpMFGR[NUM_MF] = {0.0};
	float gi_MFtoGR[NUM_MF] = {0.0};
	float gSum_MFGR[NUM_GR] = {0.0};
	
	ct_uint8_t apGR[NUM_GR] = {0};
	ct_uint32_t apBufGR[NUM_GR] = {0};
	float gMFGR[NUM_GR][MAX_NUM_P_GR_FROM_MF_TO_GR] = {0.0};
	float gUBCGR[NUM_GR][MAX_NUM_P_GR_FROM_MF_TO_GR] = {0.0};
	float gMFSumGR[NUM_GR] = {0.0};
	float gMFDirectGR[NUM_GR] = {0.0};
	float gMFSpilloverGR[NUM_GR] = {0.0};
	float gGODirectGR[NUM_GR] = {0.0};
	float gGOSpilloverGR[NUM_GR] = {0.0};
	int apMFtoGR[NUM_GR] = {0};
	int apUBCtoGR[NUM_GR] = {0};
	float gUBCSumGR[NUM_GR] = {0.0};
	float gUBCDirectGR[NUM_GR] = {0.0};
	float gUBCSpilloverGR[NUM_GR] = {0.0};
	float gNMDAGR[NUM_GR] = {0.0};
	float gNMDAIncGR[NUM_GR] = {0.0};
	float gLeakGR[NUM_GR] = {0.0};
	float depAmpMFtoGR[NUM_GR] = {0.0};
	float depAmpUBCtoGR[NUM_GR] = {0.0};
	float depAmpGOtoGR[NUM_GR] = {0.0}; 
	float dynamicAmpGOtoGR[NUM_GR] = {0.0};

	float gGOGR[NUM_GR][MAX_NUM_P_GR_FROM_GO_TO_GR] = {0.0};
	float gGOSumGR[NUM_GR] = {0.0};
	float threshGR[NUM_GR] = {0.0};
	float vGR[NUM_GR] = {0.0};
	float gKCaGR[NUM_GR] = {0.0};
	ct_uint64_t historyGR[NUM_GR] = {0};

	//stellate cells
	ct_uint8_t apSC[NUM_SC] = {0};
	ct_uint32_t apBufSC[NUM_SC] = {0};

	float gPFSC[NUM_SC] = {0.0};
	float threshSC[NUM_SC] = {0.0};
	float vSC[NUM_SC] = {0.0};
	ct_uint32_t inputSumPFSC[NUM_SC] = {0};

	//UBCs
	float gRise_MFtoUBC[NUM_UBC] = {0.0};
	float gDecay_MFtoUBC[NUM_UBC] = {0.0};
	float gSum_MFtoUBC[NUM_UBC] = {0.0};
	float depAmpUBCtoUBC[NUM_UBC] = {0.0};
	float gRise_UBCNMDA[NUM_UBC] = {0.0};
	float gDecay_UBCNMDA[NUM_UBC] = {0.0};
	float gSum_UBCNMDA[NUM_UBC] = {0.0};
	float gK_UBC[NUM_UBC] = {0.0};
	
	float gRise_UBCtoUBC[NUM_UBC] = {0.0};
	float gDecay_UBCtoUBC[NUM_UBC] = {0.0};
	float gSumOutUBCtoUBC[NUM_UBC] = {0.0};
	float gSumInUBCtoUBC[NUM_UBC] = {0.0};
	
	int inputMFUBC[NUM_UBC] = {0};
	int inputGOUBC[NUM_UBC] = {0};
	float gSum_GOtoUBC[NUM_UBC] = {0.0};
	ct_uint8_t apUBC[NUM_UBC] = {0};
	float vUBC[NUM_UBC] = {0.0};
	float threshUBC[NUM_UBC] = {0.0};
	int inputUBCtoUBC[NUM_UBC] = {0};

	float gi_UBCtoGO[NUM_UBC] = {0.0};
	float depAmpUBCGO[NUM_UBC] = {0.0};
	float depAmpUBCGR[NUM_UBC] = {0.0};

private:
	void stateRW(bool read, std::fstream &file);
	void allocateArrMem(ActivityParams &ap);
	void initializeVals(ActivityParams &ap);
};

#endif /* INNETACTIVITYSTATE_H_ */
