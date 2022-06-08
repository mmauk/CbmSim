/*
 * innetactivitystate.h
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#ifndef INNETACTIVITYSTATE_H_
#define INNETACTIVITYSTATE_H_

#include <fstream>

#include <memoryMgmt/dynamic2darray.h>
#include <memoryMgmt/arrayvalidate.h>
#include <fileIO/rawbytesrw.h>
#include <stdDefinitions/pstdint.h>

#include "params/connectivityparams.h"
#include "params/activityparams.h"
#include "innetconnectivitystate.h"


class InNetActivityState
{
public:
	InNetActivityState();
	InNetActivityState(ConnectivityParams &cp, ActivityParams *ap);
	InNetActivityState(ConnectivityParams &cp, std::fstream &infile);
	//InNetActivityState(const InNetActivityState &state);

	virtual ~InNetActivityState();

	void writeState(ConnectivityParams &cp, std::fstream &outfile);

	bool state_equal(ConnectivityParams &cp, const InNetActivityState &compState);
	bool state_unequal(ConnectivityParams &cp, const InNetActivityState &compState);

	//bool validateState();

	void resetState(ConnectivityParams &cp, ActivityParams *ap);

	//mossy fiber
	ct_uint8_t histMF[cp.NUM_MF]();
	ct_uint32_t apBufMF[cp.NUM_MF]();
	float depAmpMFUBC[cp.NUM_MF]();

	//golgi cells
	int goTimeStep;
	float synWscalerGRtoGO[cp.NUM_GO]();
	float synWscalerGOtoGO[cp.NUM_GO]();
	int goSpkSumHP[cp.NUM_GO]();
	float goFR_HP[cp.NUM_GO]();
	ct_uint8_t apGO[cp.NUM_GO]();
	ct_uint32_t apBufGO[cp.NUM_GO]();
	int spkGO[cp.NUM_GO]();
	float vGO[cp.NUM_GO]();
	float exGOInput[cp.NUM_GO]();
	float inhGOInput[cp.NUM_GO]();
	float vCoupleGO[cp.NUM_GO]();
	float threshCurGO[cp.NUM_GO]();
	ct_uint32_t inputMFGO[cp.NUM_GO]();
	ct_uint32_t inputUBCGO[cp.NUM_GO]();
	float depAmpMFGO[cp.NUM_MF]();
	float gi_MFtoGO[cp.NUM_MF]();
	float gSum_MFGO[cp.NUM_GO]();
	ct_uint32_t inputGOGO[cp.NUM_GO]();
	
	float gi_GOtoGO[cp.NUM_GO]();
	float depAmpGOGO[cp.NUM_GO]();
	float gSum_GOGO[cp.NUM_GO]();
	float depAmpGOGR[cp.NUM_GO]();
	float dynamicAmpGOGR[cp.NUM_GO]();

	float gSum_UBCtoGO[cp.NUM_GO]();

	float vSum_GOGO[cp.NUM_GO]();
	float vSum_GRGO[cp.NUM_GO]();
	float vSum_MFGO[cp.NUM_GO]();


	//todo: synaptic depression test
	float inputGOGABASynDepGO[cp.NUM_GO]();
	float goGABAOutSynScaleGOGO[cp.NUM_GO]();

	float gMFGO[cp.NUM_GO]();
	float gNMDAMFGO[cp.NUM_GO]();
	float gNMDAUBCGO[cp.NUM_GO]();
	float gNMDAIncMFGO[cp.NUM_GO]();
	float gGRGO[cp.NUM_GO]();
	float gGRGO_NMDA[cp.NUM_GO]();
	float gGOGO[cp.NUM_GO]();
	float gMGluRGO[cp.NUM_GO]();
	float gMGluRIncGO[cp.NUM_GO]();
	float mGluRGO[cp.NUM_GO]();
	float gluGO[cp.NUM_GO]();

	//granule cells
	float depAmpMFGR[cp.NUM_MF]();
	float gi_MFtoGR[cp.NUM_MF]();
	float gSum_MFGR[cp.NUM_GR]();
	
	ct_uint8_t apGR[cp.NUM_GR]();
	ct_uint32_t apBufGR[cp.NUM_GR]();
	float gMFGR[cp.NUM_GR][cp.MAX_NUM_P_GR_FROM_MF_TO_GR]();
	float gUBCGR[cp.NUM_GR][cp.MAX_NUM_P_GR_FROM_MF_TO_GR]();
	float gMFSumGR[cp.NUM_GR]();
	float gMFDirectGR[cp.NUM_GR]();
	float gMFSpilloverGR[cp.NUM_GR]();
	float gGODirectGR[cp.NUM_GR]();
	float gGOSpilloverGR[cp.NUM_GR]();
	int apMFtoGR[cp.NUM_GR]();
	int apUBCtoGR[cp.NUM_GR]();
	float gUBCSumGR[cp.NUM_GR]();
	float gUBCDirectGR[cp.NUM_GR]();
	float gUBCSpilloverGR[cp.NUM_GR]();
	float gNMDAGR[cp.NUM_GR]();
	float gNMDAIncGR[cp.NUM_GR]();
	float gLeakGR[cp.NUM_GR]();
	float depAmpMFtoGR[cp.NUM_GR]();
	float depAmpUBCtoGR[cp.NUM_GR]();
	float depAmpGOtoGR[cp.NUM_GR](); 
	float dynamicAmpGOtoGR[cp.NUM_GR]();

	float gGOGR[cp.NUM_GR][cp.MAX_NUM_P_GR_FROM_GO_TO_GR]();
	float gGOSumGR[cp.NUM_GR]();
	float threshGR[cp.NUM_GR]();
	float vGR[cp.NUM_GR]();
	float gKCaGR[cp.NUM_GR]();
	ct_uint64_t historyGR[cp.NUM_GR]();

	//stellate cells
	ct_uint8_t apSC[cp.NUM_SC]();
	ct_uint32_t apBufSC[cp.NUM_SC]();

	float gPFSC[cp.NUM_SC]();
	float threshSC[cp.NUM_SC]();
	float vSC[cp.NUM_SC]();
	ct_uint32_t inputSumPFSC[cp.NUM_SC]();

	//UBCs
	float gRise_MFtoUBC[cp.NUM_UBC]();
	float gDecay_MFtoUBC[cp.NUM_UBC]();
	float gSum_MFtoUBC[cp.NUM_UBC]();
	float depAmpUBCtoUBC[cp.NUM_UBC]();
	float gRise_UBCNMDA[cp.NUM_UBC]();
	float gDecay_UBCNMDA[cp.NUM_UBC]();
	float gSum_UBCNMDA[cp.NUM_UBC]();
	float gK_UBC[cp.NUM_UBC]();
	
	float gRise_UBCtoUBC[cp.NUM_UBC]();
	float gDecay_UBCtoUBC[cp.NUM_UBC]();
	float gSumOutUBCtoUBC[cp.NUM_UBC]();
	float gSumInUBCtoUBC[cp.NUM_UBC]();
	
	int inputMFUBC[cp.NUM_UBC]();
	int inputGOUBC[cp.NUM_UBC]();
	float gSum_GOtoUBC[cp.NUM_UBC]();
	ct_uint8_t apUBC[cp.NUM_UBC]();
	float vUBC[cp.NUM_UBC]();
	float threshUBC[cp.NUM_UBC]();
	int inputUBCtoUBC[cp.NUM_UBC]();

	float gi_UBCtoGO[cp.NUM_UBC]();
	float depAmpUBCGO[cp.NUM_UBC]();
	float depAmpUBCGR[cp.NUM_UBC]();

private:

	void allocateMemory();
	void stateRW(ConnectivityParams &cp, bool read, std::fstream &file);
	void initializeVals(ConnectivityParams &cp, ActivityParams *ap);
};


#endif /* INNETACTIVITYSTATE_H_ */
