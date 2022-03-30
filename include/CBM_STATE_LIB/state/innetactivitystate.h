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
	InNetActivityState(ConnectivityParams *conParams, ActivityParams *actParams);
	InNetActivityState(ConnectivityParams *conParams, std::fstream &infile);
	InNetActivityState(const InNetActivityState &state);

	virtual ~InNetActivityState();

	void writeState(std::fstream &outfile);

	bool operator==(const InNetActivityState &compState);
	bool operator!=(const InNetActivityState &compState);

	bool validateState();

	void resetState(ActivityParams *ap);

	ConnectivityParams *cp;
	

	//mossy fiber
	ct_uint8_t *histMF;
	ct_uint32_t *apBufMF;
	float *depAmpMFUBC;

	//golgi cells
	int goTimeStep;
	float *synWscalerGRtoGO;
	float *synWscalerGOtoGO;
	int *goSpkSumHP;
	float *goFR_HP;
	ct_uint8_t *apGO;
	int *spkGO;
	ct_uint32_t *apBufGO;
	float *vGO;
	float *exGOInput;
	float *inhGOInput;
	float *vCoupleGO;
	float *threshCurGO;
	ct_uint32_t *inputMFGO;
	ct_uint32_t *inputUBCGO;
	float *depAmpMFGO;
	float *gi_MFtoGO;
	float *gSum_MFGO;
	
	float *gi_GOtoGO;
	float *depAmpGOGO;
	float *gSum_GOGO;

	float *vSum_GOGO;
	float *vSum_GRGO;
	float *vSum_MFGO;




	float *depAmpGOGR;
	float *dynamicAmpGOGR;

	float *gSum_UBCtoGO;

	ct_uint32_t *inputGOGO;

	//todo: synaptic depression test
	float *inputGOGABASynDepGO;
	float *goGABAOutSynScaleGOGO;

	float *gMFGO;
	float *gNMDAMFGO;
	float *gNMDAUBCGO;
	float *gNMDAIncMFGO;
	float *gGRGO;
	float *gGRGO_NMDA;
	float *gGOGO;
	float *gMGluRGO;
	float *gMGluRIncGO;
	float *mGluRGO;
	float *gluGO;

	//granule cells
	ct_uint8_t *apGR;
	ct_uint32_t *apBufGR;

	
	float *depAmpMFGR;
	float *gi_MFtoGR;
	float *gSum_MFGR;
	
	
	float **gMFGR;
	float **gUBCGR;
	float *gMFSumGR;
	float *gMFDirectGR;
	float *gMFSpilloverGR;
	float *gGODirectGR;
	float *gGOSpilloverGR;
	int *apMFtoGR;
	int *apUBCtoGR;
	float *gUBCSumGR;
	float *gUBCDirectGR;
	float *gUBCSpilloverGR;
	float *gNMDAGR;
	float *gNMDAIncGR;
	float *gLeakGR;
	float *depAmpMFtoGR;
	float *depAmpUBCtoGR;
	float *depAmpGOtoGR; 
	float *dynamicAmpGOtoGR;

	float **gGOGR;
	float *gGOSumGR;

	float *threshGR;
	float *vGR;
	float *gKCaGR;
	ct_uint64_t *historyGR;

	//stellate cells
	ct_uint8_t *apSC;
	ct_uint32_t *apBufSC;

	float *gPFSC;
	float *threshSC;
	float *vSC;

	ct_uint32_t *inputSumPFSC;

	//UBCs
	
	float *gRise_MFtoUBC;
	float *gDecay_MFtoUBC;
	float *gSum_MFtoUBC;
	float *depAmpUBCtoUBC;

	float *gRise_UBCNMDA;
	float *gDecay_UBCNMDA;
	float *gSum_UBCNMDA;
	float *gK_UBC;
	
	float *gRise_UBCtoUBC;
	float *gDecay_UBCtoUBC;
	float *gSumOutUBCtoUBC;
	float *gSumInUBCtoUBC;
	
	ct_uint8_t *apUBC;
	int *inputMFUBC;
	int *inputGOUBC;
	float *gSum_GOtoUBC;
	float *vUBC;
	float *threshUBC;
	int *inputUBCtoUBC;

	float *gi_UBCtoGO;
	float *depAmpUBCGO;
	float *depAmpUBCGR;
private:
	InNetActivityState();

	void allocateMemory();

	void stateRW(bool read, std::fstream &file);

	void initializeVals(ActivityParams *ap);
};


#endif /* INNETACTIVITYSTATE_H_ */
