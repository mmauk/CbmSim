/*
 * activityparams.h
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#ifndef ACTIVITYPARAMS_H_
#define ACTIVITYPARAMS_H_

#include <math.h>
#include <stdlib.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include "fileIO/build_file.h"
#include <stdDefinitions/pstdint.h>

#define NUM_AP_PARAMS 131
#define NUM_DERIVED_AP_PARAMS 43

enum ap_name_id 
{
	coupleRiRjRatioGO,
	coupleRiRjRatioIO,
	eBCtoPC,
	eGABAGO,
	eGOGR,
	eLeakBC,
	eLeakGO,
	eLeakGR,
	eLeakIO, 
	eLeakNC,
	eLeakPC,
	eLeakSC,
	eMFGR,
	eMGluRGO,
	eNCtoIO, 
	ePCtoBC,
	ePCtoNC,
	eSCtoPC,
	gDecTauBCtoPC,
	gIncBCtoPC,
	gGABADecTauGOtoGO,
	gIncDirectGOtoGR,
	gDirectTauGOtoGR,
	gIncFracSpilloverGOtoGR,
	gSpilloverTauGOtoGR,
	gGABAIncGOtoGO,
	gDecTauGRtoGO,
	gIncGRtoGO,
	gDecTauMFtoGO,
	gIncMFtoGO,
	gConstGO,
	NMDA_AMPAratioMFGO,
	gDecTauMFtoGONMDA,
	gIncDirectMFtoGR,
	gDirectTauMFtoGR,
	gIncFracSpilloverMFtoGR,
	gSpilloverTauMFtoGR,
	recoveryTauMF,
	fracDepMF,
	recoveryTauGO,
	fracDepGO,
	gIncMFtoUBC,
	gIncGOtoUBC,
	gIncUBCtoUBC,
	gIncUBCtoGO,
	gIncUBCtoGR,
	gKIncUBC,
	gKTauUBC,
	gConstUBC,
	threshTauUBC,
	gMGluRDecGRtoGO,
	gMGluRIncDecayGO,
	gMGluRIncScaleGO,
	gMGluRScaleGRtoGO,
	gDecT0ofNCtoIO,
	gDecTSofNCtoIO,
	gDecTTofNCtoIO,
	gIncNCtoIO,
	gIncTauNCtoIO,
	gDecTauPCtoBC,
	gDecTauPCtoNC,
	gIncAvgPCtoNC,
	gDecTauGRtoBC,
	gDecTauGRtoPC,
	gDecTauGRtoSC,
	gIncGRtoPC,
	gDecTauSCtoPC,
	gIncSCtoPC,
	gluDecayGO,
	gluScaleGO,
	goGABAGOGOSynDepF,
	goGABAGOGOSynRecTau,
	synLTDStepSizeGRtoPC,
	synLTPStepSizeGRtoPC,
	mGluRDecayGO,
	mGluRScaleGO,
	maxExtIncVIO,
	gmaxAMPADecTauMFtoNC,
	synLTDStepSizeMFtoNC,
	synLTDPCPopActThreshMFtoNC,
	synLTPStepSizeMFtoNC,
	synLTPPCPopActThreshMFtoNC,
	gmaxNMDADecTauMFtoNC,
	initSynWofMFtoNC,
	msLTDDurationIO,
	msLTDStartAPIO,
	msLTPEndAPIO,
	msLTPStartAPIO,
	msPerHistBinGR,
	msPerHistBinMF,
	msPerTimeStep,
	relPDecT0ofNCtoIO,
	relPDecTSofNCtoIO,
	relPDecTTofNCtoIO,
	relPIncNCtoIO,
	relPIncTauNCtoIO,
	gIncPCtoBC,
	gIncGRtoBC,
	gIncGRtoSC,
	initSynWofGRtoPC,
	rawGLeakBC,
	rawGLeakGO,
	rawGLeakGR,
	rawGLeakIO,
	rawGLeakNC,
	rawGLeakPC,
	rawGLeakSC,
	rawGMFAMPAIncNC,
	rawGMFNMDAIncNC,
	threshRestBC,
	threshRestGO,
	threshRestGR,
	threshRestIO,
	threshRestNC,
	threshRestPC,
	threshRestSC,
	threshDecTauBC,
	threshDecTauGO,
	threshDecTauUBC,
	threshDecTauGR,
	threshDecTauIO,
	threshDecTauNC,
	threshDecTauPC,
	threshDecTauSC,
	threshMaxBC,
	threshMaxGO,
	threshMaxGR,
	threshMaxIO,
	threshMaxNC,
	threshMaxPC,
	threshMaxSC
};

enum derived_ap_name_id
{
	numTSinMFHist,
	gLeakGO,
	gDecMFtoGO,
	gDecayMFtoGONMDA,
	gDecGRtoGO,
	gGABADecGOtoGO,
	goGABAGOGOSynRec,
	threshDecGO,
	gDirectDecMFtoGR,
	gSpilloverDecMFtoGR,
	gDirectDecGOtoGR, 
	gSpilloverDecGOtoGR,
	threshDecGR,
	tsPerHistBinGR, 
	gLeakSC,
	gDecGRtoSC,
	threshDecSC,
	gDecGRtoBC,
	gDecPCtoBC,
	threshDecBC,
	threshDecPC,
	gLeakPC,
	gDecGRtoPC,
	gDecBCtoPC,
	gDecSCtoPC,
	tsPopHistPC, /* used for updating MFNC syn plasticity */
	tsPerPopHistBinPC, /* used for updating MFNC syn plasticity */ 
	numPopHistBinsPC, /* used for updating MFNC syn plasticity */ 
	gLeakIO,
	threshDecIO,
	tsLTDDurationIO,
	tsLTDstartAPIO,
	tsLTPstartAPIO,
	tsLTPEndAPIO,  
	grPCHistCheckBinIO, /* used in PFPC syn plasticity */
	gmaxNMDADecMFtoNC,
	gmaxAMPADecMFtoNC,
	gNMDAIncMFtoNC,
	gAMPAIncMFtoNC,
	gDecPCtoNC,
	gLeakNC,
	threshDecNC,
	gLeakBC
};

extern std::string ap_names[NUM_AP_PARAMS];
extern float act_params[NUM_AP_PARAMS]; //TODO: change name to 'raw_ap'
extern float derived_act_params[NUM_DERIVED_AP_PARAMS];

void populate_act_params(parsed_build_file &p_file);
void populate_derived_act_params();
bool act_params_populated();
bool derived_act_params_populated();
void print_ap();
//void print_derived_ap();

#endif /* ACTIVITYPARAMS_H_ */

