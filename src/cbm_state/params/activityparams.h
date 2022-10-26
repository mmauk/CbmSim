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
#include <string>
#include <iostream>
#include <fstream>
#include "file_parse.h"
#include "stdint.h"

#define NUM_ACT_PARAMS 174

extern bool act_params_populated;

/* raw params */
extern float coupleRiRjRatioGO;
extern float coupleRiRjRatioIO;
extern float eBCtoPC;
extern float eGABAGO;
extern float eGOGR;
extern float eMFGR;
extern float eMGluRGO;
extern float eNCtoIO; 
extern float ePCtoBC;
extern float ePCtoNC;
extern float eSCtoPC;
extern float gDecTauBCtoPC;
extern float gIncBCtoPC;
extern float gGABADecTauGOtoGO;
extern float gIncDirectGOtoGR;
extern float gDirectTauGOtoGR;
extern float gIncFracSpilloverGOtoGR;
extern float gSpilloverTauGOtoGR;
extern float gGABAIncGOtoGO;
extern float gDecTauGRtoGO;
extern float gIncGRtoGO;
extern float gDecTauMFtoGO;
extern float gIncMFtoGO;
extern float gConstGO;
extern float NMDA_AMPAratioMFGO;
extern float gDecTauMFtoGONMDA;
extern float gIncDirectMFtoGR;
extern float gDirectTauMFtoGR;
extern float gIncFracSpilloverMFtoGR;
extern float gSpilloverTauMFtoGR;
extern float recoveryTauMF;
extern float fracDepMF;
extern float recoveryTauGO;
extern float fracDepGO;
extern float gIncMFtoUBC;
extern float gIncGOtoUBC;
extern float gIncUBCtoUBC;
extern float gIncUBCtoGO;
extern float gIncUBCtoGR;
extern float gKIncUBC;
extern float gKTauUBC;
extern float gConstUBC;
extern float threshTauUBC;
extern float gMGluRDecGRtoGO;
extern float gMGluRIncDecayGO;
extern float gMGluRIncScaleGO;
extern float gMGluRScaleGRtoGO;
extern float gDecT0ofNCtoIO;
extern float gDecTSofNCtoIO;
extern float gDecTTofNCtoIO;
extern float gIncNCtoIO;
extern float gIncTauNCtoIO;
extern float gDecTauPCtoBC;
extern float gDecTauPCtoNC;
extern float gIncAvgPCtoNC;
extern float gDecTauGRtoBC;
extern float gDecTauGRtoPC;
extern float gDecTauGRtoSC;
extern float gIncGRtoPC;
extern float gDecTauSCtoPC;
extern float gIncSCtoPC;
extern float gluDecayGO;
extern float gluScaleGO;
extern float goGABAGOGOSynDepF;
extern float goGABAGOGOSynRecTau;
extern float synLTDStepSizeGRtoPC;
extern float synLTPStepSizeGRtoPC;
extern float mGluRDecayGO;
extern float mGluRScaleGO;
extern float maxExtIncVIO;
extern float gmaxAMPADecTauMFtoNC;
extern float synLTDStepSizeMFtoNC;
extern float synLTDPCPopActThreshMFtoNC;
extern float synLTPStepSizeMFtoNC;
extern float synLTPPCPopActThreshMFtoNC;
extern float gmaxNMDADecTauMFtoNC;
extern float msLTDDurationIO;
extern float msLTDStartAPIO;
extern float msLTPEndAPIO;
extern float msLTPStartAPIO;
extern float msPerHistBinGR;
extern float msPerHistBinMF;
extern float relPDecT0ofNCtoIO;
extern float relPDecTSofNCtoIO;
extern float relPDecTTofNCtoIO;
extern float relPIncNCtoIO;
extern float relPIncTauNCtoIO;
extern float gIncPCtoBC;
extern float gIncGRtoBC;
extern float gIncGRtoSC;
extern float rawGLeakBC;
extern float rawGLeakGO;
extern float rawGLeakGR;
extern float rawGLeakIO;
extern float rawGLeakNC;
extern float rawGLeakPC;
extern float rawGLeakSC;
extern float rawGMFAMPAIncNC;
extern float rawGMFNMDAIncNC;
extern float threshDecTauBC;
extern float threshDecTauGO;
extern float threshDecTauUBC;
extern float threshDecTauGR;
extern float threshDecTauIO;
extern float threshDecTauNC;
extern float threshDecTauPC;
extern float threshDecTauSC;
extern float threshMaxBC;
extern float threshMaxGO;
extern float threshMaxGR;
extern float threshMaxIO;
extern float threshMaxNC;
extern float threshMaxPC;
extern float threshMaxSC;
extern float weightScale;
extern float rawGRGOW;
extern float rawMFGOW;
extern float gogrW;
extern float gogoW;

/* derived act params */
extern float numTSinMFHist;
extern float gLeakGO;
extern float gDecMFtoGO;
extern float gDecayMFtoGONMDA;
extern float gDecGRtoGO;
extern float gGABADecGOtoGO;
extern float goGABAGOGOSynRec;
extern float threshDecGO;
extern float gDirectDecMFtoGR;
extern float gSpilloverDecMFtoGR;
extern float gDirectDecGOtoGR; 
extern float gSpilloverDecGOtoGR;
extern float threshDecGR;
extern float tsPerHistBinGR;
extern float gLeakSC;
extern float gDecGRtoSC;
extern float threshDecSC;
extern float gDecGRtoBC;
extern float gDecPCtoBC;
extern float threshDecBC;
extern float threshDecPC;
extern float gLeakPC;
extern float gDecGRtoPC;
extern float gDecBCtoPC;
extern float gDecSCtoPC;
extern float tsPopHistPC; /* used for updating MFNC syn plasticity */
extern float tsPerPopHistBinPC; /* used for updating MFNC syn plasticity */ 
// extern float numPopHistBinsPC; /* used for updating MFNC syn plasticity */ 
extern float gLeakIO;
extern float threshDecIO;
extern float tsLTDDurationIO;
extern float tsLTDstartAPIO;
extern float tsLTPstartAPIO;
extern float tsLTPEndAPIO; 
extern float grPCHistCheckBinIO; /* used in PFPC syn plasticity */
extern float gmaxNMDADecMFtoNC;
extern float gmaxAMPADecMFtoNC;
extern float gNMDAIncMFtoNC;
extern float gAMPAIncMFtoNC;
extern float gDecPCtoNC;
extern float gLeakNC;
extern float threshDecNC;
extern float gLeakBC;
extern float grgoW;
extern float mfgoW;

void populate_act_params(parsed_sess_file &s_file);
void read_act_params(std::fstream &in_param_buf);
void write_act_params(std::fstream &out_param_buf);

#endif /* ACTIVITYPARAMS_H_ */

