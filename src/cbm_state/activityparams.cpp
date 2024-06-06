/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "activityparams.h"
#include "connectivityparams.h"
#include <math.h>

float coupleRiRjRatioGO = 0.0;
float coupleRiRjRatioIO = 0.05;
float eBCtoPC = -80.0;
float eGABAGO = -64.0;
float eGOGR = -75.0;
float eMFGR = 0.0;
float eMGluRGO = -96.0;
float eNCtoIO = -80.0;
float ePCtoBC = -70.0;
float ePCtoNC = -80.0;
float eSCtoPC = -80.0;
float gDecTauBCtoPC = 7.0;
float gIncBCtoPC = 0.0003;
float gGABADecTauGOtoGO = 10.0;
float gIncDirectGOtoGR = 0.01;
float gDirectTauGOtoGR = 7.0;
float gIncFracSpilloverGOtoGR = 1.0;
float gSpilloverTauGOtoGR = 150.0;
float gGABAIncGOtoGO = 0.01;
float gDecTauGRtoGO = 4.0;
float gIncGRtoGO = 0.0003;
float gDecTauMFtoGO = 3.0;
float gIncMFtoGO = 0.0009;
float gConstGO = 0.0;
float NMDA_AMPAratioMFGO = 1.3;
float gDecTauMFtoGONMDA = 30.0;
float gIncDirectMFtoGR = 0.0265; // float was = 0.0237; // float was = 0.0239;
float gDirectTauMFtoGR = 0.0;    // very crucial synapse!!!
float gIncFracSpilloverMFtoGR = 0.2;
float gSpilloverTauMFtoGR = 15.0;
float recoveryTauMF = 35.0;
float fracDepMF = 0.6;
float recoveryTauGO = 20.0;
float fracDepGO = 0.6;
float gIncMFtoUBC = 0.11;
float gIncGOtoUBC = 0.15;
float gIncUBCtoUBC = 0.15;
float gIncUBCtoGO = 0.04;
float gIncUBCtoGR = 0.12;
float gKIncUBC = 0.0135;
float gKTauUBC = 85.0;
float gConstUBC = 0.13;
float threshTauUBC = 80.0;
float gMGluRDecGRtoGO = 0.98;
float gMGluRIncDecayGO = 0.98;
float gMGluRIncScaleGO = 0.0;
float gMGluRScaleGRtoGO = 0.0;
float gDecT0ofNCtoIO = 0.56;
float gDecTSofNCtoIO = 0.5;
float gDecTTofNCtoIO = 70.0;
float gIncNCtoIO = 0.0015;
float gIncTauNCtoIO = 300.0;
float gDecTauPCtoBC = 5.0;
float gDecTauPCtoNC = 4.15;
float gIncAvgPCtoNC = 0.1;
float gDecTauGRtoBC = 2.0;
float gDecTauGRtoPC = 4.15;
float gDecTauGRtoSC = 2.0;
float gIncGRtoPC = 0.55e-05;
float gDecTauSCtoPC = 4.15;
float gIncSCtoPC = 0.0001;
float gluDecayGO = 0.98;
float gluScaleGO = 0.0;
float goGABAGOGOSynDepF = 1.0;
float goGABAGOGOSynRecTau = 100.0;

// experimental long term plasticity params
float fracSynWLow = 0.5;
float fracLowState = 0.5;
float cascPlastProbMin = 0.9;
float cascPlastProbMax = 0.1;
float cascPlastWeightLow = 0.25;
float cascPlastWeightHigh = 0.55;
float binPlastProbMin = 0.8;
float binPlastProbMax = 0.2;
float binPlastWeightLow = 0.3;
float binPlastWeightHigh = 0.7;
// experimental long term plasticity params

float synLTDStepSizeGRtoPC = -0.00275;
float synLTPStepSizeGRtoPC = 0.00030556;
float mGluRDecayGO = 0.98;
float mGluRScaleGO = 0.0;
float maxExtIncVIO = 30.0;
float gmaxAMPADecTauMFtoNC = 6.0;
float synLTDStepSizeMFtoNC = -6.25e-07;
float synLTDPCPopActThreshMFtoNC = 12.0;
float synLTPStepSizeMFtoNC = 5.0e-06;
float synLTPPCPopActThreshMFtoNC = 2.0;
float gmaxNMDADecTauMFtoNC = 50.0;
float msLTDDurationIO = 100.0;
float msLTDStartAPIO = -100.0;
float msLTPEndAPIO = -100.0;
float msLTPStartAPIO = 0.0;
float msPerHistBinGR = 5.0;
float msPerHistBinMF = 5.0;
float relPDecT0ofNCtoIO = 78.0;
float relPDecTSofNCtoIO = 40.0;
float relPDecTTofNCtoIO = 1.0;
float relPIncNCtoIO = 0.25;
float relPIncTauNCtoIO = 0.8;
float gIncPCtoBC = 0.15;
float gIncGRtoBC = 0.00225;
float gIncGRtoSC = 0.008;
float rawGLeakBC = 0.13;
float rawGLeakGO = 0.02;
float rawGLeakGR = 0.1;
float rawGLeakIO = 0.15;
float rawGLeakNC = 0.1;
float rawGLeakPC = 0.2;
float rawGLeakSC = 0.2;
float rawGMFAMPAIncNC = 2.35; // 3.0
float rawGMFNMDAIncNC = 2.35; // 3.0
float threshDecTauBC = 10.0;
float threshDecTauGO = 11.0;
float threshDecTauUBC = 20.0;
float threshDecTauGR = 3.0;
float threshDecTauIO = 200.0;
float threshDecTauNC = 5.0;
float threshDecTauPC = 6.0;
float threshDecTauSC = 18.0;
float threshMaxBC = 0.0;
float threshMaxGO = 10.0;
float threshMaxGR = -20.0;
float threshMaxIO = 10.0;
float threshMaxNC = -40.0;
float threshMaxPC = -48.0;
float threshMaxSC = 0.0;
float weightScale = 1.0;
float rawGRGOW = 0.0007;
float rawMFGOW = 0.0035;
float gogrW = 0.015;
float gogoW = 0.0125;

/* derived act params */
float numTSinMFHist = msPerHistBinMF / msPerTimeStep;
float gLeakGO = rawGLeakGO; // / (6 - msPerTimeStep);
float gDecMFtoGO = exp(-msPerTimeStep / gDecTauMFtoGO);
float gDecayMFtoGONMDA = exp(-msPerTimeStep / gDecTauMFtoGONMDA);
float gDecGRtoGO = exp(-msPerTimeStep / gDecTauGRtoGO);
float gGABADecGOtoGO = exp(-msPerTimeStep / gGABADecTauGOtoGO);
float goGABAGOGOSynRec = 1 - exp(-msPerTimeStep / goGABAGOGOSynRecTau);
float threshDecGO = 1 - exp(-msPerTimeStep / threshDecTauGO);
float gDirectDecMFtoGR = exp(-msPerTimeStep / gDirectTauMFtoGR);
float gSpilloverDecMFtoGR = exp(-msPerTimeStep / gSpilloverTauMFtoGR);
float gDirectDecGOtoGR = exp(-msPerTimeStep / gDirectTauGOtoGR);
float gSpilloverDecGOtoGR = exp(-msPerTimeStep / gSpilloverTauGOtoGR);
float threshDecGR = 1 - exp(-msPerTimeStep / threshDecTauGR);
float tsPerHistBinGR = msPerHistBinGR / msPerTimeStep;
float gLeakSC = rawGLeakSC / (6 - msPerTimeStep);
float gDecGRtoSC = exp(-msPerTimeStep / gDecTauGRtoSC);
float threshDecSC = 1 - exp(-msPerTimeStep / threshDecTauSC);
float gDecGRtoBC = exp(-msPerTimeStep / gDecTauGRtoBC);
float gDecPCtoBC = exp(-msPerTimeStep / gDecTauPCtoBC);
float threshDecBC = 1 - exp(-msPerTimeStep / threshDecTauBC);
float threshDecPC = 1 - exp(-msPerTimeStep / threshDecTauPC);
float gLeakPC = rawGLeakPC / (6 - msPerTimeStep);
float gDecGRtoPC = exp(-msPerTimeStep / gDecTauGRtoPC);
float gDecBCtoPC = exp(-msPerTimeStep / gDecTauBCtoPC);
float gDecSCtoPC = exp(-msPerTimeStep / gDecTauSCtoPC);
float tsPopHistPC = 40 / msPerTimeStep;
float tsPerPopHistBinPC = 5 / msPerTimeStep;
// numPopHistBinsPC    =  8.0; tsPopHistPC / tsPerPopHistBinPC
float gLeakIO = rawGLeakIO / (6 - msPerTimeStep);
float threshDecIO = 1 - exp(-msPerTimeStep / threshDecTauIO);
float tsLTDDurationIO = msLTDDurationIO / msPerTimeStep;
float tsLTDstartAPIO = msLTDStartAPIO / msPerTimeStep;
float tsLTPstartAPIO = msLTPStartAPIO / msPerTimeStep;
float tsLTPEndAPIO = msLTPEndAPIO / msPerTimeStep;
float grPCHistCheckBinIO = abs(msLTPEndAPIO / msPerHistBinGR);
float gmaxNMDADecMFtoNC = exp(-msPerTimeStep / gmaxNMDADecTauMFtoNC);
float gmaxAMPADecMFtoNC = exp(-msPerTimeStep / gmaxAMPADecTauMFtoNC);
float gNMDAIncMFtoNC =
    rawGMFNMDAIncNC; // 1 - exp(-msPerTimeStep /
                     // rawGMFNMDAIncNC); // modified 09/29/2022
float gAMPAIncMFtoNC =
    rawGMFAMPAIncNC; // 1 - exp(-msPerTimeStep /
                     // rawGMFAMPAIncNC); // modified 09/29/2022
float gDecPCtoNC = exp(-msPerTimeStep / gDecTauPCtoNC);
float gLeakNC = rawGLeakNC / (6 - msPerTimeStep);
float threshDecNC = 1 - exp(-msPerTimeStep / threshDecTauNC);
float gLeakBC = rawGLeakBC;
float grgoW = rawGRGOW * weightScale;
float mfgoW = rawMFGOW * weightScale;
