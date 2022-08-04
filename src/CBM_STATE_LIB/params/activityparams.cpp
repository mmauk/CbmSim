/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include <fstream>
#include <cstring>
#include <assert.h>
#include "fileIO/rawbytesrw.h"
#include "fileIO/serialize.h"
#include "params/activityparams.h"

// these are used to read the raw parameters from file
std::string ap_names[NUM_AP_PARAMS] = 
{
	"coupleRiRjRatioGO",
	"coupleRiRjRatioIO",
	"eBCtoPC",
	"eGABAGO",
	"eGOGR",
	"eLeakBC",
	"eLeakGO",
	"eLeakGR",
	"eLeakIO", 
	"eLeakNC",
	"eLeakPC",
	"eLeakSC",
	"eMFGR",
	"eMGluRGO",
	"eNCtoIO", 
	"ePCtoBC",
	"ePCtoNC",
	"eSCtoPC",
	"gDecTauBCtoPC",
	"gIncBCtoPC",
	"gGABADecTauGOtoGO",
	"gIncDirectGOtoGR",
	"gDirectTauGOtoGR",
	"gIncFracSpilloverGOtoGR",
	"gSpilloverTauGOtoGR",
	"gGABAIncGOtoGO",
	"gDecTauGRtoGO",
	"gIncGRtoGO",
	"gDecTauMFtoGO",
	"gIncMFtoGO",
	"gConstGO",
	"NMDA_AMPAratioMFGO",
	"gDecTauMFtoGONMDA",
	"gIncDirectMFtoGR",
	"gDirectTauMFtoGR",
	"gIncFracSpilloverMFtoGR",
	"gSpilloverTauMFtoGR",
	"recoveryTauMF",
	"fracDepMF",
	"recoveryTauGO",
	"fracDepGO",
	"gIncMFtoUBC",
	"gIncGOtoUBC",
	"gIncUBCtoUBC",
	"gIncUBCtoGO",
	"gIncUBCtoGR",
	"gKIncUBC",
	"gKTauUBC",
	"gConstUBC",
	"threshTauUBC",
	"gMGluRDecGRtoGO",
	"gMGluRIncDecayGO",
	"gMGluRIncScaleGO",
	"gMGluRScaleGRtoGO",
	"gDecT0ofNCtoIO",
	"gDecTSofNCtoIO",
	"gDecTTofNCtoIO",
	"gIncNCtoIO",
	"gIncTauNCtoIO",
	"gDecTauPCtoBC",
	"gDecTauPCtoNC",
	"gIncAvgPCtoNC",
	"gDecTauGRtoBC",
	"gDecTauGRtoPC",
	"gDecTauGRtoSC",
	"gIncGRtoPC",
	"gDecTauSCtoPC",
	"gIncSCtoPC",
	"gluDecayGO",
	"gluScaleGO",
	"goGABAGOGOSynDepF",
	"goGABAGOGOSynRecTau",
	"synLTDStepSizeGRtoPC",
	"synLTPStepSizeGRtoPC",
	"mGluRDecayGO",
	"mGluRScaleGO",
	"maxExtIncVIO",
	"gmaxAMPADecTauMFtoNC",
	"synLTDStepSizeMFtoNC",
	"synLTDPCPopActThreshMFtoNC",
	"synLTPStepSizeMFtoNC",
	"synLTPPCPopActThreshMFtoNC",
	"gmaxNMDADecTauMFtoNC",
	"initSynWofMFtoNC",
	"msLTDDurationIO",
	"msLTDStartAPIO",
	"msLTPEndAPIO",
	"msLTPStartAPIO",
	"msPerHistBinGR",
	"msPerHistBinMF",
	"msPerTimeStep",
	"relPDecT0ofNCtoIO",
	"relPDecTSofNCtoIO",
	"relPDecTTofNCtoIO",
	"relPIncNCtoIO",
	"relPIncTauNCtoIO",
	"gIncPCtoBC",
	"gIncGRtoBC",
	"gIncGRtoSC",
	"initSynWofGRtoPC",
	"rawGLeakBC",
	"rawGLeakGO",
	"rawGLeakGR",
	"rawGLeakIO",
	"rawGLeakNC",
	"rawGLeakPC",
	"rawGLeakSC",
	"rawGMFAMPAIncNC",
	"rawGMFNMDAIncNC",
	"threshRestBC",
	"threshRestGO",
	"threshRestGR",
	"threshRestIO",
	"threshRestNC",
	"threshRestPC",
	"threshRestSC",
	"threshDecTauBC",
	"threshDecTauGO",
	"threshDecTauUBC",
	"threshDecTauGR",
	"threshDecTauIO",
	"threshDecTauNC",
	"threshDecTauPC",
	"threshDecTauSC",
	"threshMaxBC",
	"threshMaxGO",
	"threshMaxGR",
	"threshMaxIO",
	"threshMaxNC",
	"threshMaxPC",
	"threshMaxSC"
};

float act_params[NUM_AP_PARAMS] = {0.0};
float derived_act_params[NUM_DERIVED_AP_PARAMS] = {0.0};

void populate_act_params(parsed_build_file &p_file)
{
	for (int i = 0; i < NUM_AP_PARAMS; i++)
	{
		act_params[i] = std::stof(p_file.parsed_sections["activity"].param_map[ap_names[i]].value);
	}
}

/* NOTE: should be called *after* populate_act_params */
void populate_derived_act_params()
{
	derived_act_params[numTSinMFHist]       = act_params[msPerHistBinMF] / act_params[msPerTimeStep];
	derived_act_params[gLeakGO]             = act_params[rawGLeakGO] / (6 - act_params[msPerTimeStep]);
	derived_act_params[gDecMFtoGO]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauMFtoGO]);
	derived_act_params[gDecayMFtoGONMDA]    = exp(-act_params[msPerTimeStep] / act_params[gDecTauMFtoGONMDA]);
	derived_act_params[gDecGRtoGO]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauGRtoGO]);
	derived_act_params[gGABADecGOtoGO]      = exp(-act_params[msPerTimeStep] / act_params[gGABADecTauGOtoGO]);
	derived_act_params[goGABAGOGOSynRec]    = 1 - exp(-act_params[msPerTimeStep] / act_params[goGABAGOGOSynRecTau]);
	derived_act_params[threshDecGO]         = 1 - exp(-act_params[msPerTimeStep] / act_params[threshDecTauGO]);
	derived_act_params[gDirectDecMFtoGR]    = exp(-act_params[msPerTimeStep] / act_params[gDirectTauMFtoGR]);
	derived_act_params[gSpilloverDecMFtoGR] = exp(-act_params[msPerTimeStep] / act_params[gSpilloverTauMFtoGR]);
	derived_act_params[gDirectDecGOtoGR]    = exp(-act_params[msPerTimeStep] / act_params[gDirectTauGOtoGR]);
	derived_act_params[gSpilloverDecGOtoGR] = exp(-act_params[msPerTimeStep] / act_params[gSpilloverTauGOtoGR]);
	derived_act_params[threshDecGR]         = 1 - exp(-act_params[msPerTimeStep] / act_params[threshDecTauGR]);
	derived_act_params[tsPerHistBinGR]      = act_params[msPerHistBinGR] / act_params[msPerTimeStep];
	derived_act_params[gLeakSC]             = act_params[rawGLeakSC] / (6 - act_params[msPerTimeStep]);
	derived_act_params[gDecGRtoSC]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauGRtoSC]);
	derived_act_params[threshDecSC]         = 1 - exp(-act_params[msPerTimeStep] / act_params[threshDecTauSC]);
	derived_act_params[gDecGRtoBC]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauGRtoBC]);
	derived_act_params[gDecPCtoBC]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauPCtoBC]);
	derived_act_params[threshDecBC]         = 1 - exp(-act_params[msPerTimeStep] / act_params[threshDecTauBC]);
	derived_act_params[threshDecPC]         = 1 - exp(-act_params[msPerTimeStep] / act_params[threshDecTauPC]);
	derived_act_params[gLeakPC]             = act_params[rawGLeakPC] / (6 - act_params[msPerTimeStep]);
	derived_act_params[gDecGRtoPC]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauGRtoPC]);
	derived_act_params[gDecBCtoPC]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauBCtoPC]);
	derived_act_params[gDecSCtoPC]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauSCtoPC]);
	derived_act_params[tsPopHistPC]         = 40 / act_params[msPerTimeStep];
	derived_act_params[tsPerPopHistBinPC]   =  5 / act_params[msPerTimeStep], 
	derived_act_params[numPopHistBinsPC]    =  8, /* tsPopHistPC / tsPerPopHistBinPC */
	derived_act_params[gLeakIO]             = act_params[rawGLeakIO] / (6 - act_params[msPerTimeStep]);
	derived_act_params[threshDecIO]         = 1 - exp(-act_params[msPerTimeStep] / act_params[threshDecTauIO]);
	derived_act_params[tsLTDDurationIO]     = act_params[msLTDDurationIO] / act_params[msPerTimeStep];
	derived_act_params[tsLTDstartAPIO]      = act_params[msLTDStartAPIO]  / act_params[msPerTimeStep];
	derived_act_params[tsLTPstartAPIO]      = act_params[msLTPStartAPIO]  / act_params[msPerTimeStep];
	derived_act_params[tsLTPEndAPIO]        = act_params[msLTPEndAPIO]    / act_params[msPerTimeStep];
	derived_act_params[grPCHistCheckBinIO]  = abs(act_params[msLTPEndAPIO] / act_params[msPerHistBinGR]);
	derived_act_params[gmaxNMDADecMFtoNC]   = exp(-act_params[msPerTimeStep] / act_params[gmaxNMDADecTauMFtoNC]);
	derived_act_params[gmaxAMPADecMFtoNC]   = exp(-act_params[msPerTimeStep] / act_params[gmaxAMPADecTauMFtoNC]);
	derived_act_params[gNMDAIncMFtoNC]      = 1 - exp(-act_params[msPerTimeStep] / act_params[rawGMFNMDAIncNC]);
	derived_act_params[gAMPAIncMFtoNC]      = 1 - exp(-act_params[msPerTimeStep] / act_params[rawGMFAMPAIncNC]);
	derived_act_params[gDecPCtoNC]          = exp(-act_params[msPerTimeStep] / act_params[gDecTauPCtoNC]);
	derived_act_params[gLeakNC]             = act_params[rawGLeakNC] / (6 - act_params[msPerTimeStep]);
	derived_act_params[threshDecNC]         = 1 - exp(-act_params[msPerTimeStep] / act_params[threshDecTauNC]);
	derived_act_params[gLeakBC]             = act_params[rawGLeakBC];
}

bool act_params_populated()
{
	for (int i = 0; i < NUM_AP_PARAMS; i++)
	{
		if (act_params[i] != 0.0) /* float equality :weird_champ: */
		{
			return true;
		}
	}
	return false;
}

bool derived_act_params_populated()
{
	for (int i = 0; i < NUM_DERIVED_AP_PARAMS; i++)
	{
		if (derived_act_params[i] != 0.0) /* float equality :weird_champ: */
		{
			return true;
		}
	}
	return false;
}

void print_ap()
{
	for (int i = 0; i < NUM_AP_PARAMS; i++)
	{
		std::cout << "[ '" << ap_names[i] << "', '" << act_params[i] << "']" << std::endl;
	}
}

//void print_derived_ap()
//{
//	for (int i = 0; i < NUM_DERIVED_AP_PARAMS; i++)
//	{
//		std::cout << "[ '" << derived_ap_names[i] << "', '" << derived_act_params[i] << "']" << std::endl;
//	}
//}

