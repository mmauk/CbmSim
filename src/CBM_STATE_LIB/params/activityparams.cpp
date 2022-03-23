/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "params/activityparams.h"

ActivityParams::ActivityParams(std::fstream &infile)
{
	//Assumes that file is in the following format:
	//key\tvalue\n
	//key\tvalue\n

//	//loop through file and add key/value pair to map
//	//** this is done to remove the necessity of order in the original file

	std::string key;
	float val;
	char temp;

	while(true)
	{
		temp = infile.peek();
		while (temp == ' ')
		{
			temp = infile.get();
			temp = infile.peek();
		}		
		if (temp == '#')
		{
			while (temp != '\n')
			{
				temp = infile.get();
			}
		}

		infile >> key >> val;

		if (key.compare("activityParamEnd") == 0)
		{
			break;
		}

		paramMap[key] = val;
	}

	updateParams();
}

void ActivityParams::writeParams(std::fstream &outfile)
{
	for (auto i = paramMap.begin(); i != paramMap.end(); i++)
	{
		outfile << i->first << " " << i->second << std::endl;
	}

	outfile << "activityParamEnd 1" << std::endl;
}

unsigned int ActivityParams::getMSPerTimeStep()
{
	return msPerTimeStep;
}

void ActivityParams::showParams(std::ostream &outSt)
{
	outSt << "msPerTimeStep " << msPerTimeStep << std::endl << std::endl;

	outSt << "msPerHistBinMF " << msPerHistBinMF << std::endl;
	outSt << "tsPerHistbinMF " << numTSinMFHist << std::endl << std::endl;

	outSt << "coupleRiRjRatioGO " << coupleRiRjRatioGO << std::endl;
	outSt << "eLeakGO " << eLeakGO << std::endl;
	outSt << "eMGluRGO " << eMGluRGO << std::endl;
	outSt << "eGABAGO " << eGABAGO << std::endl;
	outSt  <<  "threshMaxGO " << threshMaxGO << std::endl;
	outSt  <<  "threshRestGO " << threshRestGO << std::endl;
	outSt  <<  "gIncMFtoGO " << gIncMFtoGO << std::endl;
	outSt  <<  "gIncGRtoGO " << gIncGRtoGO << std::endl;
	outSt  <<  "gGABAIncGOtoGO " << gGABAIncGOtoGO << std::endl;
	outSt  <<  "gMGluRScaleGRtoGO " << gMGluRScaleGRtoGO << std::endl;
	outSt  <<  "mGluRScaleGO " << mGluRScaleGO << std::endl;
	outSt  <<  "gluScaleGO " << gluScaleGO << std::endl;
	outSt  <<  "gLeakGO " << gLeakGO << std::endl;
	outSt  <<  "gDecTauMFtoGO " << gDecTauMFtoGO << std::endl;
	outSt  <<  "gDecMFtoGO " << gDecMFtoGO << std::endl;
	outSt  <<  "gDecTauGRtoGO " << gDecTauGRtoGO << std::endl;
	outSt  <<  "gDecGRtoGO " << gDecGRtoGO << std::endl;
	outSt  <<  "gGABADecTauGOtoGO" << gGABADecTauGOtoGO << std::endl;
	outSt  <<  "gGABADecGOtoGO " << gGABADecGOtoGO << std::endl;
	outSt  <<  "mGluRDecayGO " << mGluRDecayGO << std::endl;
	outSt  <<  "gMGluRIncDecayGO " << gMGluRIncDecayGO << std::endl;
	outSt  <<  "gluDecayGO " << gluDecayGO << std::endl;
	outSt  <<  "threshDecTauGO " << threshDecTauGO << std::endl;
	outSt  <<  "threshDecGO " << threshDecGO << std::endl << std::endl;

	outSt  <<  "eLeakGR " << eLeakGR << std::endl;
	outSt  <<  "eGOGR " << eGOGR << std::endl;
	outSt  <<  "eMFGR " << eMFGR << std::endl;
	outSt  <<  "threshMaxGR " << threshMaxGR << std::endl;
	outSt  <<  "threshRestGR " << threshRestGR << std::endl;
	outSt << "threshDecTauGR " << threshDecTauGR << std::endl;
	outSt << "threshDecGR " << threshDecGR << std::endl;
	outSt << "gLeakGR " << gLeakGR << std::endl;
	outSt << "msPerHistBinGR " << msPerHistBinGR << std::endl;
	outSt << "tsPerHistBinGR " << tsPerHistBinGR << std::endl << std::endl;

	outSt << "eLeakSC " << eLeakSC << std::endl;
	outSt << "gLeakSC " << gLeakSC << std::endl;
	outSt << "gDecTauGRtoSC " << gDecTauGRtoSC << std::endl;
	outSt << "gDecGRtoSC " << gDecGRtoSC << std::endl;
	outSt << "threshMaxSC " << threshMaxSC << std::endl;
	outSt << "threshRestSC " << threshRestSC << std::endl;
	outSt << "threshDecTauSC " << threshDecTauSC << std::endl;
	outSt << "threshDecSC " << threshDecSC << std::endl;
	outSt << "gIncGRtoSC " << gIncGRtoSC << std::endl << std::endl;

	outSt << "eLeakBC " << eLeakBC << std::endl;
	outSt << "ePCtoBC " << ePCtoBC << std::endl;
	outSt << "gLeakBC " << gLeakBC << std::endl;
	outSt << "gDecTauGRtoBC " << gDecTauGRtoBC << std::endl;
	outSt << "gDecGRtoBC " << gDecGRtoBC << std::endl;
	outSt << "gDecTauPCtoBC " << gDecTauPCtoBC << std::endl;
	outSt << "gDecPCtoBC " << gDecPCtoBC << std::endl;
	outSt << "threshRestBC " << threshRestBC << std::endl;
	outSt << "threshMaxBC " << threshMaxBC << std::endl;
	outSt << "threshDecTauBC " << threshDecTauBC << std::endl;
	outSt << "threshDecBC " << threshDecBC << std::endl;
	outSt << "gIncGRtoBC " << gIncGRtoBC << std::endl;
	outSt << "gIncPCtoBC " << gIncPCtoBC << std::endl << std::endl;

	outSt << "initSynWofGRtoPC " << initSynWofGRtoPC << std::endl;
	outSt << "eLeakPC " << eLeakPC << std::endl;
	outSt << "eBCtoPC " << eBCtoPC << std::endl;
	outSt << "eSCtoPC " << eSCtoPC << std::endl;
	outSt << "threshMaxPC " << threshMaxPC << std::endl;
	outSt << "threshRestPC " << threshRestPC << std::endl;
	outSt << "threshDecTauPC " << threshDecTauPC << std::endl;
	outSt << "threshDecPC " << threshDecPC << std::endl;
	outSt << "gLeakPC " << gLeakPC << std::endl;
	outSt << "gDecTauGRtoPC " << gDecTauGRtoPC << std::endl;
	outSt << "gDecGRtoPC " << gDecGRtoPC << std::endl;
	outSt << "gDecTauBCtoPC " << gDecTauBCtoPC << std::endl;
	outSt << "gDecBCtoPC " << gDecBCtoPC << std::endl;
	outSt << "gDecTauSCtoPC " << gDecTauSCtoPC << std::endl;
	outSt << "gDecSCtoPC " << gDecSCtoPC << std::endl;
	outSt << "gIncSCtoPC " << gIncSCtoPC << std::endl;
	outSt << "gIncGRtoPC " << gIncGRtoPC << std::endl;
	outSt << "gIncBCtoPC " << gIncBCtoPC << std::endl;
	outSt << "tsPopHistPC " << tsPopHistPC << std::endl;
	outSt << "tsPerPopHistBinPC " << tsPerPopHistBinPC << std::endl;
	outSt << "numPopHistBinsPC " << numPopHistBinsPC << std::endl << std::endl;

	outSt << "coupleRiRjRatioIO " << coupleRiRjRatioIO << std::endl;
	outSt << "eLeakIO " << eLeakIO << std::endl;
	outSt << "eNCtoIO " << eNCtoIO << std::endl;
	outSt << "gLeakIO " << gLeakIO << std::endl;
	outSt << "gDecTSofNCtoIO " << gDecTSofNCtoIO << std::endl;
	outSt << "gDecTTofNCtoIO " << gDecTTofNCtoIO << std::endl;
	outSt << "gDecT0ofNCtoIO " << gDecT0ofNCtoIO << std::endl;
	outSt << "gIncNCtoIO " << gIncNCtoIO << std::endl;
	outSt << "gIncTauNCtoIO " << gIncTauNCtoIO << std::endl;
	outSt << "threshRestIO " << threshRestIO << std::endl;
	outSt << "threshMaxIO " << threshMaxIO << std::endl;
	outSt << "threshDecTauIO " << threshDecTauIO << std::endl;
	outSt << "threshDecIO " << threshDecIO << std::endl;
	outSt << "tsLTDDurationIO " << tsLTDDurationIO << std::endl;
	outSt << "tsLTDStartAPIO " << tsLTDStartAPIO << std::endl;
	outSt << "tsLTPStartAPIO " << tsLTPStartAPIO << std::endl;
	outSt << "tsLTPEndAPIO " << tsLTPEndAPIO << std::endl;
	outSt << "synLTPStepSizeGRtoPC " << synLTPStepSizeGRtoPC << std::endl;
	outSt << "synLTDStepSizeGRtoPC " << synLTDStepSizeGRtoPC << std::endl;
	outSt << "grPCHistCheckBinIO " << grPCHistCheckBinIO << std::endl;
	outSt << "maxExtIncVIO " << maxExtIncVIO << std::endl << std::endl;

	outSt << "eLeakNC " << eLeakNC << std::endl;
	outSt << "ePCtoNC " << ePCtoNC << std::endl;
	outSt << "gmaxNMDADecTauMFtoNC " << gmaxNMDADecTauMFtoNC << std::endl;
	outSt << "gmaxNMDADecMFtoNC " << gmaxNMDADecMFtoNC << std::endl;
	outSt << "gmaxAMPADecTauMFtoNC " << gmaxAMPADecTauMFtoNC << std::endl;
	outSt << "gmaxAMPADecMFtoNC " << gmaxAMPADecMFtoNC << std::endl;
	outSt << "gNMDAIncMFtoNC " << gNMDAIncMFtoNC << std::endl;
	outSt << "gAMPAIncMFtoNC " << gAMPAIncMFtoNC << std::endl;
	outSt << "gIncAvgPCtoNC " << gIncAvgPCtoNC << std::endl;
	outSt << "gDecTauPCtoNC " << gDecTauPCtoNC << std::endl;
	outSt << "gDecPCtoNC " << gDecPCtoNC << std::endl;
	outSt << "gLeakNC " << gLeakNC << std::endl;
	outSt << "threshDecTauNC " << threshDecTauNC << std::endl;
	outSt << "threshDecNC " << threshDecNC << std::endl;
	outSt << "threshMaxNC " << threshMaxNC << std::endl;
	outSt << "threshRestNC " << threshRestNC << std::endl;
	outSt << "relPDecTSofNCtoIO " << relPDecTSofNCtoIO << std::endl;
	outSt << "relPDecTTofNCtoIO " << relPDecTTofNCtoIO << std::endl;
	outSt << "relPDecT0ofNCtoIO " << relPDecT0ofNCtoIO << std::endl;
	outSt << "relPIncNCtoIO " << relPIncNCtoIO << std::endl;
	outSt << "relPIncTauNCtoIO " << relPIncTauNCtoIO << std::endl;
	outSt << "initSynWofMFtoNC " << initSynWofMFtoNC << std::endl;
	outSt << "synLTDPCPopActThreshMFtoNC " << synLTDPCPopActThreshMFtoNC << std::endl;
	outSt << "synLTPPCPopActThreshMFtoNC " << synLTPPCPopActThreshMFtoNC << std::endl;
	outSt << "synLTDStepSizeMFtoNC " << synLTDStepSizeMFtoNC << std::endl;
	outSt << "synLTPStepSizeMFtoNC " << synLTPStepSizeMFtoNC << std::endl;
}

std::map<std::string, float> ActivityParams::getParamCopy()
{
	std::map<std::string, float> paramCopy;

	for (auto i = paramMap.begin(); i!=paramMap.end(); i++)
	{
		paramCopy[i->first] = i->second;
	}

	return paramCopy;
}

float ActivityParams::getParam(std::string paramName)
{
	return paramMap[paramName];
}

bool ActivityParams::setParam(std::string paramName, float value)
{
	if (paramMap.find(paramName) == paramMap.end())
	{
		return false;
	}
	paramMap[paramName] = value;
	updateParams();

	return true;
}

void ActivityParams::updateParams()
{
	int flag = paramMap["parameterVersion"];
	switch(flag){
		case 0:
			updateParamsOriginal();
			break;
		case 1:
			updateParamsV1();
			break;
	}
}

//updateParamsOriginal() --- parameterVersion == 0/NULL
//---> original parameter names
//---> no original comments (has the ability to though)
//---> no parameterVersion parameter

void ActivityParams::updateParamsOriginal()
{
	msPerTimeStep = paramMap["msPerTimeStep"];

	msPerHistBinMF = paramMap["msPerHistBinMF"];
	numTSinMFHist = msPerHistBinMF/msPerTimeStep;

	//move elements from map to public variables
	if (paramMap.find("coupleRiRjRatioGO") == paramMap.end())
	{
		paramMap["coupleRiRjRatioGO"] = 0;
	}

	if (paramMap.find("goGABAGOGOSynRecTau") == paramMap.end())
	{
		paramMap["goGABAGOGOSynRecTau"] = 1;
	}

	if (paramMap.find("goGABAGOGOSynDepF") == paramMap.end())
	{
		paramMap["goGABAGOGOSynDepF"] = 1;
	}

	eLeakGO = paramMap["eLeakGO"];
	eMGluRGO = paramMap["eMGluRGO"];
	eGABAGO = paramMap["eGABAGO"];
	threshMaxGO = paramMap["threshMaxGO"];
	threshRestGO = paramMap["threshBaseGO"];
	gIncMFtoGO = paramMap["gMFIncGO"];
	gIncGRtoGO = paramMap["gGRIncGO"];
	gGABAIncGOtoGO = paramMap["gGOIncGO"];
	coupleRiRjRatioGO = paramMap["coupleRiRjRatioGO"];

	gMGluRScaleGRtoGO = paramMap["gMGluRScaleGO"];
	gMGluRIncScaleGO = paramMap["gMGluRIncScaleGO"];
	mGluRScaleGO = paramMap["mGluRScaleGO"];
	gluScaleGO = paramMap["gluScaleGO"];
	gLeakGO = paramMap["rawGLeakGO"]/(6-msPerTimeStep);

	gDecTauMFtoGO = paramMap["gMFDecayTGO"];
	gDecMFtoGO = exp(-msPerTimeStep/gDecTauMFtoGO);
	NMDA_AMPAratioMFGO = paramMap["NMDA_AMPAratioMFGO"];
	gDecTauMFtoGONMDA = paramMap["gDecTauMFtoGONMDA"];
	gDecayMFtoGONMDA = exp(-msPerTimeStep/gDecTauMFtoGONMDA);

	gDecTauGRtoGO = paramMap["gGRDecayTGO"];
	gDecGRtoGO = exp(-msPerTimeStep/gDecTauGRtoGO);

	gGABADecTauGOtoGO = paramMap["gGODecayTGO"];
	gGABADecGOtoGO = exp(-msPerTimeStep/gGABADecTauGOtoGO);

	//synaptic depression test for GOGABAGO
	goGABAGOGOSynRecTau = paramMap["goGABAGOGOSynRecTau"];
	goGABAGOGOSynRec = 1 - exp(-msPerTimeStep / goGABAGOGOSynRecTau);
	goGABAGOGOSynDepF = paramMap["goGABAGOGOSynDepF"];

	mGluRDecayGO = paramMap["mGluRDecayGO"];
	gMGluRIncDecayGO = paramMap["gMGluRIncDecayGO"];
	gMGluRDecGRtoGO = paramMap["gMGluRDecayGO"];
	gluDecayGO = paramMap["gluDecayGO"];

	threshDecTauGO = paramMap["threshDecayTGO"];
	threshDecGO = 1 - exp(-msPerTimeStep/threshDecTauGO);

	eLeakGR = paramMap["eLeakGR"];
	eGOGR = paramMap["eGOGR"];
	eMFGR = paramMap["eMFGR"];
	threshMaxGR = paramMap["threshMaxGR"];
	threshRestGR = paramMap["threshBaseGR"];

	threshDecTauGR = paramMap["threshDecayTGR"];
	threshDecGR = 1 - exp(-msPerTimeStep / threshDecTauGR);

	gLeakGR = paramMap["rawGLeakGR"] / (6 - msPerTimeStep);

	msPerHistBinGR = paramMap["msPerHistBinGR"];
	tsPerHistBinGR = msPerHistBinGR/msPerTimeStep;

	eLeakSC = paramMap["eLeakSC"];
	gLeakSC = paramMap["rawGLeakSC"] / (6 - msPerTimeStep);
	gDecTauGRtoSC = paramMap["gPFDecayTSC"];
	gDecGRtoSC = exp(-msPerTimeStep / gDecTauGRtoSC);
	threshMaxSC = paramMap["threshMaxSC"];
	threshRestSC = paramMap["threshBaseSC"];
	threshDecTauSC = paramMap["threshDecayTSC"];
	threshDecSC = 1 - exp(-msPerTimeStep / threshDecTauSC);
	gIncGRtoSC = paramMap["pfIncSC"];

	//**From mzone**
	eLeakBC = paramMap["eLeakBC"];
	ePCtoBC = paramMap["ePCBC"];
	gLeakBC = paramMap["rawGLeakBC"];

	gDecTauGRtoBC = paramMap["gPFDecayTBC"];
	gDecGRtoBC = exp(-msPerTimeStep / gDecTauGRtoBC);

	gDecTauPCtoBC = paramMap["gPCDecayTBC"];
	gDecPCtoBC = exp(-msPerTimeStep / gDecTauPCtoBC);

	threshDecTauBC = paramMap["threshDecayTBC"];
	threshDecBC = 1 - exp(-msPerTimeStep / threshDecTauBC);

	threshRestBC = paramMap["threshBaseBC"];
	threshMaxBC = paramMap["threshMaxBC"];
	gIncGRtoBC = paramMap["pfIncConstBC"];
	gIncPCtoBC = paramMap["pcIncConstBC"];

	initSynWofGRtoPC = paramMap["pfSynWInitPC"];
	eLeakPC = paramMap["eLeakPC"];
	eBCtoPC = paramMap["eBCPC"];
	eSCtoPC = paramMap["eSCPC"];
	threshMaxPC = paramMap["threshMaxPC"];
	threshRestPC = paramMap["threshBasePC"];

	threshDecTauPC = paramMap["threshDecayTPC"];
	threshDecPC = 1 - exp(-msPerTimeStep / threshDecTauPC);

	gLeakPC = paramMap["rawGLeakPC"] / (6 - msPerTimeStep);

	gDecTauGRtoPC = paramMap["gPFDecayTPC"];
	gDecGRtoPC = exp(-msPerTimeStep / gDecTauGRtoPC);

	gDecTauBCtoPC = paramMap["gBCDecayTPC"];
	gDecBCtoPC = exp(-msPerTimeStep / gDecTauBCtoPC);

	gDecTauSCtoPC = paramMap["gSCDecayTPC"];
	gDecSCtoPC = exp(-msPerTimeStep / gDecTauSCtoPC);

	gIncSCtoPC = paramMap["gSCIncConstPC"];
	gIncGRtoPC = paramMap["gPFScaleConstPC"];
	gIncBCtoPC = paramMap["gBCScaleConstPC"];

	tsPopHistPC = 40 / msPerTimeStep;  
	tsPerPopHistBinPC = 5 / msPerTimeStep;
	numPopHistBinsPC = tsPopHistPC / tsPerPopHistBinPC;

	coupleRiRjRatioIO = paramMap["coupleScaleIO"];
	eLeakIO = paramMap["eLeakIO"];
	eNCtoIO = paramMap["eNCIO"];
	gLeakIO = paramMap["rawGLeakIO"] / (6 - msPerTimeStep);
	gDecTSofNCtoIO = paramMap["gNCDecTSIO"];
	gDecTTofNCtoIO = paramMap["gNCDecTTIO"];
	gDecT0ofNCtoIO = paramMap["gNCDecT0IO"];
	gIncNCtoIO = paramMap["gNCIncScaleIO"];
	gIncTauNCtoIO = paramMap["gNCIncTIO"];
	threshRestIO = paramMap["threshBaseIO"];
	threshMaxIO = paramMap["threshMaxIO"];

	threshDecTauIO = paramMap["threshDecayTIO"];
	threshDecIO = 1 - exp(-msPerTimeStep / threshDecTauIO);

	tsLTDDurationIO = paramMap["msLTDDurationIO"] / msPerTimeStep;
	tsLTDStartAPIO = paramMap["msLTDStartAPIO"] / msPerTimeStep;
	tsLTPStartAPIO = paramMap["msLTPStartAPIO"] / msPerTimeStep;
	tsLTPEndAPIO = paramMap["msLTPEndAPIO"] / msPerTimeStep;
	synLTPStepSizeGRtoPC = paramMap["grPCLTPIncIO"];
	synLTDStepSizeGRtoPC = paramMap["grPCLTDDecIO"];
	grPCHistCheckBinIO = abs(tsLTPEndAPIO / ((int)tsPerHistBinGR));

	maxExtIncVIO = paramMap["maxErrDriveIO"];

	eLeakNC = paramMap["eLeakNC"];
	ePCtoNC = paramMap["ePCNC"];

	gmaxNMDADecTauMFtoNC = paramMap["mfNMDADecayTNC"];
	gmaxNMDADecMFtoNC = exp(-msPerTimeStep / gmaxNMDADecTauMFtoNC);

	gmaxAMPADecTauMFtoNC = paramMap["mfAMPADecayTNC"];
	gmaxAMPADecMFtoNC = exp(-msPerTimeStep / gmaxAMPADecTauMFtoNC);

	gNMDAIncMFtoNC = 1 - exp(-msPerTimeStep / paramMap["rawGMFNMDAIncNC"]);
	gAMPAIncMFtoNC = 1 - exp(-msPerTimeStep / paramMap["rawGMFAMPAIncNC"]);
	gIncAvgPCtoNC = paramMap["gPCScaleAvgNC"];

	gDecTauPCtoNC = paramMap["gPCDecayTNC"];
	gDecPCtoNC = exp(-msPerTimeStep / gDecTauPCtoNC);

	gLeakNC = paramMap["rawGLeakNC"] / (6 - msPerTimeStep);

	threshDecTauNC = paramMap["threshDecayTNC"];
	threshDecNC = 1 - exp(-msPerTimeStep / threshDecTauNC);

	threshMaxNC = paramMap["threshMaxNC"];
	threshRestNC = paramMap["threshBaseNC"];
	relPDecTSofNCtoIO = paramMap["outIORelPDecTSNC"];
	relPDecTTofNCtoIO = paramMap["outIORelPDecTTNC"];
	relPDecT0ofNCtoIO = paramMap["outIORelPDecT0NC"];
	relPIncNCtoIO = paramMap["outIORelPIncScaleNC"];
	relPIncTauNCtoIO = paramMap["outIORelPIncTNC"];
	initSynWofMFtoNC = paramMap["mfSynWInitNC"];
	synLTDPCPopActThreshMFtoNC = paramMap["mfNCLTDThreshNC"];
	synLTPPCPopActThreshMFtoNC = paramMap["mfNCLTPThreshNC"];
	synLTDStepSizeMFtoNC = paramMap["mfNCLTDDecNC"];
	synLTPStepSizeMFtoNC = paramMap["mfNCLTPIncNC"];
}


 // updateParamsV1() --- parameterVersion  ==  1
 // ---> updated parameter names to match variables
 // ---> includes ability to add comments
 // ---> includes parameterVersion flag 

void ActivityParams::updateParamsV1()
{
	msPerTimeStep = paramMap["msPerTimeStep"];

	msPerHistBinMF = paramMap["msPerHistBinMF"];
	numTSinMFHist = msPerHistBinMF / msPerTimeStep;

	// move elements from map to public variables
 	// 	paramMap.
	if (paramMap.find("coupleRiRjRatioGO") == paramMap.end())
	{
		paramMap["coupleRiRjRatioGO"] = 0;
	}

	if (paramMap.find("goGABAGOGOSynRecTau") == paramMap.end())
	{
		paramMap["goGABAGOGOSynRecTau"] = 1;
	}

	if (paramMap.find("goGABAGOGOSynDepF") == paramMap.end())
	{
		paramMap["goGABAGOGOSynDepF"] = 1;
	}

	gIncMFtoUBC = paramMap["gIncMFtoUBC"];
	gIncGOtoUBC = paramMap["gIncGOtoUBC"];	
	gIncUBCtoUBC = paramMap["gIncUBCtoUBC"];
	gIncUBCtoGO = paramMap["gIncUBCtoGO"];
	gIncUBCtoGR = paramMap["gIncUBCtoGR"];

	gKIncUBC = paramMap["gKIncUBC"];
	gKTauUBC = paramMap["gKTauUBC"];	
	gConstUBC = paramMap["gConstUBC"];
	threshTauUBC = paramMap["threshTauUBC"];

	threshDecTauUBC = paramMap["threshDecayTauUBC"];
	threshDecUBC = 1 - exp(-msPerTimeStep / threshDecTauUBC);

	eLeakGO = paramMap["eLeakGO"];
	eMGluRGO = paramMap["eMGluRGO"];
	eGABAGO = paramMap["eGABAGO"];
	threshMaxGO = paramMap["threshMaxGO"];
	threshRestGO = paramMap["threshRestGO"];
	gIncMFtoGO = paramMap["gIncMFtoGO"];
	gIncGRtoGO = paramMap["gIncGRtoGO"];
	gGABAIncGOtoGO = paramMap["gGABAIncGOtoGO"];
	coupleRiRjRatioGO = paramMap["coupleRiRjRatioGO"];

	gMGluRScaleGRtoGO = paramMap["gMGluRScaleGRtoGO"];
	gMGluRIncScaleGO = paramMap["gMGluRIncScaleGO"];
	mGluRScaleGO = paramMap["mGluRScaleGO"];
	gluScaleGO = paramMap["gluScaleGO"];
	gLeakGO = paramMap["rawGLeakGO"] / (6 - msPerTimeStep);

	gDecTauMFtoGO = paramMap["gDecTauMFtoGO"];
	gDecMFtoGO = exp(-msPerTimeStep / gDecTauMFtoGO);
	NMDA_AMPAratioMFGO = paramMap["NMDA_AMPAratioMFGO"];
	gDecTauMFtoGONMDA = paramMap["gDecTauMFtoGONMDA"];
	gDecayMFtoGONMDA = exp(-msPerTimeStep / gDecTauMFtoGONMDA);

	gDecTauGRtoGO = paramMap["gDecTauGRtoGO"];
	gDecGRtoGO = exp(-msPerTimeStep / gDecTauGRtoGO);

	gGABADecTauGOtoGO = paramMap["gGABADecTauGOtoGO"];
	gGABADecGOtoGO = exp(-msPerTimeStep / gGABADecTauGOtoGO);

	// synaptic depression test for GOGABAGO
	goGABAGOGOSynRecTau = paramMap["goGABAGOGOSynRecTau"];
	goGABAGOGOSynRec = 1 - exp(-msPerTimeStep / goGABAGOGOSynRecTau);
	goGABAGOGOSynDepF = paramMap["goGABAGOGOSynDepF"];

	mGluRDecayGO = paramMap["mGluRDecayGO"];
	gMGluRIncDecayGO = paramMap["gMGluRIncDecayGO"];
	gMGluRDecGRtoGO = paramMap["gMGluRDecGRtoGO"];
	gluDecayGO = paramMap["gluDecayGO"];

	gConstGO = paramMap["gConstGO"];
	threshDecTauGO = paramMap["threshDecTauGO"];
	threshDecGO = 1 - exp(-msPerTimeStep / threshDecTauGO);

	eLeakGR = paramMap["eLeakGR"];
	eGOGR = paramMap["eGOGR"];
	eMFGR = paramMap["eMFGR"];
	threshMaxGR = paramMap["threshMaxGR"];
	threshRestGR = paramMap["threshRestGR"];

	gIncDirectMFtoGR = paramMap["gIncDirectMFtoGR"];
	gDirectTauMFtoGR = paramMap["gDecTauMFtoGR"];
	gDirectDecMFtoGR = exp(-msPerTimeStep / gDirectTauMFtoGR);
	gIncFracSpilloverMFtoGR = paramMap["gIncFracSpilloverMFtoGR"];
	gSpilloverTauMFtoGR = paramMap["gSpilloverTauMFtoGR"];
	gSpilloverDecMFtoGR = exp(-msPerTimeStep / gSpilloverTauMFtoGR);
	recoveryTauMF = paramMap["recoveryTauMF"];
	fracDepMF = paramMap["fracDepMF"];
	
	gIncDirectGOtoGR = paramMap["gIncDirectGOtoGR"];
	gDirectTauGOtoGR = paramMap["gDirectTauGOtoGR"];
	gDirectDecGOtoGR = exp(-msPerTimeStep / gDirectTauGOtoGR);
	gIncFracSpilloverGOtoGR = paramMap["gIncFracSpilloverGOtoGR"];
	gSpilloverTauGOtoGR = paramMap["gSpilloverTauGOtoGR"];
	gSpilloverDecGOtoGR = exp(-msPerTimeStep / gSpilloverTauGOtoGR);

	recoveryTauGO = paramMap["recoveryTauGO"];
	fracDepGO = paramMap["fracDepGO"];
	
	threshDecTauGR = paramMap["threshDecTauGR"];
	threshDecGR = 1 - exp(-msPerTimeStep / threshDecTauGR);

	gLeakGR = paramMap["rawGLeakGR"] / (6 - msPerTimeStep);

	msPerHistBinGR = paramMap["msPerHistBinGR"];
	tsPerHistBinGR = msPerHistBinGR / msPerTimeStep;

	eLeakSC = paramMap["eLeakSC"];
	gLeakSC = paramMap["rawGLeakSC"] / (6 - msPerTimeStep);
	gDecTauGRtoSC = paramMap["gDecTauGRtoSC"];
	gDecGRtoSC = exp(-msPerTimeStep / gDecTauGRtoSC);
	threshMaxSC = paramMap["threshMaxSC"];
	threshRestSC = paramMap["threshRestSC"];
	threshDecTauSC = paramMap["threshDecTauSC"];
	threshDecSC = 1 - exp(-msPerTimeStep / threshDecTauSC);
	gIncGRtoSC = paramMap["gIncGRtoSC"];

	// **From mzone**
	eLeakBC = paramMap["eLeakBC"];
	ePCtoBC = paramMap["ePCtoBC"];
	gLeakBC = paramMap["rawGLeakBC"];

	gDecTauGRtoBC = paramMap["gDecTauGRtoBC"];
	gDecGRtoBC = exp(-msPerTimeStep / gDecTauGRtoBC);

	gDecTauPCtoBC = paramMap["gDecTauPCtoBC"];
	gDecPCtoBC = exp(-msPerTimeStep / gDecTauPCtoBC);

	threshDecTauBC = paramMap["threshDecTauBC"];
	threshDecBC = 1 - exp(-msPerTimeStep / threshDecTauBC);

	threshRestBC = paramMap["threshRestBC"];
	threshMaxBC = paramMap["threshMaxBC"];
	gIncGRtoBC = paramMap["gIncGRtoBC"];
	gIncPCtoBC = paramMap["gIncPCtoBC"];

	initSynWofGRtoPC = paramMap["initSynWofGRtoPC"];
	eLeakPC = paramMap["eLeakPC"];
	eBCtoPC = paramMap["eBCtoPC"];
	eSCtoPC = paramMap["eSCtoPC"];
	threshMaxPC = paramMap["threshMaxPC"];
	threshRestPC = paramMap["threshRestPC"];

	threshDecTauPC = paramMap["threshDecTauPC"];
	threshDecPC = 1 - exp(-msPerTimeStep / threshDecTauPC);

	gLeakPC = paramMap["rawGLeakPC"] / (6 - msPerTimeStep);

	gDecTauGRtoPC = paramMap["gDecTauGRtoPC"];
	gDecGRtoPC = exp(-msPerTimeStep / gDecTauGRtoPC);

	gDecTauBCtoPC = paramMap["gDecTauBCtoPC"];
	gDecBCtoPC = exp(-msPerTimeStep / gDecTauBCtoPC);

	gDecTauSCtoPC = paramMap["gDecTauSCtoPC"];
	gDecSCtoPC = exp(-msPerTimeStep / gDecTauSCtoPC);

	gIncSCtoPC = paramMap["gIncSCtoPC"];
	gIncGRtoPC = paramMap["gIncGRtoPC"];
	gIncBCtoPC = paramMap["gIncBCtoPC"];

	tsPopHistPC = 40 / msPerTimeStep;  
	tsPerPopHistBinPC = 5 / msPerTimeStep;
	numPopHistBinsPC = tsPopHistPC / tsPerPopHistBinPC;

	coupleRiRjRatioIO = paramMap["coupleRiRjRatioIO"];
	eLeakIO = paramMap["eLeakIO"];
	eNCtoIO = paramMap["eNCtoIO"];
	gLeakIO = paramMap["rawGLeakIO"] / (6 - msPerTimeStep);
	gDecTSofNCtoIO = paramMap["gDecTSofNCtoIO"];
	gDecTTofNCtoIO = paramMap["gDecTTofNCtoIO"];
	gDecT0ofNCtoIO = paramMap["gDecT0ofNCtoIO"];
	gIncNCtoIO = paramMap["gIncNCtoIO"];
	gIncTauNCtoIO = paramMap["gIncTauNCtoIO"];
	threshRestIO = paramMap["threshRestIO"];
	threshMaxIO = paramMap["threshMaxIO"];

	threshDecTauIO = paramMap["threshDecTauIO"];
	threshDecIO = 1 - exp(-msPerTimeStep / threshDecTauIO);

	tsLTDDurationIO = paramMap["msLTDDurationIO"] / msPerTimeStep;
	tsLTDStartAPIO = paramMap["msLTDStartAPIO"] / msPerTimeStep;
	tsLTPStartAPIO = paramMap["msLTPStartAPIO"] / msPerTimeStep;
	tsLTPEndAPIO = paramMap["msLTPEndAPIO"] / msPerTimeStep;
	synLTPStepSizeGRtoPC = paramMap["synLTPStepSizeGRtoPC"];
	synLTDStepSizeGRtoPC = paramMap["synLTDStepSizeGRtoPC"];
	grPCHistCheckBinIO = abs(tsLTPEndAPIO / ((int)tsPerHistBinGR));

	maxExtIncVIO = paramMap["maxExtIncVIO"];

	eLeakNC = paramMap["eLeakNC"];
	ePCtoNC = paramMap["ePCtoNC"];

	gmaxNMDADecTauMFtoNC = paramMap["gmaxNMDADecTauMFtoNC"];
	gmaxNMDADecMFtoNC = exp(-msPerTimeStep / gmaxNMDADecTauMFtoNC);

	gmaxAMPADecTauMFtoNC = paramMap["gmaxAMPADecTauMFtoNC"];
	gmaxAMPADecMFtoNC = exp(-msPerTimeStep / gmaxAMPADecTauMFtoNC);

	gNMDAIncMFtoNC = 2.35; // 0.2835; // 1 - exp(-msPerTimeStep / paramMap["rawGMFNMDAIncNC"]);
	gAMPAIncMFtoNC = 2.35; // 1 - exp(-msPerTimeStep / paramMap["rawGMFAMPAIncNC"]);
	gIncAvgPCtoNC = paramMap["gIncAvgPCtoNC"];

	gDecTauPCtoNC = paramMap["gDecTauPCtoNC"];
	gDecPCtoNC = exp(-msPerTimeStep / gDecTauPCtoNC);

	gLeakNC = paramMap["rawGLeakNC"] / (6 - msPerTimeStep);

	threshDecTauNC = paramMap["threshDecTauNC"];
	threshDecNC = 1-exp(-msPerTimeStep / threshDecTauNC);

	threshMaxNC = paramMap["threshMaxNC"];
	threshRestNC = paramMap["threshRestNC"];
	relPDecTSofNCtoIO = paramMap["relPDecTSofNCtoIO"];
	relPDecTTofNCtoIO = paramMap["relPDecTTofNCtoIO"];
	relPDecT0ofNCtoIO = paramMap["relPDecT0ofNCtoIO"];
	relPIncNCtoIO = paramMap["relPIncNCtoIO"];
	relPIncTauNCtoIO = paramMap["relPIncTauNCtoIO"];
	initSynWofMFtoNC = paramMap["initSynWofMFtoNC"];
	synLTDPCPopActThreshMFtoNC = paramMap["synLTDPCPopActThreshMFtoNC"];
	synLTPPCPopActThreshMFtoNC = paramMap["synLTPPCPopActThreshMFtoNC"];
	synLTDStepSizeMFtoNC = paramMap["synLTDStepSizeMFtoNC"];
	synLTPStepSizeMFtoNC = paramMap["synLTPStepSizeMFtoNC"];
}

