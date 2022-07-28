/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include <fstream>
#include <cstring>
#include <assert.h>
#include "fileIO/serialize.h"
#include "params/activityparams.h"

ActivityParams::ActivityParams() {}

ActivityParams::ActivityParams(parsed_file &p_file)
{
	for (auto iter = p_file.parsed_sections["activity"].param_map.begin();
			  iter != p_file.parsed_sections["activity"].param_map.end();
			  iter++)
	{
		paramMap[iter->first] = std::stof(iter->second.value);
	}
	updateParams();
}

// TODO: CHANGE THIS ALGORITHM IN THE FUTURE TO ONE IN CONPARAM.cpp
ActivityParams::ActivityParams(std::string actParamFile)
{
	//Assumes that file is in the following format:
	//key\tvalue\n
	//key\tvalue\n

//	//loop through file and add key/value pair to map
//	//** this is done to remove the necessity of order in the original file

	std::cout << "[INFO]: opening activity parameter file..." << std::endl;
	std::fstream paramFileBuffer(actParamFile.c_str());

	std::string key;
	float val;
	char temp;

	while(true)
	{
		temp = paramFileBuffer.peek();
		while (temp == ' ')
		{
			temp = paramFileBuffer.get();
			temp = paramFileBuffer.peek();
		}		
		if (temp == '#')
		{
			while (temp != '\n')
			{
				temp = paramFileBuffer.get();
			}
		}

		paramFileBuffer >> key >> val;

		if (key.compare("activityParamEnd") == 0)
		{
			break;
		}

		paramMap[key] = val;
	}
	std::cout << "[INFO]: activity parameter file opened, params loaded in param map..." << std::endl;
	updateParams();
	paramFileBuffer.close();
}

ActivityParams::ActivityParams(std::fstream &sim_file_buf)
{
	readParams(sim_file_buf);
	updateParams(); 
}

ActivityParams::ActivityParams(const ActivityParams &copyFrom) : paramMap(copyFrom.paramMap)
{
	updateParams();
}

ActivityParams::~ActivityParams() {}

void ActivityParams::readParams(std::fstream &inParamBuf)
{
	// TODO: need addtl checks on whether param maps are initialized or not
	if (paramMap.size() != 0)
	{
		paramMap.clear();
	}
	std::cout << "[INFO]: Reading activity params from file..." << std::endl;
	unserialize_map_from_file<std::string, float>(paramMap, inParamBuf);
	std::cout << "[INFO]: Finished reading activity params from file." << std::endl;
}

void ActivityParams::writeParams(std::fstream &outParamBuf)
{
	std::cout << "[INFO]: Writing activity params to file..." << std::endl;
	serialize_map_to_file<std::string, float>(paramMap, outParamBuf);
	std::cout << "[INFO]: Finished writing activity params to file..." << std::endl;
}

unsigned int ActivityParams::getMSPerTimeStep()
{
	return msPerTimeStep;
}

float ActivityParams::getParam(std::string paramName)
{
   	assert(paramMap.find(paramName) != paramMap.end());
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

// NOTE: I am pretty sure we define params in the header that are not in the input file
// for reasons that I simply cannot fathom. SO this guy wont return a *full* string 
// representation of the activity parameters. *deep sigh*
std::string ActivityParams::toString()
{
	std::string out_string = "[\n";
	for (auto iter = paramMap.begin(); iter != paramMap.end(); iter++)
	{
		out_string += "[ '" + iter->first + "', '"
							+ std::to_string(iter->second)
							+ "' ]\n";
	}
	out_string += "]";
	return out_string;
}

std::ostream &operator<<(std::ostream &os, ActivityParams &ap)
{
	return os << ap.toString();
}

ActivityParams &ActivityParams::operator=(const ActivityParams &copyFrom)
{
	if (this != &copyFrom)
	{
	   this->paramMap.clear();
	   this->paramMap = copyFrom.paramMap; /* stl supports assignment op overload of maps */
	   updateParams();
	}
	return *this;
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

