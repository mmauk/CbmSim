/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "params/activityparams.h"

ActivityParams::ActivityParams(fstream &infile)
{
	//Assumes that file is in the following format:
	//key\tvalue\n
	//key\tvalue\n
//	map<string,float> tempMap;

//	string line;
//	//loop through file and add key/value pair to map
//	//** this is done to remove the necessity of order in the original file
//	while(getline(infile,line))
//	{
//		tempMap[line.substr(0,line.find_first_of("\t"))]=atof(line.substr(line.find_first_of("\t"),line.size()).c_str());
//	}

	string key;
	float val;
	char temp;

 
	while(true)
	//for(int i=0; i<140; i++)
	{
		temp = infile.peek();
		while(temp == ' '){
			temp = infile.get();
			temp = infile.peek();
		}		
		if(temp == '#'){
			while(temp != '\n'){
				temp = infile.get();
			}
		}

		infile>>key>>val;

		if(key.compare("activityParamEnd")==0)
		{
			break;
		}

		paramMap[key]=val;
	}

	updateParams();
//	msPerTimeStep=paramMap["msPerTimeStep"];
//
//	msPerHistBinMF=paramMap["msPerHistBinMF"];
//	numTSinMFHist=msPerHistBinMF/msPerTimeStep;
//
//	//move elements from map to public variables
////	paramMap.
//	if(paramMap.find("coupleRiRjRatioGO")==paramMap.end())
//	{
//		paramMap["coupleRiRjRatioGO"]=0;
//	}
//
//	if(paramMap.find("goGABAGOGOSynRecTau")==paramMap.end())
//	{
//		paramMap["goGABAGOGOSynRecTau"]=1;
//	}
//
//	if(paramMap.find("goGABAGOGOSynDepF")==paramMap.end())
//	{
//		paramMap["goGABAGOGOSynDepF"]=1;
//	}
//
//	eLeakGO=paramMap["eLeakGO"];
//	eMGluRGO=paramMap["eMGluRGO"];
//	eGABAGO=paramMap["eGABAGO"];
//	threshMaxGO=paramMap["threshMaxGO"];
//	threshRestGO=paramMap["threshBaseGO"];
//	gIncMFtoGO=paramMap["gMFIncGO"];
//	gIncGRtoGO=paramMap["gGRIncGO"];
//	gGABAIncGOtoGO=paramMap["gGOIncGO"];
//	coupleRiRjRatioGO=paramMap["coupleRiRjRatioGO"];
//
//	gMGluRScaleGRtoGO=paramMap["gMGluRScaleGO"];
//	gMGluRIncScaleGO=paramMap["gMGluRIncScaleGO"];
//	mGluRScaleGO=paramMap["mGluRScaleGO"];
//	gluScaleGO=paramMap["gluScaleGO"];
//	gLeakGO=paramMap["rawGLeakGO"]/(6-msPerTimeStep);
//
//	gDecTauMFtoGO=paramMap["gMFDecayTGO"];
//	gDecMFtoGO=exp(-msPerTimeStep/gDecTauMFtoGO);
//
//	gDecTauGRtoGO=paramMap["gGRDecayTGO"];
//	gDecGRtoGO=exp(-msPerTimeStep/gDecTauGRtoGO);
//
//	gGABADecTauGOtoGO=paramMap["gGODecayTGO"];
//	gGABADecGOtoGO=exp(-msPerTimeStep/gGABADecTauGOtoGO);
//
//	//synaptic depression test for GOGABAGO
//	goGABAGOGOSynRecTau=paramMap["goGABAGOGOSynRecTau"];
//	goGABAGOGOSynRec=1-exp(-msPerTimeStep/goGABAGOGOSynRecTau);
//	goGABAGOGOSynDepF=paramMap["goGABAGOGOSynDepF"];
//
//	mGluRDecayGO=paramMap["mGluRDecayGO"];
//	gMGluRIncDecayGO=paramMap["gMGluRIncDecayGO"];
//	gMGluRDecGRtoGO=paramMap["gMGluRDecayGO"];
//	gluDecayGO=paramMap["gluDecayGO"];
//
//	threshDecTauGO=paramMap["threshDecayTGO"];
//	threshDecGO=1-exp(-msPerTimeStep/threshDecTauGO);
//
//
//	eLeakGR=paramMap["eLeakGR"];
//	eGOGR=paramMap["eGOGR"];
//	eMFGR=paramMap["eMFGR"];
//	threshMaxGR=paramMap["threshMaxGR"];
//	threshRestGR=paramMap["threshBaseGR"];
//	gIncMFtoGR=paramMap["gMFIncGR"];
//	gIncGOtoGR=paramMap["gGOIncGR"];
//
//	gDecTauMFtoGR=paramMap["gMFDecayTGR"];
//	gDecMFtoGR=exp(-msPerTimeStep/gDecTauMFtoGR);
//
//	gDecTauGOtoGR=paramMap["gGODecayTGR"];
//	gDecGOtoGR=exp(-msPerTimeStep/gDecTauGOtoGR);
//
//	threshDecTauGR=paramMap["threshDecayTGR"];
//	threshDecGR=1-exp(-msPerTimeStep/threshDecTauGR);
//
//	gLeakGR=paramMap["rawGLeakGR"]/(6-msPerTimeStep);
//
//	msPerHistBinGR=paramMap["msPerHistBinGR"];
//	tsPerHistBinGR=msPerHistBinGR/msPerTimeStep;
//
//	eLeakSC=paramMap["eLeakSC"];
//	gLeakSC=paramMap["rawGLeakSC"]/(6-msPerTimeStep);
//	gDecTauGRtoSC=paramMap["gPFDecayTSC"];
//	gDecGRtoSC=exp(-msPerTimeStep/gDecTauGRtoSC);
//	threshMaxSC=paramMap["threshMaxSC"];
//	threshRestSC=paramMap["threshBaseSC"];
//	threshDecTauSC=paramMap["threshDecayTSC"];
//	threshDecSC=1-exp(-msPerTimeStep/threshDecTauSC);
//	gIncGRtoSC=paramMap["pfIncSC"];
//
//	//**From mzone**
//	eLeakBC=paramMap["eLeakBC"];
//	ePCtoBC=paramMap["ePCBC"];
//	gLeakBC=paramMap["rawGLeakBC"]/(6-msPerTimeStep);
//
//	gDecTauGRtoBC=paramMap["gPFDecayTBC"];
//	gDecGRtoBC=exp(-msPerTimeStep/gDecTauGRtoBC);
//
//	gDecTauPCtoBC=paramMap["gPCDecayTBC"];
//	gDecPCtoBC=exp(-msPerTimeStep/gDecTauPCtoBC);
//
//	threshDecTauBC=paramMap["threshDecayTBC"];
//	threshDecBC=1-exp(-msPerTimeStep/threshDecTauBC);
//
//	threshRestBC=paramMap["threshBaseBC"];
//	threshMaxBC=paramMap["threshMaxBC"];
//	gIncGRtoBC=paramMap["pfIncConstBC"];
//	gIncPCtoBC=paramMap["pcIncConstBC"];
//
//	initSynWofGRtoPC=paramMap["pfSynWInitPC"];
//	eLeakPC=paramMap["eLeakPC"];
//	eBCtoPC=paramMap["eBCPC"];
//	eSCtoPC=paramMap["eSCPC"];
//	threshMaxPC=paramMap["threshMaxPC"];
//	threshRestPC=paramMap["threshBasePC"];
//
//	threshDecTauPC=paramMap["threshDecayTPC"];
//	threshDecPC=1-exp(-msPerTimeStep/threshDecTauPC);
//
//	gLeakPC=paramMap["rawGLeakPC"]/(6-msPerTimeStep);
//
//	gDecTauGRtoPC=paramMap["gPFDecayTPC"];
//	gDecGRtoPC=exp(-msPerTimeStep/gDecTauGRtoPC);
//
//	gDecTauBCtoPC=paramMap["gBCDecayTPC"];
//	gDecBCtoPC=exp(-msPerTimeStep/gDecTauBCtoPC);
//
//	gDecTauSCtoPC=paramMap["gSCDecayTPC"];
//	gDecSCtoPC=exp(-msPerTimeStep/gDecTauSCtoPC);
//
//	gIncSCtoPC=paramMap["gSCIncConstPC"];
//	gIncGRtoPC=paramMap["gPFScaleConstPC"];
//	gIncBCtoPC=paramMap["gBCScaleConstPC"];
//
//	tsPopHistPC=40/msPerTimeStep; //TODO: fixed for now
//	tsPerPopHistBinPC=5/msPerTimeStep;
//	numPopHistBinsPC=tsPopHistPC/tsPerPopHistBinPC;
//
//	coupleRiRjRatioIO=paramMap["coupleScaleIO"];
//	eLeakIO=paramMap["eLeakIO"];
//	eNCtoIO=paramMap["eNCIO"];
//	gLeakIO=paramMap["rawGLeakIO"]/(6-msPerTimeStep);
//	gDecTSofNCtoIO=paramMap["gNCDecTSIO"];
//	gDecTTofNCtoIO=paramMap["gNCDecTTIO"];
//	gDecT0ofNCtoIO=paramMap["gNCDecT0IO"];
//	gIncNCtoIO=paramMap["gNCIncScaleIO"];
//	gIncTauNCtoIO=paramMap["gNCIncTIO"];
//	threshRestIO=paramMap["threshBaseIO"];
//	threshMaxIO=paramMap["threshMaxIO"];
//
//	threshDecTauIO=paramMap["threshDecayTIO"];
//	threshDecIO=1-exp(-msPerTimeStep/threshDecTauIO);
//
//	tsLTDDurationIO=paramMap["msLTDDurationIO"]/msPerTimeStep;
//	tsLTDStartAPIO=paramMap["msLTDStartAPIO"]/msPerTimeStep;
//	tsLTPStartAPIO=paramMap["msLTPStartAPIO"]/msPerTimeStep;
//	synLTPStepSizeGRtoPC=paramMap["grPCLTPIncIO"];
//	synLTDStepSizeGRtoPC=paramMap["grPCLTDDecIO"];
//	grPCHistCheckBinIO=abs(tsLTDStartAPIO/((int)tsPerHistBinGR));
//
//	maxExtIncVIO=paramMap["maxErrDriveIO"];
//
//	eLeakNC=paramMap["eLeakNC"];
//	ePCtoNC=paramMap["ePCNC"];
//
//	gmaxNMDADecTauMFtoNC=paramMap["mfNMDADecayTNC"];
//	gmaxNMDADecMFtoNC=exp(-msPerTimeStep/gmaxNMDADecTauMFtoNC);
//
//	gmaxAMPADecTauMFtoNC=paramMap["mfAMPADecayTNC"];
//	gmaxAMPADecMFtoNC=exp(-msPerTimeStep/gmaxAMPADecTauMFtoNC);
//
//	gNMDAIncMFtoNC=1-exp(-msPerTimeStep/paramMap["rawGMFNMDAIncNC"]);
//	gAMPAIncMFtoNC=1-exp(-msPerTimeStep/paramMap["rawGMFAMPAIncNC"]);
//	gIncAvgPCtoNC=paramMap["gPCScaleAvgNC"];
//
//	gDecTauPCtoNC=paramMap["gPCDecayTNC"];
//	gDecPCtoNC=exp(-msPerTimeStep/gDecTauPCtoNC);
//
//	gLeakNC=paramMap["rawGLeakNC"]/(6-msPerTimeStep);
//
//	threshDecTauNC=paramMap["threshDecayTNC"];
//	threshDecNC=1-exp(-msPerTimeStep/threshDecTauNC);
//
//	threshMaxNC=paramMap["threshMaxNC"];
//	threshRestNC=paramMap["threshBaseNC"];
//	relPDecTSofNCtoIO=paramMap["outIORelPDecTSNC"];
//	relPDecTTofNCtoIO=paramMap["outIORelPDecTTNC"];
//	relPDecT0ofNCtoIO=paramMap["outIORelPDecT0NC"];
//	relPIncNCtoIO=paramMap["outIORelPIncScaleNC"];
//	relPIncTauNCtoIO=paramMap["outIORelPIncTNC"];
//	initSynWofMFtoNC=paramMap["mfSynWInitNC"];
//	synLTDPCPopActThreshMFtoNC=paramMap["mfNCLTDThreshNC"];
//	synLTPPCPopActThreshMFtoNC=paramMap["mfNCLTPThreshNC"];
//	synLTDStepSizeMFtoNC=paramMap["mfNCLTDDecNC"];
//	synLTPStepSizeMFtoNC=paramMap["mfNCLTPIncNC"];
}

void ActivityParams::writeParams(fstream &outfile)
{
	map<string, float>::iterator i;

	for(i=paramMap.begin(); i!=paramMap.end(); i++)
	{
		outfile<<i->first<<" "<<i->second<<endl;
	}

	outfile<<"activityParamEnd 1"<<endl;
}

unsigned int ActivityParams::getMSPerTimeStep()
{
	return msPerTimeStep;
}

void ActivityParams::showParams(ostream &outSt)
{
	outSt<<"msPerTimeStep "<<msPerTimeStep<<endl<<endl;

	outSt<<"msPerHistBinMF "<<msPerHistBinMF<<endl;
	outSt<<"tsPerHistbinMF "<<numTSinMFHist<<endl<<endl;

	outSt<<"coupleRiRjRatioGO "<<coupleRiRjRatioGO<<endl;
	outSt<<"eLeakGO "<<eLeakGO<<endl;
	outSt<<"eMGluRGO "<<eMGluRGO<<endl;
	outSt<<"eGABAGO "<<eGABAGO<<endl;
	outSt<<"threshMaxGO "<<threshMaxGO<<endl;
	outSt<<"threshRestGO "<<threshRestGO<<endl;
	outSt<<"gIncMFtoGO "<<gIncMFtoGO<<endl;
	outSt<<"gIncGRtoGO "<<gIncGRtoGO<<endl;
	outSt<<"gGABAIncGOtoGO "<<gGABAIncGOtoGO<<endl;
	outSt<<"gMGluRScaleGRtoGO "<<gMGluRScaleGRtoGO<<endl;
	outSt<<"mGluRScaleGO "<<mGluRScaleGO<<endl;
	outSt<<"gluScaleGO "<<gluScaleGO<<endl;
	outSt<<"gLeakGO "<<gLeakGO<<endl;
	outSt<<"gDecTauMFtoGO "<<gDecTauMFtoGO<<endl;
	outSt<<"gDecMFtoGO "<<gDecMFtoGO<<endl;
	outSt<<"gDecTauGRtoGO "<<gDecTauGRtoGO<<endl;
	outSt<<"gDecGRtoGO "<<gDecGRtoGO<<endl;
	outSt<<"gGABADecTauGOtoGO"<<gGABADecTauGOtoGO<<endl;
	outSt<<"gGABADecGOtoGO "<<gGABADecGOtoGO<<endl;
	outSt<<"mGluRDecayGO "<<mGluRDecayGO<<endl;
	outSt<<"gMGluRIncDecayGO "<<gMGluRIncDecayGO<<endl;
	outSt<<"gluDecayGO "<<gluDecayGO<<endl;
	outSt<<"threshDecTauGO "<<threshDecTauGO<<endl;
	outSt<<"threshDecGO "<<threshDecGO<<endl<<endl;

	outSt<<"eLeakGR "<<eLeakGR<<endl;
	outSt<<"eGOGR "<<eGOGR<<endl;
	outSt<<"eMFGR "<<eMFGR<<endl;
	outSt<<"threshMaxGR "<<threshMaxGR<<endl;
	outSt<<"threshRestGR "<<threshRestGR<<endl;
	//outSt<<"gIncMFtoGR "<<gIncMFtoGR<<endl;
	//outSt<<"gIncGOtoGR "<<gIncGOtoGR<<endl;
	//outSt<<"gDecTauMFtoGR "<<gDecTauMFtoGR<<endl;
	//outSt<<"gDecMFtoGR "<<gDecMFtoGR<<endl;
	//outSt<<"gDecTauGOtoGR"<<gDecTauGOtoGR<<endl;
	//outSt<<"gDecGOtoGR "<<gDecGOtoGR<<endl;
	outSt<<"threshDecTauGR "<<threshDecTauGR<<endl;
	outSt<<"threshDecGR "<<threshDecGR<<endl;
	outSt<<"gLeakGR "<<gLeakGR<<endl;
	outSt<<"msPerHistBinGR "<<msPerHistBinGR<<endl;
	outSt<<"tsPerHistBinGR "<<tsPerHistBinGR<<endl<<endl;

	outSt<<"eLeakSC "<<eLeakSC<<endl;
	outSt<<"gLeakSC "<<gLeakSC<<endl;
	outSt<<"gDecTauGRtoSC "<<gDecTauGRtoSC<<endl;
	outSt<<"gDecGRtoSC "<<gDecGRtoSC<<endl;
	outSt<<"threshMaxSC "<<threshMaxSC<<endl;
	outSt<<"threshRestSC "<<threshRestSC<<endl;
	outSt<<"threshDecTauSC "<<threshDecTauSC<<endl;
	outSt<<"threshDecSC "<<threshDecSC<<endl;
	outSt<<"gIncGRtoSC "<<gIncGRtoSC<<endl<<endl;

	outSt<<"eLeakBC "<<eLeakBC<<endl;
	outSt<<"ePCtoBC "<<ePCtoBC<<endl;
	outSt<<"gLeakBC "<<gLeakBC<<endl;
	outSt<<"gDecTauGRtoBC "<<gDecTauGRtoBC<<endl;
	outSt<<"gDecGRtoBC "<<gDecGRtoBC<<endl;
	outSt<<"gDecTauPCtoBC "<<gDecTauPCtoBC<<endl;
	outSt<<"gDecPCtoBC "<<gDecPCtoBC<<endl;
	outSt<<"threshRestBC "<<threshRestBC<<endl;
	outSt<<"threshMaxBC "<<threshMaxBC<<endl;
	outSt<<"threshDecTauBC "<<threshDecTauBC<<endl;
	outSt<<"threshDecBC "<<threshDecBC<<endl;
	outSt<<"gIncGRtoBC "<<gIncGRtoBC<<endl;
	outSt<<"gIncPCtoBC "<<gIncPCtoBC<<endl<<endl;

	outSt<<"initSynWofGRtoPC "<<initSynWofGRtoPC<<endl;
	outSt<<"eLeakPC "<<eLeakPC<<endl;
	outSt<<"eBCtoPC "<<eBCtoPC<<endl;
	outSt<<"eSCtoPC "<<eSCtoPC<<endl;
	outSt<<"threshMaxPC "<<threshMaxPC<<endl;
	outSt<<"threshRestPC "<<threshRestPC<<endl;
	outSt<<"threshDecTauPC "<<threshDecTauPC<<endl;
	outSt<<"threshDecPC "<<threshDecPC<<endl;
	outSt<<"gLeakPC "<<gLeakPC<<endl;
	outSt<<"gDecTauGRtoPC "<<gDecTauGRtoPC<<endl;
	outSt<<"gDecGRtoPC "<<gDecGRtoPC<<endl;
	outSt<<"gDecTauBCtoPC "<<gDecTauBCtoPC<<endl;
	outSt<<"gDecBCtoPC "<<gDecBCtoPC<<endl;
	outSt<<"gDecTauSCtoPC "<<gDecTauSCtoPC<<endl;
	outSt<<"gDecSCtoPC "<<gDecSCtoPC<<endl;
	outSt<<"gIncSCtoPC "<<gIncSCtoPC<<endl;
	outSt<<"gIncGRtoPC "<<gIncGRtoPC<<endl;
	outSt<<"gIncBCtoPC "<<gIncBCtoPC<<endl;
	outSt<<"tsPopHistPC "<<tsPopHistPC<<endl;
	outSt<<"tsPerPopHistBinPC "<<tsPerPopHistBinPC<<endl;
	outSt<<"numPopHistBinsPC "<<numPopHistBinsPC<<endl<<endl;

	outSt<<"coupleRiRjRatioIO "<<coupleRiRjRatioIO<<endl;
	outSt<<"eLeakIO "<<eLeakIO<<endl;
	outSt<<"eNCtoIO "<<eNCtoIO<<endl;
	outSt<<"gLeakIO "<<gLeakIO<<endl;
	outSt<<"gDecTSofNCtoIO "<<gDecTSofNCtoIO<<endl;
	outSt<<"gDecTTofNCtoIO "<<gDecTTofNCtoIO<<endl;
	outSt<<"gDecT0ofNCtoIO "<<gDecT0ofNCtoIO<<endl;
	outSt<<"gIncNCtoIO "<<gIncNCtoIO<<endl;
	outSt<<"gIncTauNCtoIO "<<gIncTauNCtoIO<<endl;
	outSt<<"threshRestIO "<<threshRestIO<<endl;
	outSt<<"threshMaxIO "<<threshMaxIO<<endl;
	outSt<<"threshDecTauIO "<<threshDecTauIO<<endl;
	outSt<<"threshDecIO "<<threshDecIO<<endl;
	outSt<<"tsLTDDurationIO "<<tsLTDDurationIO<<endl;
	outSt<<"tsLTDStartAPIO "<<tsLTDStartAPIO<<endl;
	outSt<<"tsLTPStartAPIO "<<tsLTPStartAPIO<<endl;
	outSt<<"tsLTPEndAPIO "<<tsLTPEndAPIO<<endl;
	outSt<<"synLTPStepSizeGRtoPC "<<synLTPStepSizeGRtoPC<<endl;
	outSt<<"synLTDStepSizeGRtoPC "<<synLTDStepSizeGRtoPC<<endl;
	outSt<<"grPCHistCheckBinIO "<<grPCHistCheckBinIO<<endl;
	outSt<<"maxExtIncVIO "<<maxExtIncVIO<<endl<<endl;

	outSt<<"eLeakNC "<<eLeakNC<<endl;
	outSt<<"ePCtoNC "<<ePCtoNC<<endl;
	outSt<<"gmaxNMDADecTauMFtoNC "<<gmaxNMDADecTauMFtoNC<<endl;
	outSt<<"gmaxNMDADecMFtoNC "<<gmaxNMDADecMFtoNC<<endl;
	outSt<<"gmaxAMPADecTauMFtoNC "<<gmaxAMPADecTauMFtoNC<<endl;
	outSt<<"gmaxAMPADecMFtoNC "<<gmaxAMPADecMFtoNC<<endl;
	outSt<<"gNMDAIncMFtoNC "<<gNMDAIncMFtoNC<<endl;
	outSt<<"gAMPAIncMFtoNC "<<gAMPAIncMFtoNC<<endl;
	outSt<<"gIncAvgPCtoNC "<<gIncAvgPCtoNC<<endl;
	outSt<<"gDecTauPCtoNC "<<gDecTauPCtoNC<<endl;
	outSt<<"gDecPCtoNC "<<gDecPCtoNC<<endl;
	outSt<<"gLeakNC "<<gLeakNC<<endl;
	outSt<<"threshDecTauNC "<<threshDecTauNC<<endl;
	outSt<<"threshDecNC "<<threshDecNC<<endl;
	outSt<<"threshMaxNC "<<threshMaxNC<<endl;
	outSt<<"threshRestNC "<<threshRestNC<<endl;
	outSt<<"relPDecTSofNCtoIO "<<relPDecTSofNCtoIO<<endl;
	outSt<<"relPDecTTofNCtoIO "<<relPDecTTofNCtoIO<<endl;
	outSt<<"relPDecT0ofNCtoIO "<<relPDecT0ofNCtoIO<<endl;
	outSt<<"relPIncNCtoIO "<<relPIncNCtoIO<<endl;
	outSt<<"relPIncTauNCtoIO "<<relPIncTauNCtoIO<<endl;
	outSt<<"initSynWofMFtoNC "<<initSynWofMFtoNC<<endl;
	outSt<<"synLTDPCPopActThreshMFtoNC "<<synLTDPCPopActThreshMFtoNC<<endl;
	outSt<<"synLTPPCPopActThreshMFtoNC "<<synLTPPCPopActThreshMFtoNC<<endl;
	outSt<<"synLTDStepSizeMFtoNC "<<synLTDStepSizeMFtoNC<<endl;
	outSt<<"synLTPStepSizeMFtoNC "<<synLTPStepSizeMFtoNC<<endl;
}

map<string, float> ActivityParams::getParamCopy()
{
	map<string, float> paramCopy;

	map<string, float>::iterator i;

	for(i=paramMap.begin(); i!=paramMap.end(); i++)
	{
		paramCopy[i->first]=i->second;
	}

	return paramCopy;
}

float ActivityParams::getParam(string paramName)
{
	return paramMap[paramName];
}

bool ActivityParams::setParam(string paramName, float value)
{
	if(paramMap.find(paramName)==paramMap.end())
	{
		return false;
	}
	paramMap[paramName]=value;

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
	msPerTimeStep=paramMap["msPerTimeStep"];

	msPerHistBinMF=paramMap["msPerHistBinMF"];
	numTSinMFHist=msPerHistBinMF/msPerTimeStep;

	//move elements from map to public variables
//	paramMap.
	if(paramMap.find("coupleRiRjRatioGO")==paramMap.end())
	{
		paramMap["coupleRiRjRatioGO"]=0;
	}

	if(paramMap.find("goGABAGOGOSynRecTau")==paramMap.end())
	{
		paramMap["goGABAGOGOSynRecTau"]=1;
	}

	if(paramMap.find("goGABAGOGOSynDepF")==paramMap.end())
	{
		paramMap["goGABAGOGOSynDepF"]=1;
	}

	eLeakGO=paramMap["eLeakGO"];
	eMGluRGO=paramMap["eMGluRGO"];
	eGABAGO=paramMap["eGABAGO"];
	threshMaxGO=paramMap["threshMaxGO"];
	threshRestGO=paramMap["threshBaseGO"];
	gIncMFtoGO=paramMap["gMFIncGO"];
	gIncGRtoGO=paramMap["gGRIncGO"];
	gGABAIncGOtoGO=paramMap["gGOIncGO"];
	coupleRiRjRatioGO=paramMap["coupleRiRjRatioGO"];

	gMGluRScaleGRtoGO=paramMap["gMGluRScaleGO"];
	gMGluRIncScaleGO=paramMap["gMGluRIncScaleGO"];
	mGluRScaleGO=paramMap["mGluRScaleGO"];
	gluScaleGO=paramMap["gluScaleGO"];
	gLeakGO=paramMap["rawGLeakGO"]/(6-msPerTimeStep);

	gDecTauMFtoGO=paramMap["gMFDecayTGO"];
	gDecMFtoGO=exp(-msPerTimeStep/gDecTauMFtoGO);
	NMDA_AMPAratioMFGO=paramMap["NMDA_AMPAratioMFGO"];
	gDecTauMFtoGONMDA=paramMap["gDecTauMFtoGONMDA"];
	gDecayMFtoGONMDA=exp(-msPerTimeStep/gDecTauMFtoGONMDA);

	gDecTauGRtoGO=paramMap["gGRDecayTGO"];
	gDecGRtoGO=exp(-msPerTimeStep/gDecTauGRtoGO);

	gGABADecTauGOtoGO=paramMap["gGODecayTGO"];
	gGABADecGOtoGO=exp(-msPerTimeStep/gGABADecTauGOtoGO);

	//synaptic depression test for GOGABAGO
	goGABAGOGOSynRecTau=paramMap["goGABAGOGOSynRecTau"];
	goGABAGOGOSynRec=1-exp(-msPerTimeStep/goGABAGOGOSynRecTau);
	goGABAGOGOSynDepF=paramMap["goGABAGOGOSynDepF"];

	mGluRDecayGO=paramMap["mGluRDecayGO"];
	gMGluRIncDecayGO=paramMap["gMGluRIncDecayGO"];
	gMGluRDecGRtoGO=paramMap["gMGluRDecayGO"];
	gluDecayGO=paramMap["gluDecayGO"];

	threshDecTauGO=paramMap["threshDecayTGO"];
	threshDecGO=1-exp(-msPerTimeStep/threshDecTauGO);


	eLeakGR=paramMap["eLeakGR"];
	eGOGR=paramMap["eGOGR"];
	eMFGR=paramMap["eMFGR"];
	threshMaxGR=paramMap["threshMaxGR"];
	threshRestGR=paramMap["threshBaseGR"];

	//gIncMFtoGR=paramMap["gMFIncGR"];	
	//gDecTauMFtoGR=paramMap["gMFDecayTGR"];
	//gDecMFtoGR=exp(-msPerTimeStep/gDecTauMFtoGR);

	
	//gIncGOtoGR=paramMap["gGOIncGR"];	
	//gDecTauGOtoGR=paramMap["gGODecayTGR"];
	//gDecGOtoGR=exp(-msPerTimeStep/gDecTauGOtoGR);

	threshDecTauGR=paramMap["threshDecayTGR"];
	threshDecGR=1-exp(-msPerTimeStep/threshDecTauGR);

	gLeakGR=paramMap["rawGLeakGR"]/(6-msPerTimeStep);

	msPerHistBinGR=paramMap["msPerHistBinGR"];
	tsPerHistBinGR=msPerHistBinGR/msPerTimeStep;

	eLeakSC=paramMap["eLeakSC"];
	gLeakSC=paramMap["rawGLeakSC"]/(6-msPerTimeStep);
	gDecTauGRtoSC=paramMap["gPFDecayTSC"];
	gDecGRtoSC=exp(-msPerTimeStep/gDecTauGRtoSC);
	threshMaxSC=paramMap["threshMaxSC"];
	threshRestSC=paramMap["threshBaseSC"];
	threshDecTauSC=paramMap["threshDecayTSC"];
	threshDecSC=1-exp(-msPerTimeStep/threshDecTauSC);
	gIncGRtoSC=paramMap["pfIncSC"];

	//**From mzone**
	eLeakBC=paramMap["eLeakBC"];
	ePCtoBC=paramMap["ePCBC"];
	gLeakBC=paramMap["rawGLeakBC"];

	gDecTauGRtoBC=paramMap["gPFDecayTBC"];
	gDecGRtoBC=exp(-msPerTimeStep/gDecTauGRtoBC);

	gDecTauPCtoBC=paramMap["gPCDecayTBC"];
	gDecPCtoBC=exp(-msPerTimeStep/gDecTauPCtoBC);

	threshDecTauBC=paramMap["threshDecayTBC"];
	threshDecBC=1-exp(-msPerTimeStep/threshDecTauBC);

	threshRestBC=paramMap["threshBaseBC"];
	threshMaxBC=paramMap["threshMaxBC"];
	gIncGRtoBC=paramMap["pfIncConstBC"];
	gIncPCtoBC=paramMap["pcIncConstBC"];

	initSynWofGRtoPC=paramMap["pfSynWInitPC"];
	eLeakPC=paramMap["eLeakPC"];
	eBCtoPC=paramMap["eBCPC"];
	eSCtoPC=paramMap["eSCPC"];
	threshMaxPC=paramMap["threshMaxPC"];
	threshRestPC=paramMap["threshBasePC"];

	threshDecTauPC=paramMap["threshDecayTPC"];
	threshDecPC=1-exp(-msPerTimeStep/threshDecTauPC);

	gLeakPC=paramMap["rawGLeakPC"]/(6-msPerTimeStep);

	gDecTauGRtoPC=paramMap["gPFDecayTPC"];
	gDecGRtoPC=exp(-msPerTimeStep/gDecTauGRtoPC);

	gDecTauBCtoPC=paramMap["gBCDecayTPC"];
	gDecBCtoPC=exp(-msPerTimeStep/gDecTauBCtoPC);

	gDecTauSCtoPC=paramMap["gSCDecayTPC"];
	gDecSCtoPC=exp(-msPerTimeStep/gDecTauSCtoPC);

	gIncSCtoPC=paramMap["gSCIncConstPC"];
	gIncGRtoPC=paramMap["gPFScaleConstPC"];
	gIncBCtoPC=paramMap["gBCScaleConstPC"];

	tsPopHistPC=40/msPerTimeStep; //TODO: fixed for now
	tsPerPopHistBinPC=5/msPerTimeStep;
	numPopHistBinsPC=tsPopHistPC/tsPerPopHistBinPC;

	coupleRiRjRatioIO=paramMap["coupleScaleIO"];
	eLeakIO=paramMap["eLeakIO"];
	eNCtoIO=paramMap["eNCIO"];
	gLeakIO=paramMap["rawGLeakIO"]/(6-msPerTimeStep);
	gDecTSofNCtoIO=paramMap["gNCDecTSIO"];
	gDecTTofNCtoIO=paramMap["gNCDecTTIO"];
	gDecT0ofNCtoIO=paramMap["gNCDecT0IO"];
	gIncNCtoIO=paramMap["gNCIncScaleIO"];
	gIncTauNCtoIO=paramMap["gNCIncTIO"];
	threshRestIO=paramMap["threshBaseIO"];
	threshMaxIO=paramMap["threshMaxIO"];

	threshDecTauIO=paramMap["threshDecayTIO"];
	threshDecIO=1-exp(-msPerTimeStep/threshDecTauIO);

	tsLTDDurationIO=paramMap["msLTDDurationIO"]/msPerTimeStep;
	tsLTDStartAPIO=paramMap["msLTDStartAPIO"]/msPerTimeStep;
	tsLTPStartAPIO=paramMap["msLTPStartAPIO"]/msPerTimeStep;
	tsLTPEndAPIO=paramMap["msLTPEndAPIO"]/msPerTimeStep;
	synLTPStepSizeGRtoPC=paramMap["grPCLTPIncIO"];
	synLTDStepSizeGRtoPC=paramMap["grPCLTDDecIO"];
	grPCHistCheckBinIO=abs(tsLTPEndAPIO/((int)tsPerHistBinGR));

	maxExtIncVIO=paramMap["maxErrDriveIO"];

	eLeakNC=paramMap["eLeakNC"];
	ePCtoNC=paramMap["ePCNC"];

	gmaxNMDADecTauMFtoNC=paramMap["mfNMDADecayTNC"];
	gmaxNMDADecMFtoNC=exp(-msPerTimeStep/gmaxNMDADecTauMFtoNC);

	gmaxAMPADecTauMFtoNC=paramMap["mfAMPADecayTNC"];
	gmaxAMPADecMFtoNC=exp(-msPerTimeStep/gmaxAMPADecTauMFtoNC);

	gNMDAIncMFtoNC=1-exp(-msPerTimeStep/paramMap["rawGMFNMDAIncNC"]);
	gAMPAIncMFtoNC=1-exp(-msPerTimeStep/paramMap["rawGMFAMPAIncNC"]);
	gIncAvgPCtoNC=paramMap["gPCScaleAvgNC"];

	gDecTauPCtoNC=paramMap["gPCDecayTNC"];
	gDecPCtoNC=exp(-msPerTimeStep/gDecTauPCtoNC);

	gLeakNC=paramMap["rawGLeakNC"]/(6-msPerTimeStep);

	threshDecTauNC=paramMap["threshDecayTNC"];
	threshDecNC=1-exp(-msPerTimeStep/threshDecTauNC);

	threshMaxNC=paramMap["threshMaxNC"];
	threshRestNC=paramMap["threshBaseNC"];
	relPDecTSofNCtoIO=paramMap["outIORelPDecTSNC"];
	relPDecTTofNCtoIO=paramMap["outIORelPDecTTNC"];
	relPDecT0ofNCtoIO=paramMap["outIORelPDecT0NC"];
	relPIncNCtoIO=paramMap["outIORelPIncScaleNC"];
	relPIncTauNCtoIO=paramMap["outIORelPIncTNC"];
	initSynWofMFtoNC=paramMap["mfSynWInitNC"];
	synLTDPCPopActThreshMFtoNC=paramMap["mfNCLTDThreshNC"];
	synLTPPCPopActThreshMFtoNC=paramMap["mfNCLTPThreshNC"];
	synLTDStepSizeMFtoNC=paramMap["mfNCLTDDecNC"];
	synLTPStepSizeMFtoNC=paramMap["mfNCLTPIncNC"];
}


//updateParamsV1() --- parameterVersion == 1
//---> updated parameter names to match variables
//---> includes ability to add comments
//---> includes parameterVersion flag 

void ActivityParams::updateParamsV1()
{
	msPerTimeStep=paramMap["msPerTimeStep"];

	msPerHistBinMF=paramMap["msPerHistBinMF"];
	numTSinMFHist=msPerHistBinMF/msPerTimeStep;

	//move elements from map to public variables
//	paramMap.
	if(paramMap.find("coupleRiRjRatioGO")==paramMap.end())
	{
		paramMap["coupleRiRjRatioGO"]=0;
	}

	if(paramMap.find("goGABAGOGOSynRecTau")==paramMap.end())
	{
		paramMap["goGABAGOGOSynRecTau"]=1;
	}

	if(paramMap.find("goGABAGOGOSynDepF")==paramMap.end())
	{
		paramMap["goGABAGOGOSynDepF"]=1;
	}

	
	
	gIncMFtoUBC=paramMap["gIncMFtoUBC"];
	gIncGOtoUBC=paramMap["gIncGOtoUBC"];	
	gIncUBCtoUBC=paramMap["gIncUBCtoUBC"];
	gIncUBCtoGO=paramMap["gIncUBCtoGO"];
	gIncUBCtoGR=paramMap["gIncUBCtoGR"];

	gKIncUBC=paramMap["gKIncUBC"];
	gKTauUBC=paramMap["gKTauUBC"];	
	gConstUBC=paramMap["gConstUBC"];
	threshTauUBC=paramMap["threshTauUBC"];

	threshDecTauUBC=paramMap["threshDecayTauUBC"];
	threshDecUBC=1-exp(-msPerTimeStep/threshDecTauUBC);

	eLeakGO=paramMap["eLeakGO"];
	eMGluRGO=paramMap["eMGluRGO"];
	eGABAGO=paramMap["eGABAGO"];
	threshMaxGO=paramMap["threshMaxGO"];
	threshRestGO=paramMap["threshRestGO"];
	gIncMFtoGO=paramMap["gIncMFtoGO"];
	gIncGRtoGO=paramMap["gIncGRtoGO"];
	gGABAIncGOtoGO=paramMap["gGABAIncGOtoGO"];
	coupleRiRjRatioGO=paramMap["coupleRiRjRatioGO"];

	gMGluRScaleGRtoGO=paramMap["gMGluRScaleGRtoGO"];
	gMGluRIncScaleGO=paramMap["gMGluRIncScaleGO"];
	mGluRScaleGO=paramMap["mGluRScaleGO"];
	gluScaleGO=paramMap["gluScaleGO"];
	gLeakGO=paramMap["rawGLeakGO"]/(6-msPerTimeStep);

//	gDecTauMFtoGO=paramMap["gDecTauMFtoGO"];
//	gDecMFtoGO=exp(-msPerTimeStep/gDecTauMFtoGO);
	
	gDecTauMFtoGO=paramMap["gDecTauMFtoGO"];
	gDecMFtoGO=exp(-msPerTimeStep/gDecTauMFtoGO);
	NMDA_AMPAratioMFGO=paramMap["NMDA_AMPAratioMFGO"];
	gDecTauMFtoGONMDA=paramMap["gDecTauMFtoGONMDA"];
	gDecayMFtoGONMDA=exp(-msPerTimeStep/gDecTauMFtoGONMDA);

	gDecTauGRtoGO=paramMap["gDecTauGRtoGO"];
	gDecGRtoGO=exp(-msPerTimeStep/gDecTauGRtoGO);

	gGABADecTauGOtoGO=paramMap["gGABADecTauGOtoGO"];
	gGABADecGOtoGO=exp(-msPerTimeStep/gGABADecTauGOtoGO);

	//synaptic depression test for GOGABAGO
	goGABAGOGOSynRecTau=paramMap["goGABAGOGOSynRecTau"];
	goGABAGOGOSynRec=1-exp(-msPerTimeStep/goGABAGOGOSynRecTau);
	goGABAGOGOSynDepF=paramMap["goGABAGOGOSynDepF"];

	mGluRDecayGO=paramMap["mGluRDecayGO"];
	gMGluRIncDecayGO=paramMap["gMGluRIncDecayGO"];
	gMGluRDecGRtoGO=paramMap["gMGluRDecGRtoGO"];
	gluDecayGO=paramMap["gluDecayGO"];

	gConstGO=paramMap["gConstGO"];
	threshDecTauGO=paramMap["threshDecTauGO"];
	threshDecGO=1-exp(-msPerTimeStep/threshDecTauGO);


	eLeakGR=paramMap["eLeakGR"];
	eGOGR=paramMap["eGOGR"];
	eMFGR=paramMap["eMFGR"];
	threshMaxGR=paramMap["threshMaxGR"];
	threshRestGR=paramMap["threshRestGR"];

	gIncDirectMFtoGR=paramMap["gIncDirectMFtoGR"];
	gDirectTauMFtoGR=paramMap["gDecTauMFtoGR"];
	gDirectDecMFtoGR=exp(-msPerTimeStep/gDirectTauMFtoGR);
	gIncFracSpilloverMFtoGR=paramMap["gIncFracSpilloverMFtoGR"];
	gSpilloverTauMFtoGR=paramMap["gSpilloverTauMFtoGR"];
	gSpilloverDecMFtoGR=exp(-msPerTimeStep/gSpilloverTauMFtoGR);
	recoveryTauMF=paramMap["recoveryTauMF"];
	fracDepMF=paramMap["fracDepMF"];
	
	
	gIncDirectGOtoGR=paramMap["gIncDirectGOtoGR"];
	gDirectTauGOtoGR=paramMap["gDirectTauGOtoGR"];
	gDirectDecGOtoGR=exp(-msPerTimeStep/gDirectTauGOtoGR);
	gIncFracSpilloverGOtoGR=paramMap["gIncFracSpilloverGOtoGR"];
	gSpilloverTauGOtoGR=paramMap["gSpilloverTauGOtoGR"];
	gSpilloverDecGOtoGR=exp(-msPerTimeStep/gSpilloverTauGOtoGR);

	recoveryTauGO=paramMap["recoveryTauGO"];
	fracDepGO=paramMap["fracDepGO"];
	
	
	
	
	threshDecTauGR=paramMap["threshDecTauGR"];
	threshDecGR=1-exp(-msPerTimeStep/threshDecTauGR);

	gLeakGR=paramMap["rawGLeakGR"]/(6-msPerTimeStep);

	msPerHistBinGR=paramMap["msPerHistBinGR"];
	tsPerHistBinGR=msPerHistBinGR/msPerTimeStep;

	eLeakSC=paramMap["eLeakSC"];
	gLeakSC=paramMap["rawGLeakSC"]/(6-msPerTimeStep);
	gDecTauGRtoSC=paramMap["gDecTauGRtoSC"];
	gDecGRtoSC=exp(-msPerTimeStep/gDecTauGRtoSC);
	threshMaxSC=paramMap["threshMaxSC"];
	threshRestSC=paramMap["threshRestSC"];
	threshDecTauSC=paramMap["threshDecTauSC"];
	threshDecSC=1-exp(-msPerTimeStep/threshDecTauSC);
	gIncGRtoSC=paramMap["gIncGRtoSC"];

	//**From mzone**
	eLeakBC=paramMap["eLeakBC"];
	ePCtoBC=paramMap["ePCtoBC"];
	gLeakBC=paramMap["rawGLeakBC"];

	gDecTauGRtoBC=paramMap["gDecTauGRtoBC"];
	gDecGRtoBC=exp(-msPerTimeStep/gDecTauGRtoBC);

	gDecTauPCtoBC=paramMap["gDecTauPCtoBC"];
	gDecPCtoBC=exp(-msPerTimeStep/gDecTauPCtoBC);

	threshDecTauBC=paramMap["threshDecTauBC"];
	threshDecBC=1-exp(-msPerTimeStep/threshDecTauBC);

	threshRestBC=paramMap["threshRestBC"];
	threshMaxBC=paramMap["threshMaxBC"];
	gIncGRtoBC=paramMap["gIncGRtoBC"];
	gIncPCtoBC=paramMap["gIncPCtoBC"];

	initSynWofGRtoPC=paramMap["initSynWofGRtoPC"];
	eLeakPC=paramMap["eLeakPC"];
	eBCtoPC=paramMap["eBCtoPC"];
	eSCtoPC=paramMap["eSCtoPC"];
	threshMaxPC=paramMap["threshMaxPC"];
	threshRestPC=paramMap["threshRestPC"];

	threshDecTauPC=paramMap["threshDecTauPC"];
	threshDecPC=1-exp(-msPerTimeStep/threshDecTauPC);

	gLeakPC=paramMap["rawGLeakPC"]/(6-msPerTimeStep);

	gDecTauGRtoPC=paramMap["gDecTauGRtoPC"];
	gDecGRtoPC=exp(-msPerTimeStep/gDecTauGRtoPC);

	gDecTauBCtoPC=paramMap["gDecTauBCtoPC"];
	gDecBCtoPC=exp(-msPerTimeStep/gDecTauBCtoPC);

	gDecTauSCtoPC=paramMap["gDecTauSCtoPC"];
	gDecSCtoPC=exp(-msPerTimeStep/gDecTauSCtoPC);

	gIncSCtoPC=paramMap["gIncSCtoPC"];
	gIncGRtoPC=paramMap["gIncGRtoPC"];
	gIncBCtoPC=paramMap["gIncBCtoPC"];

	tsPopHistPC=40/msPerTimeStep; //TODO: fixed for now
	tsPerPopHistBinPC=5/msPerTimeStep;
	numPopHistBinsPC=tsPopHistPC/tsPerPopHistBinPC;

	coupleRiRjRatioIO=paramMap["coupleRiRjRatioIO"];
	eLeakIO=paramMap["eLeakIO"];
	eNCtoIO=paramMap["eNCtoIO"];
	gLeakIO=paramMap["rawGLeakIO"]/(6-msPerTimeStep);
	gDecTSofNCtoIO=paramMap["gDecTSofNCtoIO"];
	gDecTTofNCtoIO=paramMap["gDecTTofNCtoIO"];
	gDecT0ofNCtoIO=paramMap["gDecT0ofNCtoIO"];
	gIncNCtoIO=paramMap["gIncNCtoIO"];
	gIncTauNCtoIO=paramMap["gIncTauNCtoIO"];
	threshRestIO=paramMap["threshRestIO"];
	threshMaxIO=paramMap["threshMaxIO"];

	threshDecTauIO=paramMap["threshDecTauIO"];
	threshDecIO=1-exp(-msPerTimeStep/threshDecTauIO);

	tsLTDDurationIO=paramMap["msLTDDurationIO"]/msPerTimeStep;
	tsLTDStartAPIO=paramMap["msLTDStartAPIO"]/msPerTimeStep;
	tsLTPStartAPIO=paramMap["msLTPStartAPIO"]/msPerTimeStep;
	tsLTPEndAPIO=paramMap["msLTPEndAPIO"]/msPerTimeStep;
	synLTPStepSizeGRtoPC=paramMap["synLTPStepSizeGRtoPC"];
	synLTDStepSizeGRtoPC=paramMap["synLTDStepSizeGRtoPC"];
	grPCHistCheckBinIO=abs(tsLTPEndAPIO/((int)tsPerHistBinGR));

	maxExtIncVIO=paramMap["maxExtIncVIO"];

	eLeakNC=paramMap["eLeakNC"];
	ePCtoNC=paramMap["ePCtoNC"];

	gmaxNMDADecTauMFtoNC=paramMap["gmaxNMDADecTauMFtoNC"];
	gmaxNMDADecMFtoNC=exp(-msPerTimeStep/gmaxNMDADecTauMFtoNC);

	gmaxAMPADecTauMFtoNC=paramMap["gmaxAMPADecTauMFtoNC"];
	gmaxAMPADecMFtoNC=exp(-msPerTimeStep/gmaxAMPADecTauMFtoNC);

	gNMDAIncMFtoNC=2.35;//0.2835;   //1-exp(-msPerTimeStep/paramMap["rawGMFNMDAIncNC"]);
	gAMPAIncMFtoNC=2.35;//1-exp(-msPerTimeStep/paramMap["rawGMFAMPAIncNC"]);
	gIncAvgPCtoNC=paramMap["gIncAvgPCtoNC"];

	gDecTauPCtoNC=paramMap["gDecTauPCtoNC"];
	gDecPCtoNC=exp(-msPerTimeStep/gDecTauPCtoNC);

	gLeakNC=paramMap["rawGLeakNC"]/(6-msPerTimeStep);

	threshDecTauNC=paramMap["threshDecTauNC"];
	threshDecNC=1-exp(-msPerTimeStep/threshDecTauNC);

	threshMaxNC=paramMap["threshMaxNC"];
	threshRestNC=paramMap["threshRestNC"];
	relPDecTSofNCtoIO=paramMap["relPDecTSofNCtoIO"];
	relPDecTTofNCtoIO=paramMap["relPDecTTofNCtoIO"];
	relPDecT0ofNCtoIO=paramMap["relPDecT0ofNCtoIO"];
	relPIncNCtoIO=paramMap["relPIncNCtoIO"];
	relPIncTauNCtoIO=paramMap["relPIncTauNCtoIO"];
	initSynWofMFtoNC=paramMap["initSynWofMFtoNC"];
	synLTDPCPopActThreshMFtoNC=paramMap["synLTDPCPopActThreshMFtoNC"];
	synLTPPCPopActThreshMFtoNC=paramMap["synLTPPCPopActThreshMFtoNC"];
	synLTDStepSizeMFtoNC=paramMap["synLTDStepSizeMFtoNC"];
	synLTPStepSizeMFtoNC=paramMap["synLTPStepSizeMFtoNC"];
}
