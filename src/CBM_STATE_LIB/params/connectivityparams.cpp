/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "params/connectivityparams.h"

ConnectivityParams::ConnectivityParams(std::fstream &infile)
{
	//Assumes that file is in the following format:
	//key\tvalue\n
	//key\tvalue\n

	//loop through file and add key/value pair to map
	//** this is done to remove the necessity of order in the original file

	string key;
	ct_uint32_t val;

	bool hasGOGOCouple;

	for (int i = 0; i < 1000; i++)
	{
		infile >> key;

		if (key.compare("connectivityParamEnd") == 0)
		{
			infile >> val;
			break;
		}
		else
		{
			infile >> val;
			paramMap[key] = val;
		}
	}
	
	//move elements from map to public variables
	ubcX   = paramMap["ubcX"];
	ubcY   = paramMap["ubcY"];
	numUBC = ubcX*ubcY;

	spanGLtoUBCX = paramMap["spanGLtoUBCX"];
	spanGLtoUBCY = paramMap["spanGLtoUBCY"];
	numpGLtoUBC  = (spanGLtoUBCX + 1) * (spanGLtoUBCY + 1);
	
	spanUBCtoGLX = paramMap["spanUBCtoGLX"];	
	spanUBCtoGLY = paramMap["spanUBCtoGLY"];
	numpUBCtoGL  = (spanUBCtoGLX + 1) * (spanUBCtoGLY + 1);
	
	
	maxnumpGOGABAInGOGO  = paramMap["maxnumpGOInGOtoGO"];
	maxnumpGOGABAOutGOGO = paramMap["maxnumpGOOutGOtoGO"];
	maxnumpGOCoupInGOGO  = paramMap["maxnumpGOCoupInGOtoGO"];
	maxnumpGOCoupOutGOGO = paramMap["maxnumpGOCoupOutGOtoGO"];
	
	
	//glomerulus
	glX   = paramMap["glX"];
	glY   = paramMap["glY"];
	numGL = glX * glY;

	maxnumpGLfromGLtoGR = paramMap["maxnumpGLfromGLtoGR"];
	lownumpGLfromGLtoGR = paramMap["lownumpGLfromGLtoGR"];
	maxnumpGLfromGLtoGO = paramMap["maxnumpGLfromGLtoGO"];
	maxnumpGLfromGOtoGL = paramMap["maxnumpGLfromGOtoGL"];

	//mossy fibers
	numMF = paramMap["numMFP2"];
	mfX   = paramMap["mfX"];
	mfY   = paramMap["mfY"];

	spanMFtoGLX = paramMap["spanMFtoGLX"];
	spanMFtoGLY = paramMap["spanMFtoGLY"];
	numpMFtoGL  = (spanMFtoGLX + 1) * (spanMFtoGLY + 1);
	
	maxnumpMFfromMFtoGO = 20 * maxnumpGLfromGLtoGO;
	maxnumpMFfromMFtoGR = 20 * maxnumpGLfromGLtoGR;

	numpMFfromMFtoNC = 1; //fixed for now

	//golgi cells
	goX 	   = paramMap["goX"];
	goY 	   = paramMap["goY"];
	numGO 	   = goX*goY;
	numConGOGO = paramMap["numConGOGO"];

	spanGOGOsynX   = paramMap["spanGOGOsynX"];
	spanGOGOsynY   = paramMap["spanGOGOsynY"];
	numpGOGOsyn    = (spanGOGOsynX + 1) * (spanGOGOsynY + 1);
	sigmaGOGOsynML = paramMap["sigmaGOGOsynML"];
	sigmaGOGOsynS  = paramMap["sigmaGOGOsynS"];
	maxGOGOsyn     = paramMap["maxGOGOsyn"];
	
	spanGOGLX = paramMap["spanGOGLX"];
	spanGOGLY = paramMap["spanGOGLY"];
	numpGOGL  = (spanGOGLX + 1) * (spanGOGLY + 1);

	maxnumpGOfromGRtoGO = paramMap["maxnumpGOfromGRtoGO"];
	maxnumpGOfromGLtoGO = paramMap["maxnumpGOfromGLtoGO"];
	maxnumpGOfromMFtoGO = maxnumpGOfromGLtoGO;
	maxnumpGOfromGOtoGL = paramMap["maxnumpGOfromGOtoGL"];
	maxnumpGOfromGOtoGR = maxnumpGOfromGOtoGL*maxnumpGLfromGLtoGR;

	spanGODecDenOnGLX = paramMap["spanGODecDenOnGLX"];
	spanGODecDenOnGLY = paramMap["spanGODecDenOnGLY"];

	spanGOAscDenOnGRX = paramMap["spanGOAscDenOnGRX"];
	spanGOAscDenOnGRY = paramMap["spanGOAscDenOnGRY"];

	spanGOAxonOnGLX = paramMap["spanGOAxonOnGLX"];
	spanGOAxonOnGLY = paramMap["spanGOAxonOnGLY"];

	spanGOtoGLX = paramMap["spanGOtoGLX"];
	spanGOtoGLY = paramMap["spanGOtoGLY"];
	numpGOtoGL  = (spanGOtoGLX + 1) * (spanGOtoGLY + 1);

	maxnumpGOCoupInGOGO  = paramMap["maxnumpGOCoupInGOtoGO"];
	maxnumpGOCoupOutGOGO = paramMap["maxnumpGOCoupOutGOtoGO"];

	//granule cells
	grX   = paramMap["grX"];
	grY   = paramMap["grY"];
	numGR = grX * grY;
	
	grPFVelInGRXPerTStep = paramMap["grPFVelInGRXPerTStep"];
	grAFDelayInTStep	 = paramMap["grAFDelayInTStep"];
	maxnumpGRfromGRtoGO  = paramMap["maxnumpGRfromGRtoGO"];
	maxnumpGRfromGLtoGR  = paramMap["maxnumpGRfromGLtoGR"];
	maxnumpGRfromGOtoGR  = paramMap["maxnumpGRfromGOtoGR"];
	maxnumpGRfromMFtoGR  = paramMap["maxnumpGRfromMFtoGR"];

	spanGRDenOnGLX = paramMap["spanGRDenOnGLX"];
	spanGRDenOnGLY = paramMap["spanGRDenOnGLY"];

	//TODO: a lot of MZone connectivity is fixed right now,
	//will add flexibility as needed
	numSC			   = 1 << paramMap["numSCP2"];
	numpSCfromGRtoSC   = numGR / numSC;
	numpSCfromGRtoSCP2 = numGRP2 - paramMap["numSCP2"];
	numpSCfromSCtoPC   = 1;

	numPC			   = 1 << paramMap["numPCP2"];
	numpPCfromGRtoPC   = numGR / numPC;
	numpPCfromGRtoPCP2 = numGRP2 - paramMap["numPCP2"];
	numpPCfromBCtoPC   = 16;//fixed for now
	numpPCfromPCtoBC   = 16; //fixed for now
	numpPCfromSCtoPC   = numSC / numPC;
	numpPCfromPCtoNC   = paramMap["numpPCfromPCtoNC"];

	numBC=numPC*4;
	//fixed for now since the connectivity pattern depends on this
	//also not that important to have the MZone network very flexible right now
	numpBCfromGRtoBC   = numGR / numBC;
	numpBCfromGRtoBCP2 = numGRP2 - (paramMap["numPCP2"] + 2);
	//+2 is connected to the *4 from above
	numpBCfromBCtoPC = 4;
	numpBCfromPCtoBC = 4;

	numNC			 = 1 << paramMap["numNCP2"];
	numpNCfromPCtoNC = numPC / numNC * numpPCfromPCtoNC;
	numpNCfromNCtoIO = 1 << paramMap["numIOP2"]; //all to all connection
	numpNCfromMFtoNC = numMF / numNC;

	numIO			 = 1 << paramMap["numIOP2"];
	numpIOfromIOtoPC = numPC / numIO;
	numpIOfromNCtoIO = numNC; //all to all connection
	numpIOInIOIO	 = numIO - 1;
	numpIOOutIOIO	 = numIO - 1;
}

ConnectivityParams::~ConnectivityParams()
{

}

void ConnectivityParams::writeParams(std::fstream &outfile)
{

	for (auto i = paramMap.begin(); i != paramMap.end(); i++)
	{
		outfile << i->first << " " << i->second << std::endl;
	}

	outfile << "GOGOLocalMatrix" << std::endl;
	outfile << "maxnumpGOInGOtoGO " << maxnumpGOGABAInGOGO << std::endl;
	outfile << "maxnumpGOOutGOtoGO " << maxnumpGOGABAOutGOGO << std::endl;
	for (int i = 0; i < maxnumpGOGABAOutGOGO; i++)
	{
		outfile << gogoGABALocalCon[i][0] << " " << gogoGABALocalCon[i][1] << " " <<
			gogoGABALocalCon[i][2] << std::endl;
	}

	outfile << "GOGOCoupLocalMatrix" << std::endl;
	outfile << "maxnumpGOCoupInGOtoGO " << maxnumpGOCoupInGOGO << std::endl;
	outfile << "maxnumpGOCoupOutGOtoGO " << maxnumpGOCoupOutGOGO << std::endl;
	for (int i = 0; i < maxnumpGOCoupOutGOGO; i++)
	{
		outfile << gogoCoupLocalCon[i][0] << " " << gogoCoupLocalCon[i][1] << " " <<
			gogoCoupLocalCon[i][2] << std::endl;
	}

	outfile << "connectivityParamEnd 1" << std::endl;
}

void ConnectivityParams::showParams(std::ostream &outSt)
{
	outSt << "glX " << glX << std::endl;
	outSt << "glY " << glY << std::endl;
	outSt << "numGL "<< numGL << std::endl;
	outSt << "maxnumpGLfromGLtoGR "<< maxnumpGLfromGLtoGR << std::endl;
	outSt << "lownumpGLfromGLtoGR "<< lownumpGLfromGLtoGR << std::endl;
	outSt << "maxnumpGLfromGLtoGO "<< maxnumpGLfromGLtoGO << std::endl;
	outSt << "maxnumpGLfromGOtoGL "<< maxnumpGLfromGOtoGL << std::endl << std::endl;

	outSt << "numMF " << numMF << std::endl;
	outSt << "maxnumpMFfromMFtoGO " << maxnumpMFfromMFtoGO << std::endl;
	outSt << "maxnumpMFfromMFtoGR " << maxnumpMFfromMFtoGR << std::endl;
	outSt << "numpMFfromMFtoNC " << numpMFfromMFtoNC << std::endl << std::endl;

	outSt << "goX " << goX << std::endl;
	outSt << "goY " << goY << std::endl;
	outSt << "numGO " << numGO << std::endl;
	outSt << "maxnumpGOfromGRtoGO " << maxnumpGOfromGRtoGO << std::endl;
	outSt << "maxnumpGOfromGLtoGO " << maxnumpGOfromGLtoGO << std::endl;
	outSt << "maxnumpGOfromMFtoGO " << maxnumpGOfromMFtoGO << std::endl;
	outSt << "maxNumpGOfromGOtoGL " << maxnumpGOfromGOtoGL << std::endl;
	outSt << "maxnumpGOfromGOtoGR " << maxnumpGOfromGOtoGR << std::endl;
	outSt << "spanGODecDenOnGLX " << spanGODecDenOnGLX << std::endl;
	outSt << "spanGODecDenOnGLY " << spanGODecDenOnGLY << std::endl;
	outSt << "spanGOAscDenOnGRX " << spanGOAscDenOnGRX << std::endl;
	outSt << "spanGOAscDenOnGRY " << spanGOAscDenOnGRY << std::endl;
	outSt << "spanGOAxonOnGLX "<< spanGOAxonOnGLX << std::endl;
	outSt << "spanGOAxonOnGLY "<< spanGOAxonOnGLY << std::endl;
	outSt << "maxnumpGOInGOGO "<< maxnumpGOGABAInGOGO << std:: endl;
	outSt << "maxnumpGOOutGOGO "<< maxnumpGOGABAOutGOGO << std::endl;
	outSt << "gogoLocalMatrix:" << std::endl;

	for (int i = 0; i < maxnumpGOGABAOutGOGO; i++)
	{
		outSt << gogoGABALocalCon[i][0] << " " << gogoGABALocalCon[i][1] << " " <<
			gogoGABALocalCon[i][2] << std::endl;
	}

	outSt << "maxnumpGOCoupInGOGO " << maxnumpGOCoupInGOGO << std::endl;
	outSt << "maxnumpGOCoupOutGOGO " << maxnumpGOCoupOutGOGO << std::endl;
	outSt << "gogoCoupLocalMatrix:" << std::endl;
	
	for (int i = 0; i < maxnumpGOCoupOutGOGO; i++)
	{
		outSt << gogoCoupLocalCon[i][0] << " " << gogoCoupLocalCon[i][1] << " " <<
			gogoCoupLocalCon[i][2] << std::endl;
	}

	outSt << "grX " << grX << std::endl;
	outSt << "grY " << grY << std::endl;
	outSt << "numGR " << numGR << std::endl;
	outSt << "grPFVelInGRXPerTStep " << grPFVelInGRXPerTStep << std::endl;
	outSt << "grAFDelayInTStep " << grAFDelayInTStep << std::endl;
	outSt << "maxnumpGRfromGRtoGO " << maxnumpGRfromGRtoGO << std::endl;
	outSt << "maxnumpGRfromGLtoGR " << maxnumpGRfromGLtoGR << std::endl;
	outSt << "maxnumpGRfromGOtoGR " << maxnumpGRfromGOtoGR << std::endl;
	outSt << "maxnumpGRfromMFtoGR " << maxnumpGRfromMFtoGR << std::endl;
	outSt << "spanGRDenOnGLX " << spanGRDenOnGLX << std::endl;
	outSt << "spanGRDenOnGLY " << spanGRDenOnGLY << std::endl << std::endl;

	outSt << "numSC " <<numSC << std::endl;
	outSt << "numpSCfromGRtoSC " << numpSCfromGRtoSC << std::endl;
	outSt << "numpSCfromGRtoSCP2 " << numpSCfromGRtoSCP2 << std::endl;
	outSt << "numpSCfromSCtoPC " << numpSCfromSCtoPC << std::endl << std::endl;

	outSt << "numPC " << numPC << std::endl;
	outSt << "numpPCfromGRtoPC " << numpPCfromGRtoPC << std::endl;
	outSt  <<  "numpPCfromGRtoPCP2 " << numpPCfromGRtoPCP2 << std::endl;
	outSt  <<  "numpPCfromBCtoPC " << numpPCfromBCtoPC << std::endl;
	outSt  <<  "numpPCfromPCtoBC " << numpPCfromPCtoBC << std::endl;
	outSt  <<  "numpPCfromSCtoPC " << numpPCfromSCtoPC << std::endl;
	outSt  <<  "numpPCfromPCtoNC " << numpPCfromPCtoNC << std::endl << std::endl;

	outSt  <<  "numBC " << numBC << std::endl;
	outSt  <<  "numpBCfromGRtoBC " << numpBCfromGRtoBC << std::endl;
	outSt  <<  "numpBCfromGRtoBCP2 " << numpBCfromGRtoBCP2 << std::endl;
	outSt  <<  "numpBCfromBCtoPC " << numpBCfromBCtoPC << std::endl;
	outSt  <<  "numpBCfromPCtoBC " << numpBCfromPCtoBC << std::endl << std::endl;

	outSt  <<  "numNC " << numNC << std::endl;
	outSt  <<  "numpNCfromPCtoNC " << numpNCfromPCtoNC << std::endl;
	outSt  <<  "numpNCfromNCtoIO " << numpNCfromNCtoIO << std::endl;
	outSt  <<  "numpNCfromMFtoNC " << numpNCfromMFtoNC << std::endl << std::endl;

	outSt  <<  "numIO " << numIO << std::endl;
	outSt  <<  "numpIOfromIOtoPC " << numpIOfromIOtoPC << std::endl;
	outSt  <<  "numpIOfromNCtoIO " << numpIOfromNCtoIO << std::endl;
	outSt  <<  "numpIOInIOIO " << numpIOInIOIO << std::endl;
	outSt  <<  "numpIOOutIOIO " << numpIOOutIOIO << std::endl;
}

ct_uint32_t ConnectivityParams::getGOX()
{
	return goX;
}

ct_uint32_t ConnectivityParams::getGOY()
{
	return goY;
}

ct_uint32_t ConnectivityParams::getGRX()
{
	return grX;
}

ct_uint32_t ConnectivityParams::getGRY()
{
	return grY;
}

ct_uint32_t ConnectivityParams::getGLX()
{
	return glX;
}

ct_uint32_t ConnectivityParams::getGLY()
{
	return glY;
}

ct_uint32_t ConnectivityParams::getNumMF()
{
	return numMF;
}

ct_uint32_t ConnectivityParams::getNumGO()
{
	return numGO;
}

ct_uint32_t ConnectivityParams::getNumGR()
{
	return numGR;
}

ct_uint32_t ConnectivityParams::getNumGL()
{
	return numGL;
}

ct_uint32_t ConnectivityParams::getNumUBC()
{
	return numUBC;
}

ct_uint32_t ConnectivityParams::getNumSC()
{
	return numSC;
}

ct_uint32_t ConnectivityParams::getNumBC()
{
	return numBC;
}

ct_uint32_t ConnectivityParams::getNumPC()
{
	return numPC;
}

ct_uint32_t ConnectivityParams::getNumNC()
{
	return numNC;
}

ct_uint32_t ConnectivityParams::getNumIO()
{
	return numIO;
}

std::map<std::string, ct_uint32_t> ConnectivityParams::getParamCopy()
{
	std::map<std::string, ct_uint32_t> paramCopy;

	for (auto i = paramMap.begin(); i!=paramMap.end(); i++)
	{
		paramCopy[i->first] = i->second;
	}

	paramCopy["maxnumpGOInGOtoGO"]  = maxnumpGOGABAInGOGO;
	paramCopy["maxnumpGOOutGOtoGO"] = maxnumpGOGABAOutGOGO;

	paramCopy["maxnumpGOCoupInGOtoGO"]  = maxnumpGOCoupInGOGO;
	paramCopy["maxnumpGOCoupOutGOtoGO"] = maxnumpGOCoupOutGOGO;

	return paramCopy;
}

