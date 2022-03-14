/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "params/connectivityparams.h"

using namespace std;

ConnectivityParams::ConnectivityParams(fstream &infile)
{
	//Assumes that file is in the following format:
	//key\tvalue\n
	//key\tvalue\n

//	string line;
	//loop through file and add key/value pair to map
	//** this is done to remove the necessity of order in the original file
//	while(getline(infile,line))
//	{
//		tempMap[line.substr(0,line.find_first_of("\t"))]=atof(line.substr(line.find_first_of("\t"),line.size()).c_str());
//	}

	string key;
	ct_uint32_t val;

	bool hasGOGOCouple;

	//hasGOGOCouple=false;
	//while(true)
	for(int i=0; i<1000; i++)
	{
		infile>>key;

		if(key.compare("connectivityParamEnd")==0)
		{
			infile>>val;
			break;
		}
		/*if(key.compare("GOGOLocalMatrix")==0)
		{
			infile>>key>>val;
			maxnumpGOGABAInGOGO=val;
			infile>>key>>val;
			maxnumpGOGABAOutGOGO=val;

			gogoGABALocalCon=allocate2DArray<float>(maxnumpGOGABAOutGOGO, 3);
			for(int i=0; i<maxnumpGOGABAOutGOGO; i++)
			{
				infile>>gogoGABALocalCon[i][0]>>gogoGABALocalCon[i][1]>>gogoGABALocalCon[i][2];
			}
		}
		else if(key.compare("GOGOCoupLocalMatrix")==0)
		{
			hasGOGOCouple=true;
			infile>>key>>val;
			maxnumpGOCoupInGOGO=val;
			infile>>key>>val;
			maxnumpGOCoupOutGOGO=val;
			gogoCoupLocalCon=allocate2DArray<float>(maxnumpGOCoupOutGOGO, 3);
			for(int i=0; i<maxnumpGOCoupOutGOGO; i++)
			{
				infile>>gogoCoupLocalCon[i][0]>>gogoCoupLocalCon[i][1]>>gogoCoupLocalCon[i][2];
			}
		}*/
		else
		{
			infile>>val;
			paramMap[key]=val;
		}
	}

	/*if(!hasGOGOCouple)
	{

		maxnumpGOCoupInGOGO=1;
		maxnumpGOCoupOutGOGO=1;

		gogoCoupLocalCon=allocate2DArray<float>(maxnumpGOCoupOutGOGO, 3);
		for(int i=0; i<maxnumpGOCoupOutGOGO; i++)
		{
			gogoCoupLocalCon[i][0]=1;
			gogoCoupLocalCon[i][1]=1;
			gogoCoupLocalCon[i][2]=0;
		}
	}
	*/

	//move elements from map to public variables
	
	
	
	ubcX=paramMap["ubcX"];
	ubcY=paramMap["ubcY"];
	numUBC=ubcX*ubcY;

	spanGLtoUBCX=paramMap["spanGLtoUBCX"];
	spanGLtoUBCY=paramMap["spanGLtoUBCY"];
	numpGLtoUBC = (spanGLtoUBCX+1)*(spanGLtoUBCY+1);
	
	spanUBCtoGLX = paramMap["spanUBCtoGLX"];	
	spanUBCtoGLY = paramMap["spanUBCtoGLY"];
	numpUBCtoGL = (spanUBCtoGLX+1)*(spanUBCtoGLY+1);
	
	
	maxnumpGOGABAInGOGO=paramMap["maxnumpGOInGOtoGO"];
	maxnumpGOGABAOutGOGO=paramMap["maxnumpGOOutGOtoGO"];
	maxnumpGOCoupInGOGO=paramMap["maxnumpGOCoupInGOtoGO"];
	maxnumpGOCoupOutGOGO=paramMap["maxnumpGOCoupOutGOtoGO"];
	
	
	//glomerulus
	//glX=1<<paramMap["glXP2"];
	//glY=1<<paramMap["glYP2"];
	glX = paramMap["glX"];
	glY = paramMap["glY"];
	numGL=glX*glY;


	maxnumpGLfromGLtoGR=paramMap["maxnumpGLfromGLtoGR"];
	lownumpGLfromGLtoGR=paramMap["lownumpGLfromGLtoGR"];
	maxnumpGLfromGLtoGO=paramMap["maxnumpGLfromGLtoGO"];
	maxnumpGLfromGOtoGL=paramMap["maxnumpGLfromGOtoGL"];

	//mossy fibers
	numMF=paramMap["numMFP2"];
	mfX=paramMap["mfX"];
	mfY=paramMap["mfY"];
	//numMF=mfX*mfY;

	spanMFtoGLX=paramMap["spanMFtoGLX"];
	spanMFtoGLY=paramMap["spanMFtoGLY"];
	numpMFtoGL = (spanMFtoGLX+1)*(spanMFtoGLY+1);
	
	maxnumpMFfromMFtoGO=20*maxnumpGLfromGLtoGO;
	maxnumpMFfromMFtoGR=20*maxnumpGLfromGLtoGR;

	numpMFfromMFtoNC=1; //fixed for now

	//golgi cells
	//goX=1<<paramMap["goXP2"];
	//goY=1<<paramMap["goYP2"];
	goX=paramMap["goX"];
	goY=paramMap["goY"];
	numGO=goX*goY;
	numConGOGO=paramMap["numConGOGO"];


	spanGOGOsynX=paramMap["spanGOGOsynX"];
	spanGOGOsynY=paramMap["spanGOGOsynY"];
	numpGOGOsyn=(spanGOGOsynX+1)*(spanGOGOsynY+1);
	sigmaGOGOsynML=paramMap["sigmaGOGOsynML"];
	sigmaGOGOsynS=paramMap["sigmaGOGOsynS"];
	maxGOGOsyn=paramMap["maxGOGOsyn"];
	

	spanGOGLX=paramMap["spanGOGLX"];
	spanGOGLY=paramMap["spanGOGLY"];
	numpGOGL=(spanGOGLX+1)*(spanGOGLY+1);


	maxnumpGOfromGRtoGO=paramMap["maxnumpGOfromGRtoGO"];
	maxnumpGOfromGLtoGO=paramMap["maxnumpGOfromGLtoGO"];
	maxnumpGOfromMFtoGO=maxnumpGOfromGLtoGO;
	maxnumpGOfromGOtoGL=paramMap["maxnumpGOfromGOtoGL"];
	maxnumpGOfromGOtoGR=maxnumpGOfromGOtoGL*maxnumpGLfromGLtoGR;

	spanGODecDenOnGLX=paramMap["spanGODecDenOnGLX"];
	spanGODecDenOnGLY=paramMap["spanGODecDenOnGLY"];

	spanGOAscDenOnGRX=paramMap["spanGOAscDenOnGRX"];
	spanGOAscDenOnGRY=paramMap["spanGOAscDenOnGRY"];

	spanGOAxonOnGLX=paramMap["spanGOAxonOnGLX"];
	spanGOAxonOnGLY=paramMap["spanGOAxonOnGLY"];

	spanGOtoGLX=paramMap["spanGOtoGLX"];
	spanGOtoGLY=paramMap["spanGOtoGLY"];
	numpGOtoGL=(spanGOtoGLX+1)*(spanGOtoGLY+1);

	maxnumpGOCoupInGOGO=paramMap["maxnumpGOCoupInGOtoGO"];
	maxnumpGOCoupOutGOGO=paramMap["maxnumpGOCoupOutGOtoGO"];


	//granule cells
	//grX=1<<paramMap["grXP2"];
	//grY=1<<paramMap["grYP2"];
	grX=paramMap["grX"];
	grY=paramMap["grY"];
	numGR=grX*grY;
	
	//numGRP2=paramMap["grXP2"]+paramMap["grYP2"];
	grPFVelInGRXPerTStep=paramMap["grPFVelInGRXPerTStep"];
	grAFDelayInTStep=paramMap["grAFDelayInTStep"];
	maxnumpGRfromGRtoGO=paramMap["maxnumpGRfromGRtoGO"];
	maxnumpGRfromGLtoGR=paramMap["maxnumpGRfromGLtoGR"];
	maxnumpGRfromGOtoGR=paramMap["maxnumpGRfromGOtoGR"];
	maxnumpGRfromMFtoGR=paramMap["maxnumpGRfromMFtoGR"];
//	numPCOutPerPF=paramMap["numPCOutPerPF"];
//	numBCOutPerPF=tempMap["numBCOutPerPF"];
//	numSCOutPerPF=tempMap["numSCOutPerPF"];

	spanGRDenOnGLX=paramMap["spanGRDenOnGLX"];
	spanGRDenOnGLY=paramMap["spanGRDenOnGLY"];


	//TODO: a lot of MZone connectivity is fixed right now,
	//will add flexibility as needed
	numSC=1<<paramMap["numSCP2"];
	numpSCfromGRtoSC=numGR/numSC;
	numpSCfromGRtoSCP2=numGRP2-paramMap["numSCP2"];
	numpSCfromSCtoPC=1;

	numPC=1<<paramMap["numPCP2"];
	numpPCfromGRtoPC=numGR/numPC;
	numpPCfromGRtoPCP2=numGRP2-paramMap["numPCP2"];
	numpPCfromBCtoPC=16;//fixed for now
	numpPCfromPCtoBC=16; //fixed for now
	numpPCfromSCtoPC=numSC/numPC;
	numpPCfromPCtoNC=paramMap["numpPCfromPCtoNC"];

	//numBC=1<<paramMap["numBCP2"];
	numBC=numPC*4;
	//fixed for now since the connectivity pattern depends on this
	//also not that important to have the MZone network very flexible right now
	numpBCfromGRtoBC=numGR/numBC;
	numpBCfromGRtoBCP2=numGRP2-(paramMap["numPCP2"]+2);
	//+2 is connected to the *4 from above
	numpBCfromBCtoPC=4;
	numpBCfromPCtoBC=4;

	numNC=1<<paramMap["numNCP2"];
	numpNCfromPCtoNC=numPC/numNC*numpPCfromPCtoNC;
	numpNCfromNCtoIO=1<<paramMap["numIOP2"]; //all to all connection
	numpNCfromMFtoNC=numMF/numNC;

	numIO=1<<paramMap["numIOP2"];
	numpIOfromIOtoPC=numPC/numIO;
	numpIOfromNCtoIO=numNC; //all to all connection
	numpIOInIOIO=numIO-1;
	numpIOOutIOIO=numIO-1;
}

ConnectivityParams::~ConnectivityParams()
{
	//delete2DArray<float>(gogoGABALocalCon);
	//delete2DArray<float>(gogoCoupLocalCon);
}

void ConnectivityParams::writeParams(fstream &outfile)
{
	map<string, ct_uint32_t>::iterator i;

	for(i=paramMap.begin(); i!=paramMap.end(); i++)
	{
		outfile<<i->first<<" "<<i->second<<endl;
	}

	outfile<<"GOGOLocalMatrix"<<endl;
	outfile<<"maxnumpGOInGOtoGO "<<maxnumpGOGABAInGOGO<<endl;
	outfile<<"maxnumpGOOutGOtoGO "<<maxnumpGOGABAOutGOGO<<endl;
	for(int i=0; i<maxnumpGOGABAOutGOGO; i++)
	{
		outfile<<gogoGABALocalCon[i][0]<<" "<<gogoGABALocalCon[i][1]<<" "<<gogoGABALocalCon[i][2]<<endl;
	}

	outfile<<"GOGOCoupLocalMatrix"<<endl;
	outfile<<"maxnumpGOCoupInGOtoGO "<<maxnumpGOCoupInGOGO<<endl;
	outfile<<"maxnumpGOCoupOutGOtoGO "<<maxnumpGOCoupOutGOGO<<endl;
	for(int i=0; i<maxnumpGOCoupOutGOGO; i++)
	{
		outfile<<gogoCoupLocalCon[i][0]<<" "<<gogoCoupLocalCon[i][1]<<" "<<gogoCoupLocalCon[i][2]<<endl;
	}

	outfile<<"connectivityParamEnd 1"<<endl;
}

void ConnectivityParams::showParams(ostream &outSt)
{
	outSt<<"glX "<<glX<<endl;
	outSt<<"glY "<<glY<<endl;
	outSt<<"numGL "<<numGL<<endl;
	outSt<<"maxnumpGLfromGLtoGR "<<maxnumpGLfromGLtoGR<<endl;
	outSt<<"lownumpGLfromGLtoGR "<<lownumpGLfromGLtoGR<<endl;
	outSt<<"maxnumpGLfromGLtoGO "<<maxnumpGLfromGLtoGO<<endl;
	outSt<<"maxnumpGLfromGOtoGL "<<maxnumpGLfromGOtoGL<<endl<<endl;

	outSt<<"numMF "<<numMF<<endl;
	outSt<<"maxnumpMFfromMFtoGO "<<maxnumpMFfromMFtoGO<<endl;
	outSt<<"maxnumpMFfromMFtoGR "<<maxnumpMFfromMFtoGR<<endl;
	outSt<<"numpMFfromMFtoNC "<<numpMFfromMFtoNC<<endl<<endl;

	outSt<<"goX "<<goX<<endl;
	outSt<<"goY "<<goY<<endl;
	outSt<<"numGO "<<numGO<<endl;
	outSt<<"maxnumpGOfromGRtoGO "<<maxnumpGOfromGRtoGO<<endl;
	outSt<<"maxnumpGOfromGLtoGO "<<maxnumpGOfromGLtoGO<<endl;
	outSt<<"maxnumpGOfromMFtoGO "<<maxnumpGOfromMFtoGO<<endl;
	outSt<<"maxNumpGOfromGOtoGL "<<maxnumpGOfromGOtoGL<<endl;
	outSt<<"maxnumpGOfromGOtoGR "<<maxnumpGOfromGOtoGR<<endl;
	outSt<<"spanGODecDenOnGLX "<<spanGODecDenOnGLX<<endl;
	outSt<<"spanGODecDenOnGLY "<<spanGODecDenOnGLY<<endl;
	outSt<<"spanGOAscDenOnGRX "<<spanGOAscDenOnGRX<<endl;
	outSt<<"spanGOAscDenOnGRY "<<spanGOAscDenOnGRY<<endl;
	outSt<<"spanGOAxonOnGLX "<<spanGOAxonOnGLX<<endl;
	outSt<<"spanGOAxonOnGLY "<<spanGOAxonOnGLY<<endl;
	outSt<<"maxnumpGOInGOGO "<<maxnumpGOGABAInGOGO<<endl;
	outSt<<"maxnumpGOOutGOGO "<<maxnumpGOGABAOutGOGO<<endl;
	outSt<<"gogoLocalMatrix:"<<endl;
	for(int i=0; i<maxnumpGOGABAOutGOGO; i++)
	{
		outSt<<gogoGABALocalCon[i][0]<<" "<<gogoGABALocalCon[i][1]<<" "<<gogoGABALocalCon[i][2]<<endl;
	}

	outSt<<"maxnumpGOCoupInGOGO "<<maxnumpGOCoupInGOGO<<endl;
	outSt<<"maxnumpGOCoupOutGOGO "<<maxnumpGOCoupOutGOGO<<endl;
	outSt<<"gogoCoupLocalMatrix:"<<endl;
	for(int i=0; i<maxnumpGOCoupOutGOGO; i++)
	{
		outSt<<gogoCoupLocalCon[i][0]<<" "<<gogoCoupLocalCon[i][1]<<" "<<gogoCoupLocalCon[i][2]<<endl;
	}

	outSt<<"grX "<<grX<<endl;
	outSt<<"grY "<<grY<<endl;
	outSt<<"numGR "<<numGR<<endl;
	outSt<<"grPFVelInGRXPerTStep "<<grPFVelInGRXPerTStep<<endl;
	outSt<<"grAFDelayInTStep "<<grAFDelayInTStep<<endl;
	outSt<<"maxnumpGRfromGRtoGO "<<maxnumpGRfromGRtoGO<<endl;
	outSt<<"maxnumpGRfromGLtoGR "<<maxnumpGRfromGLtoGR<<endl;
	outSt<<"maxnumpGRfromGOtoGR "<<maxnumpGRfromGOtoGR<<endl;
	outSt<<"maxnumpGRfromMFtoGR "<<maxnumpGRfromMFtoGR<<endl;
	outSt<<"spanGRDenOnGLX "<<spanGRDenOnGLX<<endl;
	outSt<<"spanGRDenOnGLY "<<spanGRDenOnGLY<<endl<<endl;

	outSt<<"numSC "<<numSC<<endl;
	outSt<<"numpSCfromGRtoSC "<<numpSCfromGRtoSC<<endl;
	outSt<<"numpSCfromGRtoSCP2 "<<numpSCfromGRtoSCP2<<endl;
	outSt<<"numpSCfromSCtoPC "<<numpSCfromSCtoPC<<endl<<endl;

	outSt<<"numPC "<<numPC<<endl;
	outSt<<"numpPCfromGRtoPC "<<numpPCfromGRtoPC<<endl;
	outSt<<"numpPCfromGRtoPCP2 "<<numpPCfromGRtoPCP2<<endl;
	outSt<<"numpPCfromBCtoPC "<<numpPCfromBCtoPC<<endl;
	outSt<<"numpPCfromPCtoBC "<<numpPCfromPCtoBC<<endl;
	outSt<<"numpPCfromSCtoPC "<<numpPCfromSCtoPC<<endl;
	outSt<<"numpPCfromPCtoNC "<<numpPCfromPCtoNC<<endl<<endl;

	outSt<<"numBC "<<numBC<<endl;
	outSt<<"numpBCfromGRtoBC "<<numpBCfromGRtoBC<<endl;
	outSt<<"numpBCfromGRtoBCP2 "<<numpBCfromGRtoBCP2<<endl;
	outSt<<"numpBCfromBCtoPC "<<numpBCfromBCtoPC<<endl;
	outSt<<"numpBCfromPCtoBC "<<numpBCfromPCtoBC<<endl<<endl;

	outSt<<"numNC "<<numNC<<endl;
	outSt<<"numpNCfromPCtoNC "<<numpNCfromPCtoNC<<endl;
	outSt<<"numpNCfromNCtoIO "<<numpNCfromNCtoIO<<endl;
	outSt<<"numpNCfromMFtoNC "<<numpNCfromMFtoNC<<endl<<endl;

	outSt<<"numIO "<<numIO<<endl;
	outSt<<"numpIOfromIOtoPC "<<numpIOfromIOtoPC<<endl;
	outSt<<"numpIOfromNCtoIO "<<numpIOfromNCtoIO<<endl;
	outSt<<"numpIOInIOIO "<<numpIOInIOIO<<endl;
	outSt<<"numpIOOutIOIO "<<numpIOOutIOIO<<endl;
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

map<string, ct_uint32_t> ConnectivityParams::getParamCopy()
{
	map<string, ct_uint32_t> paramCopy;

	map<string, ct_uint32_t>::iterator i;

	for(i=paramMap.begin(); i!=paramMap.end(); i++)
	{
		paramCopy[i->first]=i->second;
	}

	paramCopy["maxnumpGOInGOtoGO"]=maxnumpGOGABAInGOGO;
	paramCopy["maxnumpGOOutGOtoGO"]=maxnumpGOGABAOutGOGO;

	paramCopy["maxnumpGOCoupInGOtoGO"]=maxnumpGOCoupInGOGO;
	paramCopy["maxnumpGOCoupOutGOtoGO"]=maxnumpGOCoupOutGOGO;

	return paramCopy;
}

