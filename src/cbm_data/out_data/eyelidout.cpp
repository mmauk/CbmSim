/*
 * eyelidout.cpp
 *
 *  Created on: Aug 28, 2012
 *      Author: consciousness
 */

#include "eyelidout.h"

using namespace std;

EyelidOut::EyelidOut(unsigned int numTrials, unsigned int msPerTimeStep, unsigned int numTimeStepSmooth,
		unsigned int msPerTrial)
{
	nTrials=numTrials;
	msPerTS=msPerTimeStep;
	numTSSmooth=numTimeStepSmooth;
	this->msPerTrial=msPerTrial;
	this->tsPerTrial=msPerTrial/msPerTS;

	curTrialN=0;
	curTS=0;

	allocateMemory();
}

EyelidOut::EyelidOut(fstream &infile)
{
	infile>>nTrials;
	infile>>msPerTS;
	infile>>numTSSmooth;
	infile>>msPerTrial;
	infile>>tsPerTrial;
	infile>>curTrialN;
	infile>>curTS;

	allocateMemory();

	infile.seekg(1, ios::cur);

	dataIO(true, infile);
}

EyelidOut::~EyelidOut()
{
	delete2DArray<float>(rawData);
}

void EyelidOut::writeData(fstream &outfile)
{
	outfile<<nTrials<<" ";
	outfile<<msPerTS<<" ";
	outfile<<numTSSmooth<<" ";
	outfile<<msPerTrial<<" ";
	outfile<<tsPerTrial<<" ";
	outfile<<curTrialN<<" ";
	outfile<<curTS<<endl;

	dataIO(false, outfile);
}

bool EyelidOut::updateData(float data)
{
	float eyelidAvg;

	if(curTrialN>=nTrials)
	{
		return false;
	}

	if(curTS<numTSSmooth-1)
	{
		rawData[curTrialN][curTS]=data;
	}
	else
	{
		eyelidAvg=0;
		for(int j=-((int)numTSSmooth-1); j<0; j++)
		{
			eyelidAvg+=rawData[curTrialN][curTS+j];
		}
		eyelidAvg+=data;
		eyelidAvg=eyelidAvg/numTSSmooth;

		rawData[curTrialN][curTS]=eyelidAvg;
	}

	curTS++;
	if(curTS>=tsPerTrial)
	{
		curTrialN++;
		curTS=0;
	}

	return true;
}

ct_uint32_t EyelidOut::getNumTrials()
{
	return curTrialN;
}

ct_uint32_t EyelidOut::getNumPointsPerTrial()
{
	return tsPerTrial;
}

ct_uint32_t EyelidOut::getMSPerPoint()
{
	return msPerTS;
}

vector<float> EyelidOut::getEyelidData(unsigned int trialN)
{
	vector<float> data;

	data.resize(tsPerTrial);

	for(int i=0; i<tsPerTrial; i++)
	{
		data[i]=rawData[trialN][i];
	}

	return data;
}

vector<vector<float> > EyelidOut::getEyelidData(vector<unsigned int> trialNs)
{
	vector<vector<float> > data;

	for(int i=0; i<trialNs.size(); i++)
	{
		data.push_back(getEyelidData(trialNs[i]));
	}

	return data;
}

vector<vector<float> > EyelidOut::getEyelidData(unsigned int startTrialN, unsigned int endTrialN)
{
	vector<unsigned int>trialNs;

	for(unsigned int i=startTrialN; i<endTrialN; i++)
	{
		trialNs.push_back(i);
	}

	return getEyelidData(trialNs);
}

void EyelidOut::allocateMemory()
{
	rawData=allocate2DArray<float>(nTrials, tsPerTrial);
}

void EyelidOut::dataIO(bool read, fstream &file)
{
	rawBytesRW((char *)rawData[0], nTrials*tsPerTrial*sizeof(float), read, file);
}
