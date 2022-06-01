/*
 * cbmstate.cpp
 *
 *  Created on: Dec 5, 2012
 *      Author: consciousness
 */

#include "interfaces/cbmstate.h"

CBMState::CBMState(std::fstream &infile)
{
	infile >> numZones;

	conParams = new ConnectivityParams(infile);
	actParams = new ActivityParams(infile);

	infile.seekg(1, std::ios::cur);

	innetConState = new InNetConnectivityState(conParams, infile);
	mzoneConStates = new MZoneConnectivityState*[numZones];

	innetActState = new InNetActivityState(conParams, infile);
	mzoneActStates = new MZoneActivityState*[numZones];

	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(conParams, infile);
		mzoneActStates[i] = new MZoneActivityState(conParams, actParams, infile);
	}
}

CBMState::CBMState(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
		int goRecipParam, int simNum)
{
	int innetCRSeed;
	int *mzoneCRSeed = new int[nZones];
	int *mzoneARSeed = new int[nZones];

	CRandomSFMT0 randGen(time(0));
	innetCRSeed = randGen.IRandom(0, INT_MAX);

	for (int i = 0; i < nZones; i++)
	{
		mzoneCRSeed[i] = randGen.IRandom(0, INT_MAX);
		mzoneARSeed[i] = randGen.IRandom(0, INT_MAX);
	}

	newState(actPFile, conPFile, nZones, innetCRSeed, mzoneCRSeed, mzoneARSeed, goRecipParam, simNum);

	delete[] mzoneCRSeed;
	delete[] mzoneARSeed;
}

CBMState::~CBMState()
{
	delete actParams;	
	delete conParams;

	delete innetConState;
	delete innetActState;

	for (int i = 0; i < numZones; i++) 
	{
		delete mzoneConStates[i];
		delete mzoneActStates[i];
	}	

	delete[] mzoneConStates;
	delete[] mzoneActStates;
}

void CBMState::newState(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
			int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed, int goRecipParam, int simNum)
{
	numZones = nZones;
	
	conParams = new ConnectivityParams(conPFile);
	actParams = new ActivityParams(actPFile);

	innetConState  = new InNetConnectivityState(conParams, actParams->msPerTimeStep,
			innetCRSeed, goRecipParam, simNum);
	mzoneConStates = new MZoneConnectivityState*[numZones];
	innetActState  = new InNetActivityState(conParams, actParams);
	mzoneActStates = new MZoneActivityState*[numZones];

	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(conParams, mzoneCRSeed[i]);
		mzoneActStates[i] = new MZoneActivityState(conParams, actParams, mzoneARSeed[i]);
	}
}

void CBMState::writeState(std::fstream &outfile)
{
	// Debatable whether to write std::endl to file. Yes it flushes the output
	// after sending a newline, but doing this too often can lead to poor disk
	// access performance. Whatever that means.
	outfile << numZones << std::endl;

	conParams->writeParams(outfile);
	actParams->writeParams(outfile);
	
	innetConState->writeState(outfile);
	innetActState->writeState(outfile);
	
	for (int i=0; i < numZones; i++)
	{
		mzoneConStates[i]->writeState(outfile);
		mzoneActStates[i]->writeState(outfile);
	}
}

bool CBMState::operator==(CBMState &compState)
{
	return numZones == compState.getNumZones() ? true : 
		innetConState == compState.getInnetConStateInternal();
}

bool CBMState::operator!=(CBMState & compState)
{
	return !(*this == compState);
}

ct_uint32_t CBMState::getNumZones()
{
	return numZones;
}

IConnectivityParams* CBMState::getConnectivityParams()
{
	return (IConnectivityParams *)conParams;
}

IActivityParams* CBMState::getActivityParams()
{
	return (IActivityParams *)actParams;
}

IMZoneActState* CBMState::getMZoneActState(unsigned int zoneN)
{
	return (IMZoneActState *)mzoneActStates[zoneN];
}

ActivityParams* CBMState::getActParamsInternal()
{
	return actParams;
}

ConnectivityParams* CBMState::getConParamsInternal()
{
	return conParams;
}

InNetActivityState* CBMState::getInnetActStateInternal()
{
	return innetActState;
}

MZoneActivityState* CBMState::getMZoneActStateInternal(unsigned int zoneN)
{
	return mzoneActStates[zoneN];
}

InNetConnectivityState* CBMState::getInnetConStateInternal()
{
	return innetConState;
}

MZoneConnectivityState* CBMState::getMZoneConStateInternal(unsigned int zoneN)
{
	return mzoneConStates[zoneN];
}

