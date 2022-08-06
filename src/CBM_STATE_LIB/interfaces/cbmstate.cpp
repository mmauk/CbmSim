/*
 * cbmstate.cpp
 *
 *  Created on: Dec 5, 2012
 *      Author: consciousness
 */

#include "interfaces/cbmstate.h"

CBMState::CBMState() {}

CBMState::CBMState(unsigned int nZones) : numZones(nZones)
{
	CRandomSFMT randGen(time(0));

	int *mzoneCRSeed = new int[nZones];
	int *mzoneARSeed = new int[nZones];

	int innetCRSeed = randGen.IRandom(0, INT_MAX);
	for (int i = 0; i < nZones; i++)
	{
		mzoneCRSeed[i] = randGen.IRandom(0, INT_MAX);
		mzoneARSeed[i] = randGen.IRandom(0, INT_MAX);
	}

	newState(nZones, innetCRSeed, mzoneCRSeed, mzoneARSeed);

	delete[] mzoneCRSeed;
	delete[] mzoneARSeed;
}

CBMState::CBMState(unsigned int nZones, std::fstream &sim_file_buf) : numZones(nZones)
{
	innetConState  = new InNetConnectivityState(sim_file_buf);
	innetActState  = new InNetActivityState(sim_file_buf);

	mzoneConStates = new MZoneConnectivityState*[nZones];
	mzoneActStates = new MZoneActivityState*[nZones];

	for (int i = 0; i < nZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(sim_file_buf);
		mzoneActStates[i] = new MZoneActivityState(sim_file_buf);
	}
}

CBMState::CBMState(unsigned int nZones,
	std::string inFile) : numZones(nZones)
{
	std::fstream inStateFileBuffer(inFile.c_str(), std::ios::in | std::ios::binary);
	innetConState  = new InNetConnectivityState(inStateFileBuffer);
	innetActState  = new InNetActivityState(inStateFileBuffer);

	mzoneConStates = new MZoneConnectivityState*[nZones];
	mzoneActStates = new MZoneActivityState*[nZones];

	for (int i = 0; i < nZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(inStateFileBuffer);
		mzoneActStates[i] = new MZoneActivityState(inStateFileBuffer);
	}
	inStateFileBuffer.close();
}

CBMState::~CBMState()
{
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

void CBMState::newState(unsigned int nZones, int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed)
{
	innetConState  = new InNetConnectivityState(msPerTimeStep, innetCRSeed);
	mzoneConStates = new MZoneConnectivityState*[nZones];
	innetActState  = new InNetActivityState();
	mzoneActStates = new MZoneActivityState*[nZones];

	for (int i = 0; i < nZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(mzoneCRSeed[i]);
		mzoneActStates[i] = new MZoneActivityState(mzoneARSeed[i]);
	}
}

void CBMState::readState(std::fstream &infile)
{
	innetConState->readState(infile);
	innetActState->readState(infile);

	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i]->readState(infile);
		mzoneActStates[i]->readState(infile);
	}
}

void CBMState::writeState(std::fstream &outfile)
{
	innetConState->writeState(outfile);
	innetActState->writeState(outfile);
	
	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i]->writeState(outfile);
		mzoneActStates[i]->writeState(outfile);
	}
}

ct_uint32_t CBMState::getNumZones()
{
	return numZones;
}

IMZoneActState* CBMState::getMZoneActState(unsigned int zoneN)
{
	return (IMZoneActState *)mzoneActStates[zoneN];
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

