/*
 * cbmstate.cpp
 *
 *  Created on: Dec 5, 2012
 *      Author: consciousness
 */

#include "interfaces/cbmstate.h"

CBMState::CBMState() {}

CBMState::CBMState(ConnectivityParams *cp, unsigned int nZones) : numZones(nZones)
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

	newState(cp, nZones, innetCRSeed, mzoneCRSeed, mzoneARSeed);

	delete[] mzoneCRSeed;
	delete[] mzoneARSeed;
}

CBMState::CBMState(ConnectivityParams *cp, unsigned int nZones,
	std::fstream &sim_file_buf) : numZones(nZones)
{
	innetConState  = new InNetConnectivityState(cp, sim_file_buf);
	innetActState  = new InNetActivityState(cp, sim_file_buf);

	mzoneConStates = new MZoneConnectivityState*[nZones];
	mzoneActStates = new MZoneActivityState*[nZones];

	for (int i = 0; i < nZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(cp, sim_file_buf);
		mzoneActStates[i] = new MZoneActivityState(cp, sim_file_buf);
	}
}

CBMState::CBMState(ConnectivityParams *cp, unsigned int nZones,
	std::string inFile) : numZones(nZones)
{
	std::fstream inStateFileBuffer(inFile.c_str(), std::ios::in | std::ios::binary);
	innetConState  = new InNetConnectivityState(cp, inStateFileBuffer);
	innetActState  = new InNetActivityState(cp, inStateFileBuffer);

	mzoneConStates = new MZoneConnectivityState*[nZones];
	mzoneActStates = new MZoneActivityState*[nZones];

	for (int i = 0; i < nZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(cp, inStateFileBuffer);
		mzoneActStates[i] = new MZoneActivityState(cp, inStateFileBuffer);
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

void CBMState::newState(ConnectivityParams *cp, unsigned int nZones,
	int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed)
{
	innetConState  = new InNetConnectivityState(cp, act_params[msPerTimeStep], innetCRSeed);
	mzoneConStates = new MZoneConnectivityState*[nZones];
	innetActState  = new InNetActivityState(cp);
	mzoneActStates = new MZoneActivityState*[nZones];

	for (int i = 0; i < nZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(cp, mzoneCRSeed[i]);
		mzoneActStates[i] = new MZoneActivityState(cp, mzoneARSeed[i]);
	}
}

void CBMState::readState(ConnectivityParams *cp, std::fstream &infile)
{
	innetConState->readState(cp, infile);
	innetActState->readState(cp, infile);

	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i]->readState(cp, infile);
		mzoneActStates[i]->readState(cp, infile);
	}
}

void CBMState::writeState(ConnectivityParams *cp, std::fstream &outfile)
{
	innetConState->writeState(cp, outfile);
	innetActState->writeState(cp, outfile);
	
	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i]->writeState(cp, outfile);
		mzoneActStates[i]->writeState(cp, outfile);
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

