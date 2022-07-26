/*
 * cbmstate.cpp
 *
 *  Created on: Dec 5, 2012
 *      Author: consciousness
 */

#include "interfaces/cbmstate.h"

CBMState::CBMState() {}

CBMState::CBMState(ConnectivityParams *cp, ActivityParams *ap, 
	unsigned int nZones, std::string inFile) : numZones(nZones)
{
	std::fstream inStateFileBuffer(inFile.c_str(), std::ios::in | std::ios::binary);
	std::cout << "[INFO]: Allocating and initializing innet connectivity state from file..." << std::endl;
	innetConState  = new InNetConnectivityState(cp, inStateFileBuffer);
	std::cout << "[INFO]: Finished allocating and initializing innet connectivity state from file." << std::endl;
	std::cout << "[INFO]: Allocating and initializing innet activity state from file..." << std::endl;
	innetActState  = new InNetActivityState(cp, inStateFileBuffer);
	std::cout << "[INFO]: Finished allocating and initializing innet activity state from file." << std::endl;

	mzoneConStates = new MZoneConnectivityState*[nZones];
	mzoneActStates = new MZoneActivityState*[nZones];

	for (int i = 0; i < nZones; i++)
	{
		std::cout << "[INFO]: Allocating and initializing mzone "
				  << i << " connectivity state from file..." << std::endl;
		mzoneConStates[i] = new MZoneConnectivityState(cp, inStateFileBuffer);
		std::cout << "[INFO]: Finished allocating and initializing mzone "
				  << i << " connectivity state from file." << std::endl;
		std::cout << "[INFO]: Allocating and initializing mzone "
				  << i << " activity state from file..." << std::endl;
		mzoneActStates[i] = new MZoneActivityState(cp, ap, inStateFileBuffer);
		std::cout << "[INFO]: Finished allocating and initializing mzone "
				  << i << " activity state from file." << std::endl;
	}
	inStateFileBuffer.close();
}

CBMState::CBMState(ConnectivityParams *cp, ActivityParams *ap,
	unsigned int nZones) : numZones(nZones)
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

	newState(cp, ap, nZones, innetCRSeed, mzoneCRSeed, mzoneARSeed);

	delete[] mzoneCRSeed;
	delete[] mzoneARSeed;
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

void CBMState::newState(ConnectivityParams *cp, ActivityParams *ap,
	unsigned int nZones, int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed)
{
	innetConState  = new InNetConnectivityState(cp, ap->msPerTimeStep, innetCRSeed);
	mzoneConStates = new MZoneConnectivityState*[nZones];
	innetActState  = new InNetActivityState(cp, ap);
	mzoneActStates = new MZoneActivityState*[nZones];

	for (int i = 0; i < nZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(cp, mzoneCRSeed[i]);
		mzoneActStates[i] = new MZoneActivityState(cp, ap, mzoneARSeed[i]);
	}
}

void CBMState::readState(ConnectivityParams *cp, ActivityParams *ap, std::fstream &infile)
{
	innetConState->readState(cp, infile);
	innetActState->readState(cp, infile);

	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i]->readState(cp, infile);
		mzoneActStates[i]->readState(cp, ap, infile);
	}
}

void CBMState::writeState(ConnectivityParams *cp, ActivityParams *ap, std::fstream &outfile)
{
	std::cout << "[INFO]: Writing innet connectivity state to file..." << std::endl;   
	innetConState->writeState(cp, outfile);
	std::cout << "[INFO]: Finished writing innet connectivity state to file." << std::endl;
	std::cout << "[INFO]: Writing innet activity state to file..." << std::endl;   
	innetActState->writeState(cp, outfile);
	std::cout << "[INFO]: Finished writing innet activity state to file." << std::endl;
	
	for (int i = 0; i < numZones; i++)
	{
		std::cout << "[INFO]: Writing mzone "
				  << i << " connectivity state to file..." << std::endl;   
		mzoneConStates[i]->writeState(cp, outfile);
		std::cout << "[INFO]: Finished writing mzone "
				  << i << " connectivity state to file..." << std::endl;   
		std::cout << "[INFO]: Writing mzone "
				  << i << " activity state to file..." << std::endl;   
		mzoneActStates[i]->writeState(cp, ap, outfile);
		std::cout << "[INFO]: Finished writing mzone "
				  << i << " activity state to file..." << std::endl;   
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

