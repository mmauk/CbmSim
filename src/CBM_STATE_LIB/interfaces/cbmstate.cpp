/*
 * cbmstate.cpp
 *
 *  Created on: Dec 5, 2012
 *      Author: consciousness
 */

#include "interfaces/cbmstate.h"

CBMState::CBMState() {}

CBMState::CBMState(ActivityParams &ap, unsigned int nZones)
{
	int *mzoneCRSeed = new int[nZones];
	int *mzoneARSeed = new int[nZones];

	CRandomSFMT0 randGen(time(0));
	int innetCRSeed = randGen.IRandom(0, INT_MAX);

	for (int i = 0; i < nZones; i++)
	{
		mzoneCRSeed[i] = randGen.IRandom(0, INT_MAX);
		mzoneARSeed[i] = randGen.IRandom(0, INT_MAX);
	}

	newState(ap, nZones, innetCRSeed, mzoneCRSeed, mzoneARSeed);

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

void CBMState::newState(ActivityParams &ap,
	unsigned int nZones, int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed)
{
	innetConState  = new InNetConnectivityState(ap.msPerTimeStep, innetCRSeed);
	mzoneConStates = new MZoneConnectivityState*[numZones];
	innetActState  = new InNetActivityState(ap);
	mzoneActStates = new MZoneActivityState*[numZones];

	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(mzoneCRSeed[i]);
		mzoneActStates[i] = new MZoneActivityState(ap, mzoneARSeed[i]);
	}
}

void CBMState::writeState(ActivityParams &ap, std::fstream &outfile)
{
	// Debatable whether to write std::endl to file. Yes it flushes the output
	// after sending a newline, but doing this too often can lead to poor disk
	// access performance. Whatever that means.
	//outfile << numZones << std::endl;

	// have to comment these out for now: no methods for cp
	//cp.writeParams(outfile);
	//ap->writeParams(outfile);
	
	innetConState->writeState(outfile);
	innetActState->writeState(outfile);
	
	for (int i=0; i < numZones; i++)
	{
		mzoneConStates[i]->writeState(outfile);
		mzoneActStates[i]->writeState(ap, outfile);
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

