/*
 * cbmstate.cpp
 *
 *  Created on: Dec 5, 2012
 *      Author: consciousness
 */

#include "interfaces/cbmstate.h"

CBMState::CBMState(std::fstream &infile)
{
	std::cout << "State constructor for a single input file" << std::endl;	
	infile >> numZones;

	conParams = new ConnectivityParams(infile);
	actParams = new ActivityParams(infile);

	infile.seekg(1, std::ios::cur);
	innetConState = new InNetConnectivityState(conParams, infile);
	mzoneConStates = new MZoneConnectivityState*[numZones];

	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(conParams, infile);
	}

	innetActState = new InNetActivityState(conParams, infile);
	mzoneActStates = new MZoneActivityState*[numZones];

	for (int i = 0; i < numZones; i++)
	{
		mzoneActStates[i] = new MZoneActivityState(conParams, actParams, infile);
	}
}

CBMState::CBMState(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
		int goRecipParam, int simNum)
{
	// NOTE: again, should separate printing from the constructors
	std::cout << "State constructor using activity and connectivity files" << std::endl;

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
	std::cout << "Deleting activity parameters..." << std::endl;
	delete actParams;	
	std::cout << "Successfully deleted activity parameters." << std::endl;	

	std::cout << "Deleting connectivity parameters..." << std::endl;
	delete conParams;
	std::cout << "Successfully deleted connectivity parameters." << std::endl;

	std::cout << "Deleting input network connectivity state..." << std::endl;
	delete innetConState;
	std::cout << "Successfully deleted input network connectivity state." << std::endl;

	std::cout << "Deleting Mzone connectivity states..." << std::endl:	
	for (int i = 0; i < numZones; i++)
	{
		delete mzoneConStates[i];
	}
	
	delete[] mzoneConStates;
	std::cout << "Successfully deleted Mzone connectivity states." << std::endl;
	
	std::cout << "Deleting input network activity states..." << std::endl;
	delete innetActState;
	std::cout << "Successfully deleted input network activity states." << std::endl;

	std::cout << "Deleting Mzone activity states..." << std::endl;	
	for(int i=0; i<numZones; i++)
	{
		delete mzoneActStates[i];
	}
	delete[] mzoneActStates;
	std::cout << "Successfully deleted Mzone activity states." << std::endl;
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
	
	std::cout << "Writing "	<< numZones << " Mzone connectivity states to files..." << std::endl;
	for(int i=0; i<numZones; i++)
	{
		mzoneConStates[i]->writeState(outfile);
	}
	std::cout << "Done writing Mzone connectivity states to files." << std::endl;

	std::cout << "Writing input network activity state to file..." << std::endl;
	innetActState->writeState(outfile);
	std::cout << "Done writing input network activity state to file." << std::endl;

	std::cout << "Writing Mzone activity states to files..." << std::endl;
	for (int i = 0; i < numZones; i++)
	{
		mzoneActStates[i]->writeState(outfile);
	}
	std::cout << "Done writing Mzone activity states to files..." << std::endl;
}

bool CBMState::equivalent(CBMState &compState)
{
	if (numZones != compState.getNumZones())
	{
		return false;
	} 
	else
	{
		return innetConState->equivalent(*(compState.getInnetConStateInternal()));
	}
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

IInNetConState* CBMState::getInnetConState()
{
	return (IInNetConState *)innetConState;
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

void CBMState::newState(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
			int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed, int goRecipParam, int simNum)
{
	numZones = nZones;

	conParams = new ConnectivityParams(conPFile);
	actParams = new ActivityParams(actPFile);

	std::cout << "parameters loaded" << std::endl;

	std::cout << "Constructing input network connectivity states..." << std::endl;	
	innetConState = new InNetConnectivityState(conParams, actParams->msPerTimeStep, innetCRSeed, goRecipParam, simNum);
	std::cout << "Input network connectivity states constructed." << std::endl;

	std::cout << "Constructing MZone connectivity states..." << std::endl;
	mzoneConStates = new MZoneConnectivityState*[numZones];
	
	for (int i = 0; i < numZones; i++)
	{
		mzoneConStates[i] = new MZoneConnectivityState(conParams, mzoneCRSeed[i]);
	}
	std::cout << "Mzone connectivity states constructed." << std::endl;

	std::cout << "Constructing input network activity states..." << std::endl;
	innetActState = new InNetActivityState(conParams, actParams);
	std::cout << "Innet activity states constructed" << std::endl;
	
	std::cout << "Constructing MZone activity states..." << std::endl;
	mzoneActStates = new MZoneActivityState*[numZones];
	
	for (int i = 0; i < numZones; i++)
	{
		mzoneActStates[i] = new MZoneActivityState(conParams, actParams, mzoneARSeed[i]);
	}
	std::cout << "Mzone activity states constructed" << std::endl;
}

