/*
 * cbmstatex2grgodecouple.cpp
 *
 *  Created on: Apr 30, 2013
 *      Author: consciousness
 */


#include "interfaces/cbmstatex2grgodecouple.h"

using namespace std;

CBMStateX2GRGODecouple::CBMStateX2GRGODecouple(fstream &actPFile, fstream &actPFile1, fstream &conPFile)
{
	CRandomSFMT0 randGen(time(0));

	numInnets=2;
	numZones=2;

	conParams=new ConnectivityParams(conPFile);
	actParamsList[0]=new ActivityParams(actPFile);
	actParamsList[1]=new ActivityParams(actPFile1);

	//innetConStates[0]=new InNetConnectivityState(conParams, actParamsList[0]->msPerTimeStep,
	//		randGen.IRandom(0, INT_MAX));
	//innetConStates[1]=new InNetConnectivityState(*innetConStates[0]);

	mzoneConStates[0]=new MZoneConnectivityState(conParams, randGen.IRandom(0, INT_MAX));
	mzoneConStates[1]=new MZoneConnectivityState(*mzoneConStates[0]);

	innetActStates[0]=new InNetActivityState(conParams, actParamsList[0]);
	innetActStates[1]=new InNetActivityState(*innetActStates[0]);

	mzoneActStates[0]=new MZoneActivityState(conParams, actParamsList[0], randGen.IRandom(0, INT_MAX));
	mzoneActStates[1]=new MZoneActivityState(*mzoneActStates[0]);
}

CBMStateX2GRGODecouple::~CBMStateX2GRGODecouple()
{
	delete conParams;

	for(int i=0; i<numInnets; i++)
	{
		delete innetConStates[i];
		delete innetActStates[i];
	}

	for(int i=0; i<numZones; i++)
	{
		delete actParamsList[i];
		delete mzoneActStates[i];
		delete mzoneConStates[i];
	}
}

ct_uint32_t CBMStateX2GRGODecouple::getNumInnets()
{
	return numInnets;
}

ct_uint32_t CBMStateX2GRGODecouple::getNumZones()
{
	return numZones;
}

IConnectivityParams* CBMStateX2GRGODecouple::getConnectivityParams()
{
	return (IConnectivityParams *)conParams;
}

IActivityParams* CBMStateX2GRGODecouple::getActivityParams(unsigned int paramN)
{
	return (IActivityParams *)actParamsList[paramN];
}

IInNetConState* CBMStateX2GRGODecouple::getInnetConState(unsigned int stateN)
{
	return (IInNetConState *)innetConStates[stateN];
}

ActivityParams* CBMStateX2GRGODecouple::getActParamsInternal(unsigned int paramN)
{
	return actParamsList[paramN];
}

ConnectivityParams* CBMStateX2GRGODecouple::getConParamsInternal()
{
	return conParams;
}

InNetActivityState* CBMStateX2GRGODecouple::getInnetActStateInternal(unsigned int stateN)
{
	return innetActStates[stateN];
}

MZoneActivityState* CBMStateX2GRGODecouple::getMZoneActStateInternal(unsigned int zoneN)
{
	return mzoneActStates[zoneN];
}

InNetConnectivityState* CBMStateX2GRGODecouple::getInnetConStateInternal(unsigned int stateN)
{
	return innetConStates[stateN];
}

MZoneConnectivityState* CBMStateX2GRGODecouple::getMZoneConStateInternal(unsigned int zoneN)
{
	return mzoneConStates[zoneN];
}
