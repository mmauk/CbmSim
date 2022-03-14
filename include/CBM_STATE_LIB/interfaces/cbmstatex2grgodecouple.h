/*
 * cbmstatex2grgodecouple.h
 *
 *  Created on: Apr 30, 2013
 *      Author: consciousness
 */

#ifndef CBMSTATEX2GRGODECOUPLE_H_
#define CBMSTATEX2GRGODECOUPLE_H_

#include <fstream>
#include <iostream>
#include <time.h>
#include <limits.h>

#include <stdDefinitions/pstdint.h>
#include <memoryMgmt/dynamic2darray.h>
#include <randGenerators/sfmt.h>
#include "params/activityparams.h"
#include "params/connectivityparams.h"

#include "state/innetconnectivitystate.h"
#include "state/mzoneconnectivitystate.h"
#include "state/innetactivitystate.h"
#include "state/mzoneactivitystate.h"

#include "iconnectivityparams.h"
#include "iactivityparams.h"
#include "iinnetconstate.h"

class CBMStateX2GRGODecouple
{
public:
	CBMStateX2GRGODecouple(std::fstream &actPFile, std::fstream &actPFile1, std::fstream &conPFile);

	virtual ~CBMStateX2GRGODecouple();

	ct_uint32_t getNumInnets();
	ct_uint32_t getNumZones();

	IConnectivityParams* getConnectivityParams();
	IActivityParams* getActivityParams(unsigned int paramN);

	IInNetConState* getInnetConState(unsigned int stateN);

	ActivityParams* getActParamsInternal(unsigned int paramN);
	ConnectivityParams* getConParamsInternal();

	InNetActivityState* getInnetActStateInternal(unsigned int stateN);
	MZoneActivityState* getMZoneActStateInternal(unsigned int zoneN);

	InNetConnectivityState* getInnetConStateInternal(unsigned int stateN);
	MZoneConnectivityState* getMZoneConStateInternal(unsigned int zoneN);

private:
	CBMStateX2GRGODecouple();

	ct_uint32_t numInnets;
	ct_uint32_t numZones;

	ActivityParams *actParamsList[2];
	ConnectivityParams *conParams;

	InNetConnectivityState *innetConStates[2];
	MZoneConnectivityState *mzoneConStates[2];

	InNetActivityState *innetActStates[2];
	MZoneActivityState *mzoneActStates[2];
};


#endif /* CBMSTATEX2GRGODECOUPLE_H_ */
