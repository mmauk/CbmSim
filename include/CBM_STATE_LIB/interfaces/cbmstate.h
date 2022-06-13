/*
 * cbmstate.h
 *
 *  Created on: Dec 5, 2012
 *      Author: consciousness
 */

#ifndef CBMSTATE_H_
#define CBMSTATE_H_

#include <fstream>
#include <iostream>
#include <time.h>
#include <limits.h>

#include "stdDefinitions/pstdint.h"

#include "state/innetconnectivitystate.h"
#include "state/innetconstateggialtcon.h"
#include "state/mzoneconnectivitystate.h"
#include "state/innetactivitystate.h"
#include "state/mzoneactivitystate.h"

#include "params/connectivityparams.h" // <-- added in 06/01/2022
#include "iactivityparams.h"
#include "imzoneactstate.h"

class CBMState
{
	public:
		CBMState();
		CBMState(ActivityParams &ap, unsigned int nZones);

		// y virtual where is the inheritance
		virtual ~CBMState();	

		void writeState(ActivityParams &ap, std::fstream &outfile);

		ct_uint32_t getNumZones();

		IMZoneActState* getMZoneActState(unsigned int zoneN);

		InNetActivityState* getInnetActStateInternal();
		MZoneActivityState* getMZoneActStateInternal(unsigned int zoneN);

		InNetConnectivityState* getInnetConStateInternal();
		MZoneConnectivityState* getMZoneConStateInternal(unsigned int zoneN);

	private:
		ct_uint32_t numZones;

		InNetConnectivityState *innetConState;
		MZoneConnectivityState **mzoneConStates;

		InNetActivityState *innetActState;
		MZoneActivityState **mzoneActStates;

		void newState(ActivityParams &ap, unsigned int nZones,
			int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed);
	
};

#endif /* CBMSTATE_H_ */

