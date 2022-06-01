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
		CBMState(std::fstream &infile);

		// VVVV the one we actually use...
		CBMState(ConnectivityParams &conParams, ActivityParams *actParams,
			unsigned int nZones, int goRecipParam, int simNum);

		CBMState(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
			int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed);

		// y virtual where is the inheritance
		virtual ~CBMState();	

		void writeState(std::fstream &outfile);

		bool operator==(CBMState &compState);
		bool operator!=(CBMState &compState);

		ct_uint32_t getNumZones();

		IActivityParams* getActivityParams();

		IMZoneActState* getMZoneActState(unsigned int zoneN);

		ActivityParams* getActParamsInternal();
		ConnectivityParams* getConParamsInternal();

		InNetActivityState* getInnetActStateInternal();
		MZoneActivityState* getMZoneActStateInternal(unsigned int zoneN);

		InNetConnectivityState* getInnetConStateInternal();
		MZoneConnectivityState* getMZoneConStateInternal(unsigned int zoneN);

	private:
		
		void newState(ConnectivityParams &conParams, ActivityParams *actParams, unsigned int nZones,
				int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed, int goRecipParam, int simNum);
	
		// don't know wtf this is from	
		//void newStateParam(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
		//		int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed, int spanP, int numConP);

		ct_uint32_t numZones;

		InNetConnectivityState *innetConState;
		MZoneConnectivityState **mzoneConStates;

		InNetActivityState *innetActState;
		MZoneActivityState **mzoneActStates;
};

#endif /* CBMSTATE_H_ */

