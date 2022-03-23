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

#include "iconnectivityparams.h"
#include "iactivityparams.h"
#include "iinnetconstate.h"
#include "imzoneactstate.h"

class CBMState
{
	public:
		CBMState(std::fstream &infile);
		CBMState(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
				int goRecipParam, int simNum);
		CBMState(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
				int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed);

		void writeState(std::fstream &outfile);

		bool equivalent(CBMState &compState);

		ct_uint32_t getNumZones();

		IConnectivityParams* getConnectivityParams();
		IActivityParams* getActivityParams();

		IInNetConState* getInnetConState();
		IMZoneActState* getMZoneActState(unsigned int zoneN);

		ActivityParams* getActParamsInternal();
		ConnectivityParams* getConParamsInternal();

		InNetActivityState* getInnetActStateInternal();
		MZoneActivityState* getMZoneActStateInternal(unsigned int zoneN);

		InNetConnectivityState* getInnetConStateInternal();
		MZoneConnectivityState* getMZoneConStateInternal(unsigned int zoneN);

	private:
		CBMState();
		virtual ~CBMState();
		
		void newState(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
				int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed, int goRecipParam, int simNum);
		
		void newStateParam(std::fstream &actPFile, std::fstream &conPFile, unsigned int nZones,
				int innetCRSeed, int *mzoneCRSeed, int *mzoneARSeed, int spanP, int numConP);

		ct_uint32_t numZones;

		ActivityParams *actParams;
		ConnectivityParams *conParams;

		InNetConnectivityState *innetConState;
		MZoneConnectivityState **mzoneConStates;

		InNetActivityState *innetActState;
		MZoneActivityState **mzoneActStates;
};

#endif /* CBMSTATE_H_ */

