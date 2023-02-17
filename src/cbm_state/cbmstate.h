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

#include <cstdint>
#include "innetconnectivitystate.h"
#include "mzoneconnectivitystate.h"
#include "innetactivitystate.h"
#include "mzoneactivitystate.h"
#include "connectivityparams.h" // <-- added in 06/01/2022
#include "activityparams.h"

class CBMState
{
	public:
		CBMState();
		CBMState(unsigned int nZones);
		// TODO: make a choice which of two below constructors want to keep
		CBMState(unsigned int nZones, enum plasticity plast_type, std::fstream &sim_file_buf);
		//CBMState(unsigned int nZones, std::string inFile);
		~CBMState();

		void readState(std::fstream &infile);
		void writeState(std::fstream &outfile);

		uint32_t getNumZones();

		InNetActivityState* getInnetActStateInternal();
		MZoneActivityState* getMZoneActStateInternal(unsigned int zoneN);

		InNetConnectivityState* getInnetConStateInternal();
		MZoneConnectivityState* getMZoneConStateInternal(unsigned int zoneN);

	private:
		uint32_t numZones;

		InNetConnectivityState *innetConState;
		MZoneConnectivityState **mzoneConStates;

		InNetActivityState *innetActState;
		MZoneActivityState **mzoneActStates;
};

#endif /* CBMSTATE_H_ */

