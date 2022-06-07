/*
 * mzoneconnectivitystate.h
 *
 *  Created on: Nov 21, 2012
 *      Author: consciousness
 */

#ifndef MZONECONNECTIVITYSTATE_H_
#define MZONECONNECTIVITYSTATE_H_

#include <fstream>
#include <iostream>
#include <algorithm>
#include <string.h>
#include <math.h>
#include <limits.h>

#include <memoryMgmt/dynamic2darray.h>
#include <fileIO/rawbytesrw.h>
#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>

#include "params/connectivityparams.h"

class MZoneConnectivityState
{
public:
	MZoneConnectivityState();
	MZoneConnectivityState(ConnectivityParams &cp, int randSeed);
	MZoneConnectivityState(ConnectivityParams &cp, std::fstream &infile);
	//MZoneConnectivityState(const MZoneConnectivityState &state);

	virtual ~MZoneConnectivityState();

	void writeState(ConnectivityParams &cp, std::fstream &outfile);

	bool state_equal(const ConnectivityParams &cp, const MZoneConnectivityState &compState);
	bool state_unequal(const ConnectivityParams &cp, const MZoneConnectivityState &compState);

	//basket cells
	ct_uint32_t pBCfromBCtoPC[cp.NUM_BC][cp.NUM_P_BC_FROM_BC_TO_PC]();
	ct_uint32_t pBCfromPCtoBC[cp.NUM_BC][cp.NUM_P_BC_FROM_PC_TO_BC]();

	//stellate cells
	ct_uint32_t pSCfromSCtoPC[cp.NUM_SC][cp.NUM_P_SC_FROM_SC_TO_PC]();

	//purkinje cells
	ct_uint32_t pPCfromBCtoPC[cp.NUM_PC][cp.NUM_P_PC_FROM_BC_TO_PC]();
	ct_uint32_t pPCfromPCtoBC[cp.NUM_PC][cp.NUM_P_PC_FROM_PC_TO_BC]();
	ct_uint32_t pPCfromSCtoPC[cp.NUM_PC][cp.NUM_P_PC_FROM_SC_TO_PC]();
	ct_uint32_t pPCfromPCtoNC[cp.NUM_PC][cp.NUM_P_PC_FROM_PC_TO_NC]();
	ct_uint32_t pPCfromIOtoPC[cp.NUM_PC]();

	//nucleus cells
	ct_uint32_t pNCfromPCtoNC[cp.NUM_NC][cp.NUM_P_NC_FROM_PC_TO_NC]();
	ct_uint32_t pNCfromNCtoIO[cp.NUM_NC][cp.NUM_P_NC_FROM_NC_TO_IO]();
	ct_uint32_t pNCfromMFtoNC[cp.NUM_NC][cp.NUM_P_NC_FROM_MF_TO_NC]();

	//inferior olivary cells
	ct_uint32_t pIOfromIOtoPC[cp.NUM_IO][cp.NUM_P_IO_FROM_IO_TO_PC]();
	ct_uint32_t pIOfromNCtoIO[cp.NUM_IO][cp.NUM_P_IO_FROM_NC_TO_IO]();
	ct_uint32_t pIOInIOIO[cp.NUM_IO][cp.NUM_P_IO_IN_IO_TO_IO]();
	ct_uint32_t pIOOutIOIO[cp.NUM_IO][cp.NUM_P_IO_OUT_IO_TO_IO]();

private:
	void stateRW(ConnectivityParams &cp, bool read, std::fstream &file);

	void initializeVars(ConnectivityParams &cp);
	
	void connectBCtoPC(ConnectivityParams &cp);
	void connectPCtoBC(ConnectivityParams &cp);
	void connectSCtoPC(ConnectivityParams &cp);
	void connectPCtoNC(ConnectivityParams &cp, int randSeed);
	void connectNCtoIO(ConnectivityParams &cp);
	void connectMFtoNC(ConnectivityParams &cp);
	void connectIOtoPC(ConnectivityParams &cp);
	void connectIOtoIO(ConnectivityParams &cp);
};

#endif /* MZONECONNECTIVITYSTATE_H_ */
