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

#include "fileIO/rawbytesrw.h"
#include "stdDefinitions/pstdint.h"
#include "randGenerators/sfmt.h"
#include "memoryMgmt/dynamic2darray.h"
#include "params/connectivityparams.h"

class MZoneConnectivityState
{
public:
	MZoneConnectivityState();
	MZoneConnectivityState(int randSeed);
	MZoneConnectivityState(std::fstream &infile);
	//MZoneConnectivityState(const MZoneConnectivityState &state);

	~MZoneConnectivityState();

	void writeState(std::fstream &outfile);

	//basket cells
	ct_uint32_t pBCfromBCtoPC[NUM_BC][NUM_P_BC_FROM_BC_TO_PC] = {0};
	ct_uint32_t pBCfromPCtoBC[NUM_BC][NUM_P_BC_FROM_PC_TO_BC] = {0};

	//stellate cells
	ct_uint32_t pSCfromSCtoPC[NUM_SC][NUM_P_SC_FROM_SC_TO_PC] = {0};

	//purkinje cells
	ct_uint32_t pPCfromBCtoPC[NUM_PC][NUM_P_PC_FROM_BC_TO_PC] = {0};
	ct_uint32_t pPCfromPCtoBC[NUM_PC][NUM_P_PC_FROM_PC_TO_BC] = {0};
	ct_uint32_t pPCfromSCtoPC[NUM_PC][NUM_P_PC_FROM_SC_TO_PC] = {0};
	ct_uint32_t pPCfromPCtoNC[NUM_PC][NUM_P_PC_FROM_PC_TO_NC] = {0};
	ct_uint32_t pPCfromIOtoPC[NUM_PC] = {0};

	//nucleus cells
	ct_uint32_t pNCfromPCtoNC[NUM_NC][NUM_P_NC_FROM_PC_TO_NC] = {0};
	ct_uint32_t pNCfromNCtoIO[NUM_NC][NUM_P_NC_FROM_NC_TO_IO] = {0};
	ct_uint32_t pNCfromMFtoNC[NUM_NC][NUM_P_NC_FROM_MF_TO_NC] = {0};

	//inferior olivary cells
	ct_uint32_t pIOfromIOtoPC[NUM_IO][NUM_P_IO_FROM_IO_TO_PC] = {0};
	ct_uint32_t pIOfromNCtoIO[NUM_IO][NUM_P_IO_FROM_NC_TO_IO] = {0};
	ct_uint32_t pIOInIOIO[NUM_IO][NUM_P_IO_IN_IO_TO_IO] = {0};
	ct_uint32_t pIOOutIOIO[NUM_IO][NUM_P_IO_OUT_IO_TO_IO] = {0};

private:
	void allocateMemory();
	void initializeVals();
	void deallocMemory();
	void stateRW(bool read, std::fstream &file);
	
	void connectBCtoPC();
	void connectPCtoBC();
	void connectSCtoPC();
	void connectPCtoNC(int randSeed);
	void connectNCtoIO();
	void connectMFtoNC();
	void connectIOtoPC();
	void connectIOtoIO();
};

#endif /* MZONECONNECTIVITYSTATE_H_ */

