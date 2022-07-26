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
	MZoneConnectivityState(ConnectivityParams *cp, int randSeed);
	MZoneConnectivityState(ConnectivityParams *cp, std::fstream &infile);
	//MZoneConnectivityState(const MZoneConnectivityState &state);

	~MZoneConnectivityState();

	void readState(ConnectivityParams *cp, std::fstream &infile);
	void writeState(ConnectivityParams *cp, std::fstream &outfile);

	//basket cells
	ct_uint32_t **pBCfromBCtoPC;
	ct_uint32_t **pBCfromPCtoBC;

	//stellate cells
	ct_uint32_t **pSCfromSCtoPC;

	//purkinje cells
	ct_uint32_t **pPCfromBCtoPC;
	ct_uint32_t **pPCfromPCtoBC;
	ct_uint32_t **pPCfromSCtoPC;
	ct_uint32_t **pPCfromPCtoNC;
	ct_uint32_t *pPCfromIOtoPC;

	//nucleus cells
	ct_uint32_t **pNCfromPCtoNC;
	ct_uint32_t **pNCfromNCtoIO;
	ct_uint32_t **pNCfromMFtoNC;

	//inferior olivary cells
	ct_uint32_t **pIOfromIOtoPC;
	ct_uint32_t **pIOfromNCtoIO;
	ct_uint32_t **pIOInIOIO;
	ct_uint32_t **pIOOutIOIO;

private:
	void allocateMemory(ConnectivityParams *cp);
	void initializeVals(ConnectivityParams *cp);
	void deallocMemory();
	void stateRW(ConnectivityParams *cp, bool read, std::fstream &file);
	
	void connectBCtoPC(ConnectivityParams *cp);
	void connectPCtoBC(ConnectivityParams *cp);
	void connectSCtoPC(ConnectivityParams *cp);
	void connectPCtoNC(ConnectivityParams *cp, int randSeed);
	void connectNCtoIO(ConnectivityParams *cp);
	void connectMFtoNC(ConnectivityParams *cp);
	void connectIOtoPC(ConnectivityParams *cp);
	void connectIOtoIO(ConnectivityParams *cp);
};

#endif /* MZONECONNECTIVITYSTATE_H_ */

