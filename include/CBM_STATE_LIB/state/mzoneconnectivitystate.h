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
#include <string.h>
#include <math.h>
#include <limits.h>

#include <memoryMgmt/dynamic2darray.h>
#include <memoryMgmt/arrayinitalize.h>
#include <memoryMgmt/arraycopy.h>
#include <fileIO/rawbytesrw.h>
#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>

#include "params/connectivityparams.h"

class MZoneConnectivityState
{
public:
	MZoneConnectivityState(ConnectivityParams *parameters, int randSeed);
	MZoneConnectivityState(ConnectivityParams *parameters, std::fstream &infile);
	MZoneConnectivityState(const MZoneConnectivityState &state);

	virtual ~MZoneConnectivityState();

	void writeState(std::fstream &outfile);

	bool operator==(const MZoneConnectivityState &compState);
	bool operator!=(const MZoneConnectivityState &compState);

	ConnectivityParams *cp;

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
	MZoneConnectivityState();

	void allocateMemory();

	void stateRW(bool read, std::fstream &file);

	void initalizeVars();
	
	
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
