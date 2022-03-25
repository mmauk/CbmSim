/*
 * cbmsimx2grgodecouple.h
 *
 *  Created on: Apr 30, 2013
 *      Author: consciousness
 */

#ifndef CBMSIMX2GRGODECOUPLE_H_
#define CBMSIMX2GRGODECOUPLE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <limits.h>
#include <time.h>

#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>

#include <interfaces/cbmstatex2grgodecouple.h>

#include "innetinterface.h"
#include "mzoneinterface.h"

#include "mzonemodules/mzone.h"
#include "innetmodules/innet.h"

class CBMSimX2GRGODecouple
{
public:
	CBMSimX2GRGODecouple(CBMStateX2GRGODecouple *state);

	~CBMSimX2GRGODecouple();

	void calcActivity();

	void updateMFInput(const ct_uint8_t *mfIn);

	void updateErrDrive(float errDriveRelative);

	InNetInterface** getInputNetList();
	MZoneInterface** getMZoneList();

protected:
	void initCUDA();
	void initAuxVars();
	void syncCUDA(std::string title);

	ct_uint32_t numInnets;
	ct_uint32_t numZones;

	CBMStateX2GRGODecouple *simState;

	InNet *inputNets[2];
	MZone *zones[2];

	cudaStream_t **streams;
	int numGPUs;

private:
	CBMSimX2GRGODecouple();

	unsigned long curTime;
};

#endif /* CBMSIMX2GRGODECOUPLE_H_ */
