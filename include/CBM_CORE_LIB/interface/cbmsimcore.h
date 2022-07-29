/*
 * cbmsimcore.h
 *
 *  Created on: Dec 14, 2011
 *      Author: consciousness
 */

#ifndef CBMSIMCORE_H_
#define CBMSIMCORE_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <limits.h>
#include <time.h>

#include <stdDefinitions/pstdint.h>
#include <interfaces/cbmstate.h>
#include <randGenerators/sfmt.h>

#include "innetinterface.h"
#include "mzoneinterface.h"

#include "mzonemodules/mzone.h"
#include "innetmodules/innet.h"

/* TODO: consider altering this code so that CBMSimCore does not keep local copies
 *       of the state classes. Consider whether transferring data between classes
 *       by using classes as arguments would be just as fast as we have things now.
 *       The advantage would be that we would use less memory and it would simplify the code.
 */

class CBMSimCore
{
public:
	CBMSimCore();
	CBMSimCore(ConnectivityParams *cp, ActivityParams *ap, CBMState *state, int gpuIndStart = -1, int numGPUP2 = -1);
	~CBMSimCore();

	void calcActivity(float goMin, int simNum, float GOGR, float GRGO, float MFGO,
		float gogoW, float spillFrac);
	
	void updateMFInput(const ct_uint8_t *mfIn);
	void updateTrueMFs(bool *isTrueMF);
	void updateGRStim(int startGRStim, int numGRStim);
	void updateErrDrive(unsigned int zoneN, float errDriveRelative);

	void writeToState();
	void writeState(ConnectivityParams *cp, ActivityParams *ap, std::fstream& outfile);

	InNet* getInputNet();
	MZone** getMZoneList();

protected:
	void initCUDAStreams();
	void initAuxVars();

	void syncCUDA(std::string title);

	CBMState *simState;

	ct_uint32_t numZones;

	InNet *inputNet;
	MZone **zones;

	cudaStream_t **streams;
	int gpuIndStart;
	int numGPUs;

private:
	bool isGRStim    =  false;
	int numGRStim    =  0;
	int startGRStim  =  0;

	unsigned long curTime;

	void construct(ConnectivityParams *cp, ActivityParams *ap, CBMState *state, int *mzoneRSeed,
		int gpuIndStart, int numGPUP2);
};

#endif /* CBMSIMCORE_H_ */

