/*
 * innetconnectivitystate.h
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#ifndef INNETCONNECTIVITYSTATE_H_
#define INNETCONNECTIVITYSTATE_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <algorithm>
#include "fileIO/rawbytesrw.h"
#include "stdDefinitions/pstdint.h"
#include "memoryMgmt/dynamic2darray.h"
#include "randGenerators/sfmt.h"
#include "params/connectivityparams.h"

class InNetConnectivityState
{
public:
	InNetConnectivityState();
	InNetConnectivityState(
		ConnectivityParams *cp,
		unsigned int msPerStep,
		int randSeed
		);
	InNetConnectivityState(ConnectivityParams *cp, std::fstream &infile);
	//InNetConnectivityState(const InNetConnectivityState &state);
	~InNetConnectivityState();

	void readState(ConnectivityParams *cp, std::fstream &infile);
	void writeState(ConnectivityParams *cp, std::fstream &outfile);

	//glomerulus
	bool *haspGLfromMFtoGL;
	int *numpGLfromGLtoGO;
	int **pGLfromGLtoGO;
	int *numpGLfromGOtoGL;
	int **pGLfromGOtoGL;
	int *numpGLfromGLtoGR;
	int **pGLfromGLtoGR;
	int *pGLfromMFtoGL;
	int *numpMFfromMFtoGL;
	int **pMFfromMFtoGL;
	int *numpMFfromMFtoGR;
	int **pMFfromMFtoGR;
	int *numpMFfromMFtoGO;
	int **pMFfromMFtoGO;

	//golgi
	int *numpGOfromGLtoGO;
	int **pGOfromGLtoGO;
	int *numpGOfromGOtoGL;
	int **pGOfromGOtoGL;
	int *numpGOfromMFtoGO;
	int **pGOfromMFtoGO;
	int *numpGOfromGOtoGR;
	int **pGOfromGOtoGR;
	int *numpGOfromGRtoGO;
	int **pGOfromGRtoGO;

	// coincidentally, numcongotogo == maxnumpgogabaingogo
	int *numpGOGABAInGOGO;
	int **pGOGABAInGOGO;
	int *numpGOGABAOutGOGO;
	int **pGOGABAOutGOGO;

	// go <-> go gap junctions
	int *numpGOCoupInGOGO;
	int **pGOCoupInGOGO;
	int *numpGOCoupOutGOGO;
	int **pGOCoupOutGOGO;
	float **pGOCoupOutGOGOCCoeff;
	float **pGOCoupInGOGOCCoeff;

	//granule
	ct_uint32_t *pGRDelayMaskfromGRtoBSP;
	int *numpGRfromGLtoGR;
	int **pGRfromGLtoGR;
	int *numpGRfromGRtoGO;
	int **pGRfromGRtoGO;
	int **pGRDelayMaskfromGRtoGO;
	int *numpGRfromGOtoGR;
	int **pGRfromGOtoGR;
	int *numpGRfromMFtoGR;
	int **pGRfromMFtoGR;

protected:
	void allocateMemory(ConnectivityParams *cp);
	void initializeVals(ConnectivityParams *cp);
	void deallocMemory();
	void stateRW(ConnectivityParams *cp, bool read, std::fstream &file);

	void connectMFGL_noUBC(ConnectivityParams *cp);
	void connectGLGR(ConnectivityParams *cp, CRandomSFMT &randGen);
	void connectGRGO(ConnectivityParams *cp);
	void connectGOGL(ConnectivityParams *cp, CRandomSFMT &randGen);
	void connectGOGODecayP(ConnectivityParams *cp, CRandomSFMT &randGen);
	void connectGOGO_GJ(ConnectivityParams *cp, CRandomSFMT &randGen);
	void translateMFGL(ConnectivityParams *cp);
	void translateGOGL(ConnectivityParams *cp);
	void assignGRDelays(ConnectivityParams *cp, unsigned int msPerStep);
};

#endif /* INNETCONNECTIVITYSTATE_H_ */
