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
	InNetConnectivityState(unsigned int msPerStep, int randSeed);
	InNetConnectivityState(std::fstream &infile);
	~InNetConnectivityState();

	void readState(std::fstream &infile);
	void writeState(std::fstream &outfile);

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
	void allocateMemory();
	void initializeVals();
	void deallocMemory();
	void stateRW(bool read, std::fstream &file);

	void connectMFGL_noUBC();
	void connectGLGR(CRandomSFMT &randGen);
	void connectGRGO();
	void connectGOGL(CRandomSFMT &randGen);
	void connectGOGODecayP(CRandomSFMT &randGen);
	void connectGOGO_GJ(CRandomSFMT &randGen);
	void translateMFGL();
	void translateGOGL();
	void assignGRDelays(unsigned int msPerStep);
};

#endif /* INNETCONNECTIVITYSTATE_H_ */
