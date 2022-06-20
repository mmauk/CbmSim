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
#include "randGenerators/sfmt.h"
#include "params/connectivityparams.h"

class InNetConnectivityState
{
public:
	InNetConnectivityState();
	InNetConnectivityState(unsigned int msPerStep, int randSeed);
	//InNetConnectivityState(ConnectivityParams *parameters, std::fstream &infile);
	//InNetConnectivityState(const InNetConnectivityState &state);
	~InNetConnectivityState();
	
	void writeState(std::fstream &outfile);

	//glomerulus

	bool *haspGLfromMFtoGL[NUM_GL] = {false};
	int *numpGLfromGLtoGO[NUM_GL] = {0};
	int *pGLfromGLtoGO[NUM_GL][MAX_NUM_P_GL_FROM_GL_TO_GO] = {0};
	int *numpGLfromGOtoGL[NUM_GL] = {0};
	int *pGLfromGOtoGL[NUM_GL][MAX_NUM_P_GL_FROM_GO_TO_GL] = {0};
	int *numpGLfromGLtoGR[NUM_GL] = {0};
	int *pGLfromGLtoGR[NUM_GL][MAX_NUM_P_GL_FROM_GL_TO_GR] = {0};
	int *pGLfromMFtoGL[NUM_GL] = {0};
	int *numpMFfromMFtoGL[NUM_MF] = {0};
	int *pMFfromMFtoGL[NUM_MF][MAX_NUM_P_MF_FROM_MF_TO_GL] = {0};
	int *numpMFfromMFtoGR[NUM_MF] = {0};
	int *pMFfromMFtoGR[NUM_MF][MAX_NUM_P_MF_FROM_MF_TO_GR] = {0};
	int *numpMFfromMFtoGO[NUM_MF] = {0};
	int *pMFfromMFtoGO[NUM_MF][MAX_NUM_P_MF_FROM_MF_TO_GO] = {0};

	//golgi
	int *numpGOfromGLtoGO[NUM_GO] = {0};
	int *pGOfromGLtoGO[NUM_GO][MAX_NUM_P_GO_FROM_GL_TO_GO] = {0};
	int *numpGOfromGOtoGL[NUM_GO] = {0};
	int *pGOfromGOtoGL[NUM_GO][MAX_NUM_P_GO_FROM_GO_TO_GL] = {0};
	int *numpGOfromMFtoGO[NUM_GO] = {0};
	int *pGOfromMFtoGO[NUM_GO][MAX_NUM_P_GO_FROM_MF_TO_GO] = {0};
	int *numpGOfromGOtoGR[NUM_GO] = {0};
	int *pGOfromGOtoGR[NUM_GO][MAX_NUM_P_GO_FROM_GO_TO_GR] = {0};
	int *numpGOfromGRtoGO[NUM_GO] = {0};
	int *pGOfromGRtoGO[NUM_GO][MAX_NUM_P_GO_FROM_GR_TO_GO] = {0};

	// coincidentally, numcongotogo == maxnumpgogabaingogo
	int *numpGOGABAInGOGO[NUM_GO] = {0};
	int *pGOGABAInGOGO[NUM_GO][NUM_CON_GO_TO_GO] = {0};
	int *numpGOGABAOutGOGO[NUM_GO] = {0};			
	int *pGOGABAOutGOGO[NUM_GO][NUM_CON_GO_TO_GO] = {0};			

	// go <-> go gap junctions
	int *numpGOCoupInGOGO[NUM_GO] = {0};
	int *pGOCoupInGOGO[NUM_GO][NUM_P_GO_TO_GO_GJ] = {0};
	int *numpGOCoupOutGOGO[NUM_GO] = {0};
	int *pGOCoupOutGOGO[NUM_GO][NUM_P_GO_TO_GO_GJ] = {0};
	float *pGOCoupOutGOGOCCoeff[NUM_GO][NUM_P_GO_TO_GO_GJ] = {0};
	float *pGOCoupInGOGOCCoeff[NUM_GO][NUM_P_GO_TO_GO_GJ] = {0};

	//granule
	ct_uint32_t *pGRDelayMaskfromGRtoBSP[NUM_GR] = {0};
	int *numpGRfromGLtoGR[NUM_GR] = {0};
	int *pGRfromGLtoGR[NUM_GR][MAX_NUM_P_GR_FROM_GL_TO_GR] = {0};
	int *numpGRfromGRtoGO[NUM_GR] = {0};
	int *pGRfromGRtoGO[NUM_GR][MAX_NUM_P_GR_FROM_GR_TO_GO] = {0};
	int *pGRDelayMaskfromGRtoGO[NUM_GR][MAX_NUM_P_GR_FROM_GR_TO_GO] = {0};
	int *numpGRfromGOtoGR[NUM_GR] = {0};
	int *pGRfromGOtoGR[NUM_GR][MAX_NUM_P_GR_FROM_GO_TO_GR] = {0};
	int *numpGRfromMFtoGR[NUM_GR] = {0};
	int *pGRfromMFtoGR[NUM_GR][MAX_NUM_P_GR_FROM_MF_TO_GR] = {0};

protected:
	void allocateMemory();
	void initializeVals();
	void stateRW(bool read, std::fstream &file);

	void connectMFGL_noUBC(CRandomSFMT &randGen);
	void connectGLGR(CRandomSFMT &randGen);
	void connectGRGO(CRandomSFMT &randGen);
	void connectGOGL(CRandomSFMT &randGen);
	void connectGOGODecayP(CRandomSFMT &randGen);
	void connectGOGO_GJ(CRandomSFMT &randGen);
	void translateMFGL();
	void translateGOGL();
	void assignGRDelays(unsigned int msPerStep);
};

#endif /* INNETCONNECTIVITYSTATE_H_ */
