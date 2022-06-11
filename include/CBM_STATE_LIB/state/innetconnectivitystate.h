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
#include <vector>
#include <limits.h>
#include <algorithm>

#include <memoryMgmt/dynamic2darray.h>
#include <fileIO/rawbytesrw.h>
#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>

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

	bool state_equal(const InNetConnectivityState &compState);
	bool state_unequal(const InNetConnectivityState &compState);

	//bool deleteGOGOConPair(int srcGON, int destGON);
	//bool addGOGOConPair(int srcGON, int destGON);

	// leaving as a public param so that we can initialize all of our public arrays below
	// NOTE: in the future, classes will be converted to structs
	//static const ConnectivityParams *cp;

	//glomerulus

	int numpGLfromGLtoGO[NUM_GL] = {0};
	int pGLfromGLtoGO[NUM_GL][MAX_NUM_P_GL_FROM_GL_TO_GO] = {0};

	int haspGLfromGOtoGL[NUM_GL] = {0};
	int numpGLfromGOtoGL[NUM_GL] = {0};
	int pGLfromGOtoGL[NUM_GL][MAX_NUM_P_GL_FROM_GO_TO_GL] = {0};

	int numpGLfromGLtoGR[NUM_GL] = {0};
	int pGLfromGLtoGR[NUM_GL][MAX_NUM_P_GL_FROM_GL_TO_GR] = {0};

	int haspGLfromMFtoGL[NUM_GL] = {0};
	int pGLfromMFtoGL[NUM_GL] = {0};
	int numpMFfromMFtoGL[NUM_MF] = {0};
	int pMFfromMFtoGL[NUM_MF][MAX_NUM_P_MF_FROM_MF_TO_GL] = {0};

	int numpMFfromMFtoGR[NUM_MF] = {0};
	int pMFfromMFtoGR[NUM_MF][MAX_NUM_P_MF_FROM_MF_TO_GR] = {0};
	int numpMFfromMFtoGO[NUM_MF] = {0};
	int pMFfromMFtoGO[NUM_MF][MAX_NUM_P_MF_FROM_MF_TO_GO] = {0};

	//ubc
	//int *numpGLfromUBCtoGL;
	//int **pGLfromUBCtoGL;
	//int *numpUBCfromUBCtoGL;
	//int **pUBCfromUBCtoGL;//[numMF][numGLOutPerMF];
	//int *xCoorsUBCGL;
	//int *yCoorsUBCGL;
	//int *spanArrayUBCtoGLX;
	//int *spanArrayUBCtoGLY;
	//int spanUBCtoGLX;
	//int spanUBCtoGLY;
	//int numpUBCtoGL;
	//int ubcX;
	//int ubcY;
	//int numUBC;
	//int numUBCfromUBCtoGL;

	//int *pUBCfromGLtoUBC;
	//int *numpUBCfromGLtoUBC;
	//int *pGLfromGLtoUBC;
	//int *numpGLfromGLtoUBC;
	//int *spanArrayGLtoUBCX;
	//int *spanArrayGLtoUBCY;
	//int *xCoorsGLUBC;
	//int *yCoorsGLUBC;

	//int *numpUBCfromUBCtoGR;
	//int **pUBCfromUBCtoGR;//[numMF][maxNumGROutPerMF];
	//int *numpGRfromUBCtoGR;
	//int **pGRfromUBCtoGR;//[numMF][maxNumGROutPerMF];
	//
	//int *numpUBCfromUBCtoGO;
	//int **pUBCfromUBCtoGO;//[numMF][maxNumGROutPerMF];
	//int *numpGOfromUBCtoGO;
	//int **pGOfromUBCtoGO;//[numMF][maxNumGROutPerMF];
	//
	//int ubcIndex;
	//int *numpUBCfromUBCOutUBC;
	//int **pUBCfromUBCOutUBC;//[numMF][maxNumGROutPerMF];
	//int *numpUBCfromUBCInUBC;
	//int **pUBCfromUBCInUBC;//[numMF][maxNumGROutPerMF];

	////mossy fiber

	//int **pMFfromMFtoUBC;
	//int *numpMFfromMFtoUBC;
	//int *pUBCfromMFtoUBC;
	//int *numpUBCfromMFtoUBC;

	//golgi
	int numpGOfromGLtoGO[NUM_GO] = {0};
	int pGOfromGLtoGO[NUM_GO][MAX_NUM_P_GO_FROM_GL_TO_GO] = {0};

	int numpGOfromGOtoGL[NUM_GO] = {0};
	int pGOfromGOtoGL[NUM_GO][MAX_NUM_P_GO_FROM_GO_TO_GL] = {0};

	int numpGOfromMFtoGO[NUM_GO] = {0};
	int pGOfromMFtoGO[NUM_GO][MAX_NUM_P_GO_FROM_MF_TO_GO] = {0};

	int numpGOfromGOtoGR[NUM_GO] = {0};
	int pGOfromGOtoGR[NUM_GO][MAX_NUM_P_GO_FROM_GO_TO_GR] = {0};
	
	int numpGOfromGRtoGO[NUM_GO] = {0};
	int pGOfromGRtoGO[NUM_GO][MAX_NUM_P_GO_FROM_GR_TO_GO] = {0};

	// coincidentally, numcongotogo == maxnumpgogabaingogo
	int numpGOGABAInGOGO[NUM_GO] = {0};
	int pGOGABAInGOGO[NUM_GO][NUM_CON_GO_TO_GO] = {0};
	int numpGOGABAOutGOGO[NUM_GO] = {0};			
	int pGOGABAOutGOGO[NUM_GO][NUM_CON_GO_TO_GO] = {0};			

	//int *numpGOfromGOtoUBC;//[numGO];
	//int **pGOfromGOtoUBC;//[numGO][maxNumGROutPerGO];
	//int *numpUBCfromGOtoUBC;//[numGO];
	//int **pUBCfromGOtoUBC;//[numGO][maxNumGROutPerGO];


	// go <-> go gap junctions
	int numpGOCoupInGOGO[NUM_GO] = {0};
	int pGOCoupInGOGO[NUM_GO][NUM_P_GO_TO_GO_GJ] = {0};
	int numpGOCoupOutGOGO[NUM_GO] = {0};
	int pGOCoupOutGOGO[NUM_GO][NUM_P_GO_TO_GO_GJ] = {0};
	float pGOCoupOutGOGOCCoeff[NUM_GO][NUM_P_GO_TO_GO_GJ] = {0};
	float pGOCoupInGOGOCCoeff[NUM_GO][NUM_P_GO_TO_GO_GJ] = {0};


	//granule
	ct_uint32_t pGRDelayMaskfromGRtoBSP[NUM_GR] = {0};

	int numpGRfromGLtoGR[NUM_GR] = {0};
	int pGRfromGLtoGR[NUM_GR][MAX_NUM_P_GR_FROM_GL_TO_GR] = {0};

	int numpGRfromGRtoGO[NUM_GR] = {0};
	int pGRfromGRtoGO[NUM_GR][MAX_NUM_P_GR_FROM_GR_TO_GO] = {0};
	int pGRDelayMaskfromGRtoGO[NUM_GR][MAX_NUM_P_GR_FROM_GR_TO_GO] = {0};

	int numpGRfromGOtoGR[NUM_GR] = {0};
	int pGRfromGOtoGR[NUM_GR][MAX_NUM_P_GR_FROM_GO_TO_GR] = {0};

	int numpGRfromMFtoGR[NUM_GR] = {0};
	int pGRfromMFtoGR[NUM_GR][MAX_NUM_P_GR_FROM_MF_TO_GR] = {0};

	//int **pBCfromPFtoBC;
	//int *numpBCfromPFtoBC;
	//int **pGRfromPFtoBC;
	//int *numpGRfromPFtoBC;
	//
	//int **pGRDelayMaskfromGRtoBC;	

protected:

	//virtual std::vector<ct_uint32_t> getConCommon(int cellN, ct_int32_t *numpCellCon, ct_uint32_t **pCellCon);
	//virtual std::vector<std::vector<ct_uint32_t> > getPopConCommon(int numCells, ct_int32_t *numpCellCon, ct_uint32_t **pCellCon);

	//virtual std::vector<int> getConCommon(int cellN, int *numpCellCon, int **pCellCon);
	//virtual std::vector<std::vector<int> > getPopConCommon(int numCells, int *numpCellCon, int **pCellCon);

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
	//void connectGLUBC();
	//void connectMFGL_withUBC(CRandomSFMT *randGen);
	//void translateUBCGL();
	//void connectGOGO(CRandomSFMT *randGen);
	//void connectGOGOBias(CRandomSFMT *randGen);
	//void connectGOGODecay(CRandomSFMT *randGen);
	//void connectUBCGL();
	//void connectPFtoBC();
	//void assignPFtoBCDelays(unsigned int msPerStep);


private:
	//void connectCommon(int **srcConArr, int32_t *srcNumCon,
	//		int **destConArr, int *destNumCon,
	//		int srcMaxNumCon, int numSrcCells,
	//		int destMaxNumCon, int destNormNumCon,
	//		int srcGridX, int srcGridY, int destGridX, int destGridY,
	//		int srcSpanOnDestGridX, int srcSpanOnDestGridY,
	//		int normConAttempts, int maxConAttempts, bool needUnique, CRandomSFMT &randGen);

	
	//void translateCommon(int **pPreGLConArr, int *numpPreGLCon,
	//		int **pGLPostGLConArr, int *numpGLPostGLCon,
	//		int **pPreConArr, int *numpPreCon,
	//		int **pPostConArr, int *numpPostCon,
	//		int numPre);
};


#endif /* INNETCONNECTIVITYSTATE_H_ */
