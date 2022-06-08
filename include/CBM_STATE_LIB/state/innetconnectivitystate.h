/*
 * innetconnectivitystate.h
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#ifndef INNETCONNECTIVITYSTATE_H_
#define INNETCONNECTIVITYSTATE_H_


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
	InNetConnectivityState(ConnectivityParams &cp, unsigned int msPerStep, int randSeed);
	//InNetConnectivityState(ConnectivityParams *parameters, std::fstream &infile);
	//InNetConnectivityState(const InNetConnectivityState &state);

	~InNetConnectivityState();
	
	void writeState(ConnectivityParams &cp, std::fstream &outfile);

	bool state_equal(ConnectivityParams &cp, const InNetConnectivityState &compState);
	bool state_unequal(ConnectivityParams &cp, const InNetConnectivityState &compState);

	//bool deleteGOGOConPair(int srcGON, int destGON);
	//bool addGOGOConPair(int srcGON, int destGON);

	//glomerulus

	int numpGLfromGLtoGO[cp.NUM_GL]();
	int pGLfromGLtoGO[cp.NUM_GL][cp.MAX_NUM_P_GL_FROM_GL_TO_GO]();

	int haspGLfromGOtoGL[cp.NUM_GL]();
	int numpGLfromGOtoGL[cp.NUM_GL]();
	int pGLfromGOtoGL[cp.NUM_GL, cp.MAX_NUM_P_GL_FROM_GO_TO_GL]();

	int numpGLfromGLtoGR[cp.NUM_GL]();
	int pGLfromGLtoGR[cp.NUM_GL][cp.MAX_NUM_P_GL_FROM_GL_TO_GR]();

	int haspGLfromMFtoGL[cp.NUM_GL]();
	int pGLfromMFtoGL[cp.NUM_GL]();
	int numpMFfromMFtoGL[cp.NUM_MF]();
	int pMFfromMFtoGL[cp.NUM_MF][cp.MAX_NUM_P_MF_FROM_MF_TO_GL]();

	int numpMFfromMFtoGR[cp.NUM_MF]();
	int pMFfromMFtoGR[cp.NUM_MF][cp.MAX_NUM_P_MF_FROM_MF_TO_GR]();
	int numpMFfromMFtoGO[cp.NUM_MF]();
	int pMFfromMFtoGO[cp.NUM_MF][cp.MAX_NUM_P_MF_FROM_MF_TO_GO]();

	// NOTE: not needed for now, using old connectCommon for gl -> gr (06/02/2022)
	//int *spanArrayGRtoGLX;
	//int *spanArrayGRtoGLY;
	//int *xCoorsGRGL;
	//int *yCoorsGRGL;

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
	int numpGOfromGLtoGO[cp.NUM_GO]();
	int pGOfromGLtoGO[cp.NUM_GO][cp.MAX_NUM_P_GO_FROM_GL_TO_GO]();

	int numpGOfromGOtoGL[cp.NUM_GO]();
	int pGOfromGOtoGL[cp.NUM_GO][cp.MAX_NUM_P_GO_FROM_GO_TO_GL]();

	int numpGOfromMFtoGO[cp.NUM_GO]();
	int pGOfromMFtoGO[cp.NUM_GO][cp.MAX_NUM_P_GO_FROM_MF_TO_GO]();

	int numpGOfromGOtoGR[cp.NUM_GO]();
	int pGOfromGOtoGR[cp.NUM_GO][cp.MAX_NUM_P_GO_FROM_GO_TO_GR]();
	
	int numpGOfromGRtoGO[cp.NUM_GO]();
	int pGOfromGRtoGO[cp.NUM_GO][cp.MAX_NUM_P_GO_FROM_GR_TO_GO]();

	// coincidentally, numcongotogo == maxnumpgogabaingogo
	int numpGOGABAInGOGO[cp.NUM_GO]();
	int pGOGABAInGOGO[cp.NUM_GO][cp.NUM_CON_GO_TO_GO]();
	int numpGOGABAOutGOGO[cp.NUM_GO][cp.NUM_CON_GO_TO_GO]();			
	int pGOGABAOutGOGO[cp.NUM_GO][cp.NUM_CON_GO_TO_GO]();			

	//int *numpGOfromGOtoUBC;//[numGO];
	//int **pGOfromGOtoUBC;//[numGO][maxNumGROutPerGO];
	//int *numpUBCfromGOtoUBC;//[numGO];
	//int **pUBCfromGOtoUBC;//[numGO][maxNumGROutPerGO];


	// go <-> go gap junctions
	int numpGOCoupInGOGO[cp.NUM_GO]();
	int pGOCoupInGOGO[cp.NUM_GO][cp.NUM_P_GO_TO_GO_GJ]();
	int numpGOCoupOutGOGO[cp.NUM_GO]();
	int pGOCoupOutGOGO[cp.NUM_GO][cp.NUM_P_GO_TO_GO_GJ]();
	float pGOCoupOutGOGOCCoeff[cp.NUM_GO][cp.NUM_P_GO_TO_GO_GJ]();
	float pGOCoupInGOGOCCoeff[cp.NUM_GO][cp.NUM_P_GO_TO_GO_GJ]();


	//granule
	ct_uint32_t pGRDelayMaskfromGRtoBSP[cp.NUM_GR]();

	int numpGRfromGLtoGR[cp.NUM_GR]();
	int pGRfromGLtoGR[cp.NUM_GR][cp.MAX_NUM_P_GR_FROM_GL_TO_GR]();

	int numpGRfromGRtoGO[cp.NUM_GR]();
	int pGRfromGRtoGO[cp.NUM_GR][cp.MAX_NUM_P_GR_FROM_GR_TO_GO]();
	int pGRDelayMaskfromGRtoGO[cp.NUM_GR][cp.MAX_NUM_P_GR_FROM_GR_TO_GO]();

	int numpGRfromGOtoGR[cp.NUM_GR]();
	int pGRfromGOtoGR[cp.NUM_GR][cp.MAX_NUM_P_GR_FROM_GO_TO_GR]();

	int numpGRfromMFtoGR[cp.NUM_GR]();
	int pGRfromMFtoGR[cp.NUM_GR][cp.MAX_NUM_P_GR_FROM_MF_TO_GR]();

	//int postSynGRIndex;
	//int *spanArrayPFtoBCX;
	//int *spanArrayPFtoBCY;
	//int *xCoorsPFBC;
	//int *yCoorsPFBC;
	//
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

	void initializeVals(ConnectivityParams &cp);

	void stateRW(ConnectivityParams &cp, bool read, std::fstream &file);

	void connectMFGL_noUBC(ConnectivityParams &cp, CRandomSFMT &randGen);
	void connectGLGR(ConnectivityParams &cp, CRandomSFMT &randGen);
	void connectGRGO(ConnectivityParams &cp, CRandomSFMT &randGen);
	void connectGOGL(ConnectivityParams &cp, CRandomSFMT &randGen);
	void connectGOGODecayP(ConnectivityParams &cp, CRandomSFMT &randGen);
	void connectGOGO_GJ(ConnectivityParams &cp, CRandomSFMT &randGen);
	void translateMFGL(ConnectivityParams &cp);
	void translateGOGL(ConnectivityParams &cp);
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
	void connectCommon(int srcConArr[][], int32_t srcNumCon[],
			int destConArr[][], int destNumCon[],
			int srcMaxNumCon, int numSrcCells,
			int destMaxNumCon, int destNormNumCon,
			int srcGridX, int srcGridY, int destGridX, int destGridY,
			int srcSpanOnDestGridX, int srcSpanOnDestGridY,
			int normConAttempts, int maxConAttempts, bool needUnique)

	
	//void translateCommon(int **pPreGLConArr, int *numpPreGLCon,
	//		int **pGLPostGLConArr, int *numpGLPostGLCon,
	//		int **pPreConArr, int *numpPreCon,
	//		int **pPostConArr, int *numpPostCon,
	//		int numPre);
};


#endif /* INNETCONNECTIVITYSTATE_H_ */
