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
#include <memoryMgmt/arraycopy.h>
#include <fileIO/rawbytesrw.h>
#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>

#include "params/connectivityparams.h"
#include "interfaces/iinnetconstate.h"

class InNetConnectivityState : public virtual IInNetConState
{
public:
	InNetConnectivityState(ConnectivityParams *parameters, unsigned int msPerStep, int randSeed, int goRecipParam, int simNum);
	InNetConnectivityState(ConnectivityParams *parameters, std::fstream &infile);
	InNetConnectivityState(const InNetConnectivityState &state);

	virtual ~InNetConnectivityState();
	

	virtual void writeState(std::fstream &outfile);

	virtual bool operator==(const InNetConnectivityState &compState);
	virtual bool operator!=(const InNetConnectivityState &compState);

	virtual std::vector<int> getpGOfromGOtoGLCon(int goN);
	virtual std::vector<int> getpGOfromGLtoGOCon(int goN);
	virtual std::vector<int> getpMFfromMFtoGLCon(int mfN);
	virtual std::vector<int> getpGLfromGLtoGRCon(int glN);

	virtual std::vector<int> getpGRfromMFtoGR(int grN);
	virtual std::vector<std::vector<int> > getpGRPopfromMFtoGR();

	virtual std::vector<int> getpGRfromGOtoGRCon(int grN);
	virtual std::vector<std::vector<int> > getpGRPopfromGOtoGRCon();

	virtual std::vector<int> getpGRfromGRtoGOCon(int grN);
	virtual std::vector<std::vector<int> > getpGRPopfromGRtoGOCon();

	virtual std::vector<int> getpGOfromGRtoGOCon(int goN);
	virtual std::vector<std::vector<int> > getpGOPopfromGRtoGOCon();

	virtual std::vector<int> getpGOfromGOtoGRCon(int goN);
	virtual std::vector<std::vector<int> > getpGOPopfromGOtoGRCon();

	virtual std::vector<int> getpMFfromMFtoGRCon(int mfN);

	virtual std::vector<int> getpMFfromMFtoGOCon(int mfN);
	virtual std::vector<int> getpGOfromMFtoGOCon(int goN);
	virtual std::vector<std::vector<int> > getpGOPopfromMFtoGOCon();

	virtual std::vector<int> getpGOOutGOGOCon(int goN);
	virtual std::vector<std::vector<int> > getpGOPopOutGOGOCon();

	virtual std::vector<int> getpGOInGOGOCon(int goN);
	virtual std::vector<std::vector<int> > getpGOPopInGOGOCon();

	virtual std::vector<int> getpGOCoupOutGOGOCon(int goN);
	virtual std::vector<std::vector<int> > getpGOPopCoupOutGOGOCon();

	virtual std::vector<int> getpGOCoupInGOGOCon(int goN);
	virtual std::vector<std::vector<int> > getpGOPopCoupInGOGOCon();

	virtual std::vector<ct_uint32_t> getGOIncompIndfromGRtoGO();
	virtual std::vector<ct_uint32_t> getGRIncompIndfromGRtoGO();

	virtual bool deleteGOGOConPair(int srcGON, int destGON);
	virtual bool addGOGOConPair(int srcGON, int destGON);

	//glomerulus

	int *numpGLfromGLtoGO;
	int **pGLfromGLtoGO;

	int totalGOGL;
	int totalGOGR;
	int *numpGLfromGOtoGL;
	int *haspGLfromGOtoGL;
	int **pGLfromGOtoGL;
	int *spanArrayGOtoGLX;
	int *spanArrayGOtoGLY;
	int *xCoorsGOGL;
	int *yCoorsGOGL;


	int *numpGLfromGLtoGR;
	int **pGLfromGLtoGR;
	
	//ubc
	int *numpGLfromUBCtoGL;
	int **pGLfromUBCtoGL;
	int *numpUBCfromUBCtoGL;
	int **pUBCfromUBCtoGL;//[numMF][numGLOutPerMF];
	int *xCoorsUBCGL;
	int *yCoorsUBCGL;
	int *spanArrayUBCtoGLX;
	int *spanArrayUBCtoGLY;
	int spanUBCtoGLX;
	int spanUBCtoGLY;
	int numpUBCtoGL;
	int ubcX;
	int ubcY;
	int numUBC;
	int numUBCfromUBCtoGL;

	int *pUBCfromGLtoUBC;
	int *numpUBCfromGLtoUBC;
	int *pGLfromGLtoUBC;
	int *numpGLfromGLtoUBC;
	int *spanArrayGLtoUBCX;
	int *spanArrayGLtoUBCY;
	int *xCoorsGLUBC;
	int *yCoorsGLUBC;

	int *numpUBCfromUBCtoGR;
	int **pUBCfromUBCtoGR;//[numMF][maxNumGROutPerMF];
	int *numpGRfromUBCtoGR;
	int **pGRfromUBCtoGR;//[numMF][maxNumGROutPerMF];
	
	int *numpUBCfromUBCtoGO;
	int **pUBCfromUBCtoGO;//[numMF][maxNumGROutPerMF];
	int *numpGOfromUBCtoGO;
	int **pGOfromUBCtoGO;//[numMF][maxNumGROutPerMF];
	
	int ubcIndex;
	int *numpUBCfromUBCOutUBC;
	int **pUBCfromUBCOutUBC;//[numMF][maxNumGROutPerMF];
	int *numpUBCfromUBCInUBC;
	int **pUBCfromUBCInUBC;//[numMF][maxNumGROutPerMF];

	//mossy fiber
	int *haspGLfromMFtoGL;
	int *pGLfromMFtoGL;
	int *numpMFfromMFtoGL;
	int **pMFfromMFtoGL;//[numMF][numGLOutPerMF];
	int *xCoorsMFGL;
	int *yCoorsMFGL;
	int *spanArrayMFtoGLX;
	int *spanArrayMFtoGLY;

	int *numpMFfromMFtoGR;
	int **pMFfromMFtoGR;//[numMF][maxNumGROutPerMF];
	int *numpMFfromMFtoGO;
	int **pMFfromMFtoGO;//[numMF][maxNumGOOutPerMF];

	int **pMFfromMFtoUBC;
	int *numpMFfromMFtoUBC;
	int *pUBCfromMFtoUBC;
	int *numpUBCfromMFtoUBC;

	//golgi
	int *numpGOfromGLtoGO;
	int **pGOfromGLtoGO;//[numGO][maxNumGLInPerGO];

	int *numpGOfromGOtoGL;
	int **pGOfromGOtoGL;//[numGO][maxNumGLOutPerGO];

	int *numpGOfromMFtoGO;
	int **pGOfromMFtoGO;//[numGO][maxNumMFInPerGO];

	int *numpGOfromGOtoGR;//[numGO];
	int **pGOfromGOtoGR;//[numGO][maxNumGROutPerGO];
	
	int *numpGOfromGOtoUBC;//[numGO];
	int **pGOfromGOtoUBC;//[numGO][maxNumGROutPerGO];
	int *numpUBCfromGOtoUBC;//[numGO];
	int **pUBCfromGOtoUBC;//[numGO][maxNumGROutPerGO];

	int *numpGOfromGRtoGO;
	int **pGOfromGRtoGO;
	int *spanArrayPFtoGOX;
	int *spanArrayPFtoGOY;
	int *xCoorsPFGO;
	int *yCoorsPFGO;
	int *spanArrayAAtoGOX;
	int *spanArrayAAtoGOY;
	int *xCoorsAAGO;
	int *yCoorsAAGO;



	float *PconGOGL;
	float *Pcon;
	int *numpGOGABAInGOGO;
	int **pGOGABAInGOGO;
	int *numpGOGABAOutGOGO;			// GOGO
	int **pGOGABAOutGOGO;			// GOGO
	int *spanArrayGOtoGOsynX;
	int *spanArrayGOtoGOsynY;
	int *xCoorsGOGOsyn;
	int *yCoorsGOGOsyn;
	bool **conGOGOBoolOut;

	int *numpGOCoupInGOGO;
	int **pGOCoupInGOGO;
	int *numpGOCoupOutGOGO;
	int **pGOCoupOutGOGO;
	float **pGOCoupOutGOGOCCoeff;
	float **pGOCoupInGOGOCCoeff;
	bool **gjConBool;

	int *spanArrayGOtoGOgjX;
	int *spanArrayGOtoGOgjY;
	int *xCoorsGOGOgj;
	int *yCoorsGOGOgj;
	float *gjPcon;
	float *gjCC;
	float gjPconX;	
	float gjPconY;
	float gjCCX;
	float gjCCY;


	//[numGO][maxNumGOOutPerGO];//TODO: special
	//ct_int32_t goConGOOutLocal;//[2][maxNumGOOutPerGO];//TODO: how to define gogo local connectivity in the parameters?

	//bool *pGOGABAUniOutGOGO;

	//granule
	ct_uint32_t *pGRDelayMaskfromGRtoBSP;//[numGR]; //TODO: add in parameters delay stuff

	int *numpGRfromGLtoGR;
	int **pGRfromGLtoGR;//[numGR][maxNumInPerGR];
	int *xCoorsGRGL;
	int *yCoorsGRGL;
	int *spanArrayGRtoGLX;
	int *spanArrayGRtoGLY;


	int *numpGRfromGRtoGO;
	int **pGRfromGRtoGO;//[maxNumGOOutPerGR][numGR];
	int **pGRDelayMaskfromGRtoGO;//[maxNumGOOutPerGR][numGR];

	int *numpGRfromGOtoGR;
	int **pGRfromGOtoGR;//[maxNumInPerGR][numGR];

	int *numpGRfromMFtoGR;//[numGR];
	int **pGRfromMFtoGR;//[maxNumInPerGR][numGR];
	int postSynGRIndex;
	int glIndex;
	int mfIndex;
	int goIndex;

	int *spanArrayPFtoBCX;
	int *spanArrayPFtoBCY;
	int *xCoorsPFBC;
	int *yCoorsPFBC;
	
	int **pBCfromPFtoBC;
	int *numpBCfromPFtoBC;
	int **pGRfromPFtoBC;
	int *numpGRfromPFtoBC;
	
	int **pGRDelayMaskfromGRtoBC;	

protected:
	ConnectivityParams *cp;

	virtual std::vector<ct_uint32_t> getConCommon(int cellN, ct_int32_t *numpCellCon, ct_uint32_t **pCellCon);
	virtual std::vector<std::vector<ct_uint32_t> > getPopConCommon(int numCells, ct_int32_t *numpCellCon, ct_uint32_t **pCellCon);


	virtual std::vector<int> getConCommon(int cellN, int *numpCellCon, int **pCellCon);
	virtual std::vector<std::vector<int> > getPopConCommon(int numCells, int *numpCellCon, int **pCellCon);


	virtual void allocateMemory();

	virtual void stateRW(bool read, std::fstream &file);

	virtual void initializeVals();

	virtual void connectGLUBC();
	virtual void connectGRGL(CRandomSFMT *randGen);
	virtual void connectGOGL(CRandomSFMT *randGen);
	virtual void connectMFGL_noUBC(CRandomSFMT *randGen);
	virtual void connectMFGL_withUBC(CRandomSFMT *randGen);
	virtual void translateUBCGL();
	virtual void translateMFGL();
	virtual void translateGOGL(CRandomSFMT *randGen);
	virtual void connectGRGO(CRandomSFMT *randGen, int goRecipParam);
	virtual void connectGOGO(CRandomSFMT *randGen);
	virtual void connectGOGO_GJ(CRandomSFMT *randGen);
	virtual void connectGOGOBias(CRandomSFMT *randGen);
		virtual void connectGOGODecay(CRandomSFMT *randGen);
		virtual void connectGOGODecayP(CRandomSFMT *randGen, int goRecipParam, int simNum);
	virtual void connectUBCGL();
	virtual void assignGRDelays(unsigned int msPerStep);
	virtual void connectPFtoBC();
	virtual void assignPFtoBCDelays(unsigned int msPerStep);

/*	virtual void connectCommon(ct_uint32_t **srcConArr, int32_t *srcNumCon,
			ct_uint32_t **destConArr, ct_int32_t *destNumCon,
			ct_uint32_t srcMaxNumCon, ct_uint32_t numSrcCells,
			ct_uint32_t destMaxNumCon, ct_uint32_t destNormNumCon,
			ct_uint32_t srcGridX, ct_uint32_t srcGridY, ct_uint32_t destGridX, ct_uint32_t destGridY,
			ct_uint32_t srcSpanOnDestGridX, ct_uint32_t srcSpanOnDestGridY,
			ct_uint32_t normConAttempts, ct_uint32_t maxConAttempts, bool needUnique,
			CRandomSFMT *randGen);*/
	virtual void connectCommon(int **srcConArr, int32_t *srcNumCon,
			int **destConArr, int *destNumCon,
			int srcMaxNumCon, int numSrcCells,
			int destMaxNumCon, int destNormNumCon,
			int srcGridX, int srcGridY, int destGridX, int destGridY,
			int srcSpanOnDestGridX, int srcSpanOnDestGridY,
			int normConAttempts, int maxConAttempts, bool needUnique,
			CRandomSFMT *randGen);
/*
	virtual void translateCommon(ct_uint32_t **pPreGLConArr, ct_int32_t *numpPreGLCon,
			ct_uint32_t **pGLPostGLConArr, ct_int32_t *numpGLPostGLCon,
			ct_uint32_t **pPreConArr, ct_int32_t *numpPreCon,
			ct_uint32_t **pPostConArr, ct_int32_t *numpPostCon,
			ct_uint32_t numPre);*/
	virtual void translateCommon(int **pPreGLConArr, int *numpPreGLCon,
			int **pGLPostGLConArr, int *numpGLPostGLCon,
			int **pPreConArr, int *numpPreCon,
			int **pPostConArr, int *numpPostCon,
			int numPre);

private:
	InNetConnectivityState();
};


#endif /* INNETCONNECTIVITYSTATE_H_ */
