/*
 * innetconnectivitystate.cpp
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#include "state/innetconnectivitystate.h"

InNetConnectivityState::InNetConnectivityState(ConnectivityParams &cp,
		unsigned int msPerStep, int randSeed)
{
	CRandomSFMT randGen(randSeed);
	
	std::cout << "Input net state construction..." << std::endl;
	
	std::cout << "Initializing connections..." << std::endl;
	initializeVals(cp);

	std::cout << "connecting MF and GL" << std::endl;
	connectMFGL_noUBC(cp, randGen);

	std::cout << "connecting GR and GL" << std::endl;
	connectGLGR(cp, randGen);

	std::cout << "connecting GR to GO" << std::endl;
	connectGRGO(cp, randGen);

	std::cout << "connecting GO and GL" << std::endl;
	connectGOGL(cp, randGen);
	
	std::cout << "connecting GO to GO" << std::endl;
	connectGOGODecayP(cp, randGen);	

	std::cout << "connecting GO to GO gap junctions" << std::endl;
	connectGOGO_GJ(cp, randGen);
	
	std::cout << "translating MF GL" << std::endl;
	translateMFGL(cp);
	
	std::cout << "translating GO and GL" << std::endl;
	translateGOGL(cp);

	std::cout << "assigning GR delays" << std::endl;
	assignGRDelays(msPerStep);
	std::cout << "done" << std::endl;
}

//InNetConnectivityState::InNetConnectivityState(ConnectivityParams *parameters, std::fstream &infile)
//{
//	allocateMemory();
//
//	stateRW(true, infile);
//}

//InNetConnectivityState::InNetConnectivityState(const InNetConnectivityState &state)
//{
//	allocateMemory();
//
//	arrayCopy<int>(haspGLfromMFtoGL, state.haspGLfromMFtoGL, cp->numGL);
//	arrayCopy<int>(pGLfromMFtoGL, state.pGLfromMFtoGL, cp->numGL);
//
//	arrayCopy<int>(numpGLfromGLtoGO, state.numpGLfromGLtoGO, cp->numGL);
//	arrayCopy<int>(pGLfromGLtoGO[0], state.pGLfromGLtoGO[0],
//			cp->numGL*cp->maxnumpGLfromGLtoGO);
//
//	arrayCopy<int>(numpGLfromGOtoGL, state.numpGLfromGOtoGL, cp->numGL);
//
//	arrayCopy<int>(numpGLfromGLtoGR, state.numpGLfromGLtoGR, cp->numGL);
//	arrayCopy<int>(pGLfromGLtoGR[0], state.pGLfromGLtoGR[0],
//			cp->numGL*cp->maxnumpGLfromGOtoGL);
//
//	arrayCopy<int>(numpMFfromMFtoGL, state.numpMFfromMFtoGL, cp->numMF);
//	arrayCopy<int>(pMFfromMFtoGL[0], state.pMFfromMFtoGL[0],
//			cp->numMF*20);
//
//	arrayCopy<int>(numpMFfromMFtoGR, state.numpMFfromMFtoGR, cp->numMF);
//	arrayCopy<int>(pMFfromMFtoGR[0], state.pMFfromMFtoGR[0],
//			cp->numMF*cp->maxnumpMFfromMFtoGR);
//
//	arrayCopy<int>(numpMFfromMFtoGO, state.numpMFfromMFtoGO, cp->numMF);
//	arrayCopy<int>(pMFfromMFtoGO[0], state.pMFfromMFtoGO[0],
//			cp->numMF*cp->maxnumpMFfromMFtoGO);
//
//	arrayCopy<int>(numpGOfromGLtoGO, state.numpGOfromGLtoGO, cp->numGO);
//	arrayCopy<int>(pGOfromGLtoGO[0], state.pGOfromGLtoGO[0],
//			cp->numGO*cp->maxnumpGOfromGLtoGO);
//
//	arrayCopy<int>(numpGOfromGOtoGL, state.numpGOfromGOtoGL, cp->numGO);
//	arrayCopy<int>(pGOfromGOtoGL[0], state.pGOfromGOtoGL[0],
//			cp->numGO*cp->maxnumpGOfromGOtoGL);
//
//	arrayCopy<int>(numpGOfromMFtoGO, state.numpGOfromMFtoGO, cp->numGO);
//	arrayCopy<int>(pGOfromMFtoGO[0], state.pGOfromMFtoGO[0],
//			cp->numGO*16);
//
//	arrayCopy<int>(numpGOfromGOtoGR, state.numpGOfromGOtoGR, cp->numGO);
//	arrayCopy<int>(pGOfromGOtoGR[0], state.pGOfromGOtoGR[0],
//			cp->numGO*cp->maxnumpGOfromGOtoGR);
//
//	arrayCopy<int>(numpGOfromGRtoGO, state.numpGOfromGRtoGO, cp->numGO);
//	arrayCopy<int>(pGOfromGRtoGO[0], state.pGOfromGRtoGO[0],
//			cp->numGO*cp->maxnumpGOfromGRtoGO);
//
//	arrayCopy<int>(numpGOGABAInGOGO, state.numpGOGABAInGOGO, cp->numGO);
//	arrayCopy<int>(pGOGABAInGOGO[0], state.pGOGABAInGOGO[0],
//			cp->numGO*cp->maxnumpGOGABAInGOGO);
//
//	arrayCopy<int>(numpGOGABAOutGOGO, state.numpGOGABAOutGOGO, cp->numGO);
//	arrayCopy<int>(pGOGABAOutGOGO[0], state.pGOGABAOutGOGO[0],
//			cp->numGO*cp->maxGOGOsyn);
//
//	arrayCopy<int>(numpGOCoupInGOGO, state.numpGOCoupInGOGO, cp->numGO);
//	arrayCopy<int>(pGOCoupInGOGO[0], state.pGOCoupInGOGO[0],
//			cp->numGO*49);
//
//	arrayCopy<int>(numpGOCoupOutGOGO, state.numpGOCoupOutGOGO, cp->numGO);
//	arrayCopy<int>(pGOCoupOutGOGO[0], state.pGOCoupOutGOGO[0],
//			cp->numGO*49);
//
//	arrayCopy<ct_uint32_t>(pGRDelayMaskfromGRtoBSP, state.pGRDelayMaskfromGRtoBSP, cp->numGR);
//
//	arrayCopy<int>(numpGRfromGLtoGR, state.numpGRfromGLtoGR, cp->numGR);
//	arrayCopy<int>(pGRfromGLtoGR[0], state.pGRfromGLtoGR[0],
//			cp->numGR*cp->maxnumpGRfromGLtoGR);
//
//	arrayCopy<int>(numpGRfromGRtoGO, state.numpGRfromGRtoGO, cp->numGR);
//	arrayCopy<int>(pGRfromGRtoGO[0], state.pGRfromGRtoGO[0],
//			cp->numGR*cp->maxnumpGRfromGRtoGO);
//	arrayCopy<int>(pGRDelayMaskfromGRtoGO[0], state.pGRDelayMaskfromGRtoGO[0],
//			cp->numGR*cp->maxnumpGRfromGRtoGO);
//
//	arrayCopy<int>(numpGRfromGOtoGR, state.numpGRfromGOtoGR, cp->numGR);
//	arrayCopy<int>(pGRfromGOtoGR[0], state.pGRfromGOtoGR[0],
//			cp->numGR*cp->maxnumpGRfromGOtoGR);
//
//	arrayCopy<int>(numpGRfromMFtoGR, state.numpGRfromMFtoGR, cp->numGR);
//	arrayCopy<int>(pGRfromMFtoGR[0], state.pGRfromMFtoGR[0],
//			cp->numGR*cp->maxnumpGRfromMFtoGR);
//}

InNetConnectivityState::~InNetConnectivityState() {}

void InNetConnectivityState::writeState(std::fstream &outfile)
{
	std::cout << "Writing input network connectivity state to disk..." << std::endl;
	stateRW(false, (std::fstream &)outfile);
	std::cout << "finished writing input network connectivity to disk." << std::endl;
}

bool InNetConnectivityState::operator==(const InNetConnectivityState &compState)
{
	bool eq = true;
	for (int i = 0; i < cp.NUM_GL; i++)
	{
		eq = eq && (haspGLfromMFtoGL[i] == compState.haspGLfromMFtoGL[i]);
	}

	for(int i = 0; i < cp.NUM_GO; i++)
	{
		eq = eq && (numpGOfromGOtoGR[i] == compState.numpGOfromGOtoGR[i]);
	}

	for (int i = 0; i < cp.NUM_GR; i++)
	{
		eq = eq && (numpGRfromGOtoGR[i] == compState.numpGRfromGOtoGR[i]);
	}

	return eq;
}

bool InNetConnectivityState::operator!=(const InNetConnectivityState &compState)
{
	return !(*this == compState);
}

//bool InNetConnectivityState::deleteGOGOConPair(int srcGON, int destGON)
//{
//	bool hasCon = false;
//	int conN;
//	
//	for (int i = 0; i < numpGOGABAOutGOGO[srcGON]; i++)
//	{
//		if (pGOGABAOutGOGO[srcGON][i] == destGON)
//		{
//			hasCon = true;
//			conN = i;
//			break;
//		}
//	}
//
//	if (!hasCon)
//	{
//		return hasCon;
//	}
//
//	for (int i = conN; i < numpGOGABAOutGOGO[srcGON] - 1; i++)
//	{
//		pGOGABAOutGOGO[srcGON][i] = pGOGABAOutGOGO[srcGON][i + 1];
//	}
//	numpGOGABAOutGOGO[srcGON]--;
//
//	for (int i = 0; i < numpGOGABAInGOGO[destGON]; i++)
//	{
//		if (pGOGABAInGOGO[destGON][i] == srcGON)
//		{
//			conN = i;
//		}
//	}
//
//	for(int i = conN; i < numpGOGABAInGOGO[destGON] - 1; i++)
//	{
//		pGOGABAInGOGO[destGON][i] = pGOGABAInGOGO[destGON][i + 1];
//	}
//	numpGOGABAInGOGO[destGON]--;
//
//	return hasCon;
//}

//bool InNetConnectivityState::addGOGOConPair(int srcGON, int destGON)
//{
//	if (numpGOGABAOutGOGO[srcGON] >= cp->maxnumpGOGABAOutGOGO ||
//			numpGOGABAInGOGO[destGON] >= cp->maxnumpGOGABAInGOGO)
//	{
//		return false;
//	}
//
//	pGOGABAOutGOGO[srcGON][numpGOGABAOutGOGO[srcGON]] = destGON;
//	numpGOGABAOutGOGO[srcGON]++;
//
//	pGOGABAInGOGO[destGON][numpGOGABAInGOGO[destGON]] = srcGON;
//	numpGOGABAInGOGO[destGON]++;
//
//	return true;
//}

void InNetConnectivityState::stateRW(bool read, std::fstream &file)
{
	std::cout << "glomerulus" << std::endl;
	//glomerulus
	rawBytesRW((char *)haspGLfromMFtoGL, cp.NUM_GL * sizeof(int), read, file);
	std::cout << "glomerulus 1.1" << std::endl;
	rawBytesRW((char *)pGLfromMFtoGL, cp.numGL * sizeof(int), read, file);

	std::cout << "glomerulus 2" << std::endl;
	rawBytesRW((char *)numpGLfromGLtoGO, cp.NUM_GL * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGO[0],
		cp.NUM_GL * cp.MAX_NUM_P_GL_FROM_GL_TO_GO * sizeof(int), read, file);

	std::cout << "glomerulus 3" << std::endl;
	rawBytesRW((char *)numpGLfromGOtoGL, cp.NUM_GL * sizeof(int), read, file);
	rawBytesRW((char *)haspGLfromGOtoGL, cp.NUM_GL * sizeof(int), read, file);
	//rawBytesRW((char *)pGLfromGOtoGL, cp->numGL*sizeof(int), read, file);

	std::cout << "glomerulus 4" << std::endl;
	rawBytesRW((char *)numpGLfromGLtoGR, cp.NUM_GL * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGR[0],
		cp.NUM_GL * cp.MAX_NUM_P_GL_FROM_GL_TO_GR * sizeof(int), read, file);

	std::cout << "mf" << std::endl;
	//mossy fibers
	rawBytesRW((char *)numpMFfromMFtoGL, cp.NUM_MF * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGL[0],
		cp.NUM_MF * cp.MAX_NUM_P_MF_FROM_MF_TO_GL * sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGR, cp.NUM_MF * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGR[0],
		cp.NUM_MF * cp.MAX_NUM_P_MF_FROM_MF_TO_GR * sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGO, cp.NUM_MF * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGO[0],
		cp.NUM_MF * cp.MAX_NUM_P_MF_FROM_MF_TO_GO * sizeof(int), read, file);

	std::cout << "golgi" << std::endl;
	//golgi
	rawBytesRW((char *)numpGOfromGLtoGO, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGLtoGO[0],
		cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_GL_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGL, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGL[0],
		cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_GO_TO_GL * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromMFtoGO, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromMFtoGO[0],
		cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_MF_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGR, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGR[0],
		cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_GO_TO_GR * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGRtoGO, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGRtoGO[0],
		cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_GR_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAInGOGO, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAInGOGO[0],
		cp.NUM_GO * cp.NUM_CON_GO_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAOutGOGO, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAOutGOGO[0],
		cp.NUM_GO * cp.NUM_CON_GO_TO_GO * sizeof(int), read, file);
	
	rawBytesRW((char *)numpGOCoupInGOGO, cp.NUM_GO*sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupInGOGO[0],
		cp.NUM_GO * cp.NUM_P_GO_TO_GO_GJ * sizeof(int), read, file);

	rawBytesRW((char *)numpGOCoupOutGOGO, cp.NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupOutGOGO[0],
		cp.NUM_GO * cp.NUM_P_GO_TO_GO_GJ * sizeof(int), read, file);

	std::cout << "granule" << std::endl;
	//granule
	rawBytesRW((char *)pGRDelayMaskfromGRtoBSP, cp.NUM_GR * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromGLtoGR, cp.NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGLtoGR[0],
		cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GL_TO_GR * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGRtoGO, cp.NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGRtoGO[0],
		cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GR_TO_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGRDelayMaskfromGRtoGO[0],
		cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GR_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGOtoGR, cp.NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGOtoGR[0],
		cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GO_TO_GR * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromMFtoGR, cp.NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromMFtoGR[0],
		cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_MF_TO_GR * cp.NUM_GR * sizeof(int), read, file);
}

void InNetConnectivityState::initializeVals(ConnectivityParams &cp)
{
	// gl
	std::fill(pGLfromGLtoGO[0], pGLfromGLtoGO[0]
		+ cp.NUM_GL * cp.MAX_NUM_P_GL_FROM_GL_TO_GO, UINT_MAX);
	std::fill(pGLfromGOtoGL[0], pGLfromGOtoGL[0]
		+ cp.NUM_GL * cp.MAX_NUM_P_GL_FROM_GO_TO_GL, UINT_MAX);
	std::fill(pGLfromGLtoGR[0], pGLfromGLtoGR[0]
		+ cp.NUM_GL * cp.MAX_NUM_P_GL_FROM_GL_TO_GR, UINT_MAX);
	std::fill(pGLfromMFtoGL, pGLfromMFtoGL + cp.NUM_GL, UINT_MAX);

	//mf
	std::fill(pMFfromMFtoGL[0], pMFfromMFtoGL[0]
		+ cp.NUM_MF * MAX_NUM_P_MF_FROM_MF_TO_GL, UINT_MAX);

	std::fill(pMFfromMFtoGR[0], pMFfromMFtoGR[0]
		+ cp.NUM_MF * cp->MAX_NUM_P_MF_FROM_MF_TO_GR, UINT_MAX);
	std::fill(pMFfromMFtoGO[0], pMFfromMFtoGO[0]
		+ cp.NUM_MF * cp.MAX_NUM_P_MF_FROM_MF_TO_GO, UINT_MAX);

	// go
	std::fill(pGOfromGLtoGO[0], pGOfromGLtoGO[0]
		+ cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_GL_TO_GO, UINT_MAX);

	std::fill(pGOfromGOtoGL[0], pGOfromGOtoGL[0]
		+ cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_GO_TO_GL, UINT_MAX);

	std::fill(pGOfromMFtoGO[0], pGOfromMFtoGO[0]
		+ cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_MF_TO_GO, UINT_MAX);

	std::fill(pGOfromGOtoGR[0], pGOfromGOtoGR[0]
		+ cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_GO_TO_GR, UINT_MAX);

	std::fill(pGOfromGRtoGO[0], pGOfromGRtoGO[0]
		+ cp.NUM_GO * cp.MAX_NUM_P_GO_FROM_GR_TO_GO, UINT_MAX);
	
	std::fill(pGOGABAInGOGO[0], pGOGABAInGOGO[0]
		+ cp.NUM_CON_GO_TO_GO, UINT_MAX);

	std::fill(pGOGABAOutGOGO[0], pGOGABAOutGOGO[0]
		+ cp.NUM_GO * cp.NUM_CON_GO_TO_GO, UINT_MAX);

	// go gap junctions
	std::fill(pGOCoupInGOGO[0], pGOCoupInGOGO[0]
		+ cp.NUM_GO * cp.NUM_P_GO_TO_GO_GJ, UINT_MAX);
	std::fill(pGOCoupInGOGOCCoeff[0], pGOCoupInGOGOCCoeff[0]
		+ cp.NUM_GO * cp.NUM_P_GO_TO_GO_GJ, UINT_MAX);
	std::fill(pGOCoupOutGOGO[0], pGOCoupOutGOGO[0]
		+ cp.NUM_GO * cp.NUM_P_GO_TO_GO_GJ, UINT_MAX);
	std::fill(pGOCoupOutGOGOCCoeff[0], pGOCoupOutGOGOCCoeff[0]
		+ cp.NUM_GO * cp.NUM_P_GO_TO_GO_GJ, UINT_MAX);
	
	
	// gr
	std::fill(pGRfromGLtoGR[0], pGRfromGLtoGR[0]
		+ cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GL_TO_GR, UINT_MAX);

	std::fill(pGRfromGRtoGO[0], pGRfromGRtoGO[0]
		+ cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GR_TO_GO, UINT_MAX);
	std::fill(pGRDelayMaskfromGRtoGO[0], pGRDelayMaskfromGRtoGO[0]
		+ cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GR_TO_GO, UINT_MAX);

	std::fill(pGRfromGOtoGR[0], pGRfromGOtoGR[0]
		+ cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GO_TO_GR, UINT_MAX);

	std::fill(pGRfromMFtoGR[0], pGRfromMFtoGR[0]
		+ cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_MF_TO_GR, UINT_MAX);
}

void InNetConnectivityState::connectMFGL_noUBC(ConnectivityParams &cp, CRandomSFMT &randGen)
{
	// shorter convenience var names
	int glX = cp.GL_X;
	int glY = cp.GL_Y;
	int mfX = cp.MF_X;
	int mfY = cp.MF_Y;

	// define span and coord arrays locally
	int spanArrayMFtoGLX[cp.SPAN_MF_TO_GL_X + 1];
	int spanArrayMFtoGLY[cp.SPAN_MF_TO_GL_Y + 1];
	int xCoorsMFGL[cp.NUM_P_MF_TO_GL];
	int yCoorsMFGL[cp.NUM_P_MF_TO_GL];

	// fill span arrays and coord arrays
	for(int i = 0; i < cp.SPAN_MF_TO_GL_X + 1;i++)
	{
		spanArrayMFtoGLX[i] = i - (cp.SPAN_MF_TO_GL_X / 2);
	}

	for(int i = 0; i < cp.SPAN_MF_TO_GL_Y + 1;i++)
	{
		spanArrayMFtoGLY[i] = i - (cp.SPAN_MF_TO_GL_Y / 2);
	}
		
	for(int i = 0; i < cp.NUM_P_MF_TO_GL; i++)
	{
		xCoorsMFGL[i] = spanArrayMFtoGLX[i % (cp.SPAN_MF_TO_GL_X + 1)];
		yCoorsMFGL[i] = spanArrayMFtoGLY[i / (cp.SPAN_MF_TO_GL_Y + 1)];		
	}

	// scale factors from one cell coord to another
	float gridXScaleSrctoDest = (float)mfX / (float)glX; 
	float gridYScaleSrctoDest = (float)mfY/ (float)glY; 

	// random mf index array, supposedly to even out distribution of connections
	int rMFInd[cp.NUM_MF]();	
	for(int i = 0; i < cp.NUM_MF; i++) rMFInd[i] = i;	
	std::random_shuffle(rMFInd, rMFInd + cp.NUM_MF);

	// fill random span array with linear indices
	int rMFSpanInd[cp.NUM_P_MF_TO_GL]();
	for (int i = 0; i < cp.NUM_P_MF_TO_GL; i++) rMFSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	// attempt to make connections
	for (int attempts = 0; attempts < cp.MAX_MF_TO_GL_ATTEMPTS; attempts++)
	{
		std::random_shuffle(rMFInd, rMFInd + cp.NUM_MF);	
		// for each attempt, loop through all presynaptic cells
		for (int i = 0; i < cp.NUM_MF; i++)
		{	
			//Select MF Coordinates from random index array: Complete	
			srcPosX = rMFInd[i] % mfX;
			srcPosY = rMFInd[i] / mfX;		
			
			std::random_shuffle(rMFSpanInd, rMFSpanInd + cp.NUM_P_MF_TO_GL);	
			// for each presynaptic cell, attempt to make up to initial output + max attempts
			// connections.	
			for (int j = 0; j < cp.NUM_P_MF_TO_GL; j++)
			{	
				// calculation of which gl cell this mf is connecting to
				destPosX = xCoorsMFGL[rMFSpanInd[j]]; 
				destPosY = yCoorsMFGL[rMFSpanInd[j]];	

				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);	
				
				destPosX = (destPosX % glX + glX) % glX;
				destPosY = (destPosY % glY + glY) % glY;
				
				destIndex = destPosY * glX + destPosX;
			
				// break if we hit this dynamic con limit
				if (numpMFfromMFtoGL[rMFInd[i]] == (cp.INITIAL_MF_OUTPUT + attempts)) break;
				
				// if we dont have connections, make them	
				if (!haspGLfromMFtoGL[destIndex]) 
				{	
					// assign gl index to mf array and vice versa, then set our bool array to true
					pMFfromMFtoGL[rMFInd[i]][numpMFfromMFtoGL[rMFInd[i]]] = destIndex;
					numpMFfromMFtoGL[rMFInd[i]]++;
					
					pGLfromMFtoGL[destIndex] = rMFInd[i];
					haspGLfromMFtoGL[destIndex] = true;	
				}
			}
		}
	}	

	// finish up by counting the total number of mf -> gl cons made
	int count = 0;
	for (int i = 0; i < cp.NUM_MF; i++) count += numpMFfromMFtoGL[i];
	
	std::cout << "Total number of Mossy Fiber to Glomeruli connections:	" << count << std::endl;
	std::cout << "Correct number: " << cp.NUM_GL << std::endl;
}

void InNetConnectivityState::connectGLGR(ConnectivityParams &cp, CRandomSFMT &randGen)
{
	connectCommon(pGRfromGLtoGR, numpGRfromGLtoGR,
			pGLfromGLtoGR, numpGLfromGLtoGR,
			cp.MAX_NUM_P_GR_FROM_GL_TO_GR, cp.NUM_GR,
			cp.MAX_NUM_P_GL_FROM_GL_TO_GR, cp.LOW_NUM_P_GL_FROM_GL_TO_GR,
			cp.GR_X, cp.GR_Y, cp.GL_X, cp.GL_Y,
			cp.SPAN_GL_TO_GR_X, cp.SPAN_GL_TO_GR_Y,
			cp.LOW_GL_TO_GR_ATTEMPTS, cp.MAX_GL_TO_GR_ATTEMPTS, true);

	int count = 0;

	for (int i = 0; i < cp.NUM_GL; i++)
	{
		count += numpGLfromGLtoGR[i];
	}

	std::cout << "Total number of Glomeruli to Granule connections:	" << count << std::endl; 
	std::cout << "Correct number: " << cp.NUM_GR * cp.MAX_NUM_P_GR_FROM_GL_TO_GR << std::endl;
}

void InNetConnectivityState::connectGRGO(ConnectivityParams &cp, CRandomSFMT &randGen)
{
	int grX = cp.GR_X;
	int grY = cp.GR_Y;
	int goX = cp.GO_X;
	int goY = cp.GO_Y;

	int spanArrayPFtoGOX[cp.SPAN_PF_TO_GO_X + 1];
	int spanArrayPFtoGOY[cp.SPAN_PF_TO_GO_Y + 1];
	int xCoorsPFGO[cp.NUM_P_PF_TO_GO];
	int yCoorsPFGO[cp.NUM_P_PF_TO_GO];

	//PARALLEL FIBER TO GOLGI 
	for (int i = 0; i < cp.SPAN_PF_TO_GO_X + 1; i++)
	{
		spanArrayPFtoGOX[i] = i - (spanPFtoGOX / 2);
	}

	for (int i = 0; i < cp.SPAN_PF_TO_GO_Y + 1; i++)
	{
		spanArrayPFtoGOY[i] = i - (spanPFtoGOY / 2);
	}
		
	for (int i = 0; i < cp.NUM_P_PF_TO_GO; i++)
	{
		xCoorsPFGO[i] = spanArrayPFtoGOX[i % (cp.SPAN_PF_TO_GO_X + 1)];
		yCoorsPFGO[i] = spanArrayPFtoGOY[i / (cp.SPAN_PF_TO_GO_X + 1)];	
	}

	// Grid Scale: Complete
	float gridXScaleSrctoDest = (float)goX / (float)grX; 
	float gridYScaleSrctoDest = (float)goY / (float)grY; 

	//Make Random Span Array: Complete
	int rPFSpanInd[cp.NUM_P_PF_TO_GO]();
	for (int i = 0; i < cp.NUM_P_PF_TO_GO; i++) rPFSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int attempts = 0; attempts < cp.MAX_PF_TO_GO_ATTEMPTS; attempts++)
	{
		for (int i = 0; i < cp.NUM_GO; i++)
		{	
			srcPosX = i % goX;
			srcPosY = i / goX;

			std::random_shuffle(rPFSpanInd, rPFSpanInd + cp.NUM_P_PF_TO_GO);		
			for (int j = 0; j < cp.MAX_PF_TO_GO_INPUT; j++)
			{	
				destPosX = xCoorsPFGO[rPFSpanInd[j]]; 
				destPosY = yCoorsPFGO[rPFSpanInd[j]];	
				
				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);
				
				destPosX = (destPosX % grX + grX) % grX;
				destPosY = (destPosY % grY + grY) % grY;
						
				destIndex = destPosY * grX + destPosX;

				if (numpGOfromGRtoGO[i] < cp.MAX_PF_TO_GO_INPUT)
				{	
					pGOfromGRtoGO[i][numpGOfromGRtoGO[i]] = destIndex;
					numpGOfromGRtoGO[i]++;

					pGRfromGRtoGO[destIndex][numpGRfromGRtoGO[destIndex]] = i;
					numpGRfromGRtoGO[destIndex]++;	
				}
			}
		}
	}

	int spanArrayAAtoGOX[cp.SPAN_AA_TO_GO_X + 1];
	int spanArrayAAtoGOY[cp.SPAN_AA_TO_GO_Y + 1];
	int xCoorsAAGO[cp.NUM_P_AA_TO_GO];
	int yCoorsAAGO[cp.NUM_P_AA_TO_GO];

	//Make Span Array: Complete	
	for (int i = 0; i < cp.SPAN_AA_TO_GO_X + 1;i++)
	{
		spanArrayAAtoGOX[i] = i - (cp.SPAN_AA_TO_GO_X / 2);
	}

	for (int i = 0; i < cp.SPAN_AA_TO_GO_Y + 1;i++)
	{
		spanArrayAAtoGOY[i] = i - (cp.SPAN_AA_TO_GO_Y / 2);
	}
		
	for (int i = 0; i < cp.NUM_P_AA_TO_GO; i++)
	{
		xCoorsAAGO[i] = spanArrayAAtoGOX[i % (cp.SPAN_AA_TO_GO_X + 1)];
		yCoorsAAGO[i] = spanArrayAAtoGOY[i / (cp.SPAN_AA_TO_GO_Y + 1)];	
	}
	
	int rAASpanInd[cp.NUM_P_AA_TO_GO]();
	for (int i = 0; i < cp.NUM_P_AA_TO_GO; i++) rAASpanInd[i] = i;

	for (int attempts = 0; attempts < cp.MAX_AA_TO_GO_ATTEMPTS; attempts++)
	{
		for (int i = 0; i < cp.NUM_GO; i++)
		{	
			srcPosX = i % goX;
			srcPosY = i / goX;

			std::random_shuffle(rAASpanInd, rAASpanInd + cp.NUM_P_AA_TO_GO);		
			
			for (int j = 0; j < cp.MAX_AA_TO_GO_INPUT; j++)
			{	
				destPosX = xCoorsAAGO[rAASpanInd[j]]; 
				destPosY = yCoorsAAGO[rAASpanInd[j]];	
				
				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);

				destPosX = (destPosX % grX + grX) % grX;
				destPosY = (destPosY % grY + grY) % grY;
						
				destIndex = destPosY * grX + destPosX;
				
				if (numpGOfromGRtoGO[i] < cp.MAX_AA_TO_GO_INPUT + cp.MAX_PF_TO_GO_INPUT)
				{	
					pGOfromGRtoGO[i][numpGOfromGRtoGO[i]] = destIndex;	
					numpGOfromGRtoGO[i]++;

					pGRfromGRtoGO[destIndex][numpGRfromGRtoGO[destIndex]] = i;
					numpGRfromGRtoGO[destIndex]++;	
				}
			}
		}
	}	

	int sumGOGR_GO = 0;
	
	for (int i = 0; i < cp.NUM_GO; i++)
	{
		sumGOGR_GO += numpGOfromGRtoGO[i];
	}

	std::cout << "GRtoGO_GO: " << sumGOGR_GO << std::endl;

	int sumGOGR_GR = 0;

	for (int i = 0; i < cp.NUM_GR; i++)
	{
		sumGOGR_GR += numpGRfromGRtoGO[i];
	}

	std::cout << "GRtoGO_GR: " << sumGOGR_GR << std::endl;
}

void InNetConnectivityState::connectGOGL(ConnectivityParams &cp, CRandomSFMT &randGen)
{
	connectCommon(pGOfromGLtoGO, numpGOfromGLtoGO,
			pGLfromGLtoGO, numpGLfromGLtoGO,
			cp.MAX_NUM_P_GO_FROM_GL_TO_GO, cp.NUM_GO,
			cp.MAX_NUM_P_GL_FROM_GL_TO_GO, cp.LOW_NUM_P_GL_FROM_GL_TO_GO,
			cp.GO_X, cp.GO_Y, cp.GL_X, cp.GL_Y,
			cp.SPAN_GL_TO_GO_X, cp.SPAN_GL_TO_GO_Y,
			cp.LOW_GL_TO_GO_ATTEMPTS, cp.MAX_GL_TO_GO_ATTEMPTS, false);

	int goX = cp.GO_X;
	int goY = cp.GO_Y;
	int glX = cp.GL_X;
	int glY = cp.GL_Y;

	int spanArrayGOtoGLX[cp.SPAN_GO_TO_GL_X + 1]();
	int spanArrayGOtoGLY[cp.SPAN_GO_TO_GL_Y + 1]();
	int xCoorsGOGL[cp.NUM_P_GO_TO_GL]();
	int yCoorsGOGL[cp.NUM_P_GO_TO_GL]();
	float pConGOGL[cp.NUM_P_GO_TO_GL]();

	// Make span Array
	for (int i = 0; i < cp.SPAN_GO_TO_GL_X + 1; i++)
	{
		spanArrayGOtoGLX[i] = i - (cp.SPAN_GO_TO_GL_X / 2);
	}

	for (int i = 0; i < cp.SPAN_GO_TO_GL_Y + 1; i++)
	{
		spanArrayGOtoGLY[i] = i - (cp.SPAN_GO_TO_GL_Y / 2);
	}
		
	for (int i = 0; i < cp.NUM_P_GO_TO_GL; i++)
	{
		xCoorsGOGL[i] = spanArrayGOtoGLX[i % (cp.SPAN_GO_TO_GL_X + 1)];
		yCoorsGOGL[i] = spanArrayGOtoGLY[i / (cp.SPAN_GO_TO_GL_Y + 1)];	
	}

	//Make Random Golgi cell Index Array	
	int rGOInd[cp.NUM_GO]();
	for (int i = 0; i < cp.NUM_GO; i++) rGOInd[i] = i;
	std::random_shuffle(rGOInd, rGOInd + cp.NUM_GO);

	//Make Random Span Array
	int rGOSpanInd[cp.NUM_P_GO_TO_GL]();
	for (int i = 0; i < cp.NUM_P_GO_TO_GL; i++) rGOSpanInd[i] = i;
	
	// Probability of connection as a function of distance
	for (int i = 0; i < cp.NUM_P_GO_TO_GL; i++)
	{
		float PconX = (xCoorsGOGL[i] * xCoorsGOGL[i])
			/ (2 * cp.STD_DEV_GO_TO_GL_ML * cp.STD_DEV_GO_TO_GL_ML);
		float PconY = (yCoorsGOGL[i] * yCoorsGOGL[i])
			/ (2 * cp.STD_DEV_GO_TO_GL_S * cp.STD_DEV_GO_TO_GL_S);
		PconGOGL[i] = cp.AMPL_GO_TO_GL * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < cp.NUM_P_GO_TO_GL; i++)
	{
		if ((xCoorsGOGL[i] == 0) && (yCoorsGOGL[i] == 0)) PconGOGL[i] = 0;
	}

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	float gridXScaleSrctoDest = (float)goX / (float)glX;
	float gridYScaleSrctoDest = (float)goY / (float)glY;

	for (int attempts = 0; attempts < cp.MAX_GO_TO_GL_ATTEMPTS; attempts++)
	{
		std::random_shuffle(rGOInd, rGOInd + cp.NUM_GO);	
		
		// Go through each golgi cell 
		for (int i = 0; i < cp.NUM_GO; i++)
		{
			//Select GO Coordinates from random index array: Complete	
			srcPosX = rGOInd[i] % goX;
			srcPosY = rGOInd[i] / goX;	
			
			std::random_shuffle(rGOSpanInd, rGOSpanInd + cp.NUM_P_GO_TO_GL);	
			
			for (int j = 0; j < cp.NUM_P_GO_TO_GL; j++)   
			{	
				// relative position of connection
				destPosX = xCoorsGOGL[rGOSpanInd[j]];
				destPosY = yCoorsGOGL[rGOSpanInd[j]];	

				destPosX += (int)round(srcPosX / gridXScaleSrctoDest) + destPosX; 
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest) + destPosY;

				destPosX = (destPosX % glX + glX) % glX;  
				destPosY = (destPosY % glY + glY) % glY;
				
				// Change position to Index	
				destIndex = destPosY * glX + destPosX;
					
				if (numpGOfromGOtoGL[rGOInd[i]] >= cp.INITIAL_GO_INPUT + attempts) break; 
				if ( randGen->Random() >= 1 - PconGOGL[rGOSpanInd[j]] && 
						numpGLfromGOtoGL[destIndex] < cp.MAX_NUM_P_GO_FROM_GO_TO_GL) 
				{	
					pGOfromGOtoGL[rGOInd[i]][numpGOfromGOtoGL[rGOInd[i]]] = destIndex;
					numpGOfromGOtoGL[rGOInd[i]]++;

					pGLfromGOtoGL[destIndex][numpGLfromGOtoGL[destIndex]] = rGOInd[i];
					numpGLfromGOtoGL[destIndex]++;
				}
			}
		}

		int shitCounter = 0;
		int totalGOGL = 0;

		for (int i = 0; i < cp->numGL; i++)
		{
			if (numpGLfromGOtoGL[i] < cp.MAX_NUM_P_GL_FROM_GO_TO_GL) shitCounter++;
			totalGOGL += numpGLfromGOtoGL[i];
		}
	}

	std::cout << "Empty Glomeruli Counter: " << shitCounter << std::endl;
	std::cout << "Total GO -> GL: " << totalGOGL << std::endl;
	std::cout << "avg Num  GO -> GL Per GL: " << (float)totalGOGL / (float)cp.NUM_GL << std::endl;
}

void InNetConnectivityState::connectGOGODecayP(ConnectivityParams &cp, CRandomSFMT &randGen)
{
	int goX = cp.GO_X;
	int goY = cp.GO_Y;

	int spanArrayGOtoGOsynX[cp.SPAN_GO_TO_GO_X + 1]();
	int spanArrayGOtoGOsynY[cp.SPAN_GO_TO_GO_Y + 1]();
	int xCoorsGOGOsyn[cp.NUM_P_GO_TO_GO]();
	int yCoorsGOGOsyn[cp.NUM_P_GO_TO_GO]();
	float Pcon[cp.NUM_P_GO_TO_GO](); 				

	bool conGOGOBoolOut[cp.NUM_GO][cp.NUM_GO];
	std::fill(conGOGOBoolOut[0], conGOGOBoolOut[0] + cp.NUM_GO * cp.NUM_GO, false);

	for (int i = 0; i < cp.SPAN_GO_TO_GO_X + 1; i++)
   	{
		spanArrayGOtoGOsynX[i] = i - (cp.SPAN_GO_TO_GO_X / 2);
	}

	for (int i = 0; i < cp.SPAN_GO_TO_GO_Y + 1; i++)
   	{
		spanArrayGOtoGOsynY[i] = i - (cp.SPAN_GO_TO_GO_Y / 2);
	}
		
	for (int i = 0; i < cp.NUM_P_GO_TO_GO; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (cp.SPAN_GO_TO_GO_X + 1)];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (cp.SPAN_GO_TO_GO_Y + 1)];		
	}

	for (int i = 0; i < cp.NUM_P_GO_TO_GO; i++)
	{
		float PconX = (xCoorsGOGOsyn[i] * xCoorsGOGOsyn[i])
			/ (2 * (cp.STD_DEV_GO_TO_GO * cp.STD_DEV_GO_TO_GO));
		float PconY = (yCoorsGOGOsyn[i] * y	oorsGOGOsyn[i])
			/ (2 * (cp.STD_DEV_GO_TO_GO * cp.STD_DEV_GO_TO_GO));
		Pcon[i] = cp.AMPL_GO_TO_GO * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < cp.NUM_P_GO_TO_GO; i++) 
	{
		if ((xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0)) Pcon[i] = 0;
	}
	
	int rGOGOSpanInd[cp.NUM_P_GO_TO_GO]();
	for (int i = 0; i < cp.NUM_P_GO_TO_GO; i++) rGOGOSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int attempts = 0; attempts < cp.MAX_GO_TO_GO_ATTEMPTS; attempts++) 
	{
		for (int i = 0; i < cp.NUM_GO; i++) 
		{	
			srcPosX = i % goX;
			srcPosY = i / goX;	
			
			std::random_shuffle(rGOGOSpanInd, rGOGOSpanInd + cp.NUM_P_GO_TO_GO);		
			
			for (int j = 0; j < cp.NUM_P_GO_TO_GO; j++)
		   	{	
				destPosX = srcPosX + xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
				destPosY = srcPosY + yCoorsGOGOsyn[rGOGOSpanInd[j]];	

				destPosX = (destPosX % goX + goX) % goX;
				destPosY = (destPosY % goY + goY) % goY;
						
				destIndex = destPosY * goX + destPosX;
			
				// Normal One	
				if (cp.GO_GO_RECIP_CONS && !cp.REDUCE_BASE_RECIP 
						&& randGen.Random()>= 1 - Pcon[rGOGOSpanInd[j]]
						&& !conGOGOBoolOut[i][destIndex]
						&& numpGOGABAOutGOGO[i] < cp.NUM_CON_GO_TO_GO 
						&& numpGOGABAInGOGO[destIndex] < cp.NUM_CON_GO_TO_GO) 
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;
							
					if (randGen.Random() >= 1 - cp.P_RECIP_GO_GO 
							&& !conGOGOBoolOut[destIndex][i]
							&& numpGOGABAOutGOGO[destIndex] < cp.NUM_CON_GO_TO_GO 
							&& numpGOGABAInGOGO[i] < cp.NUM_CON_GO_TO_GO) 
					{
						pGOGABAOutGOGO[destIndex][numpGOGABAOutGOGO[destIndex]] = i;
						numpGOGABAOutGOGO[destIndex]++;
						
						pGOGABAInGOGO[i][numpGOGABAInGOGO[i]] = destIndex;
						numpGOGABAInGOGO[i]++;
						
						conGOGOBoolOut[destIndex][i] = true;
					}
				}
			
				if (cp.GO_GO_RECIP_CONS && cp.REDUCE_BASE_RECIP_GO_GO
						&& randGen.Random() >= 1 - Pcon[rGOGOSpanInd[j]]
						&& !conGOGOBoolOut[i][destIndex] && (!conGOGOBoolOut[destIndex][i] ||
							randGen.Random() >= 1 - cp.P_RECIP_LOWER_BASE_GO_GO)
						&& numpGOGABAOutGOGO[i] < cp.NUM_CON_GO_TO_GO 
						&& numpGOGABAInGOGO[destIndex] < cp.NUM_CON_GO_TO_GO) 
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;	
				}

				if (!cp.GO_GO_RECIP_CONS && !cp.REDUCE_BASE_RECIP_GO_GO &&
						randGen.Random() >= 1 - Pcon[rGOGOSpanInd[j]]
						&& (!conGOGOBoolOut[i][destIndex]) && !conGOGOBoolOut[destIndex][i]
						&& numpGOGABAOutGOGO[i] < cp.NUM_CON_GO_TO_GO 
						&& numpGOGABAInGOGO[destIndex] < cp.NUM_CON_GO_TO_GO)
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;
				}
			}
		}
	}

	int totalGOGOcons = 0;

	for (int i = 0; i < cp.NUM_GO; i++)
	{
		totalGOGOcons += numpGOGABAInGOGO[i];
	}

	std::cout << "Total GOGO connections: " << totalGOGOcons << std::endl;
	std::cout << "Average GOGO connections:	" << (float)totalGOGOcons / float(cp.NUM_GO) << std::endl;
	std::cout << cp.NUM_GO << std::endl;

	int recipCounter = 0;

	for (int i = 0; i < cp.NUM_GO; i++)
	{
		for (int j = 0; j < numpGOGABAInGOGO[i]; j++)
		{
			for (int k = 0; k < numpGOGABAOutGOGO[i]; k++)
			{
				if (pGOGABAInGOGO[i][j] == pGOGABAOutGOGO[i][k] && pGOGABAInGOGO[i][j] != UINT_MAX 
						&& pGOGABAOutGOGO[i][k] != UINT_MAX)
				{
					recipCounter++;	
				}
			}
		}
	}

	float fracRecip = (float)recipCounter / (float)totalGOGOcons;
	std::cout << "FracRecip: " << fracRecip << std::endl;
}

void InNetConnectivityState::connectGOGO_GJ(ConnectivityParams &cp, CRandomSFMT &randGen)
{
	int goX = cp.GO_X;
	int goY = cp.GO_Y;

	int spanArrayGOtoGOgjX[cp.SPAN_GO_TO_GO_GJ_X + 1]();
	int spanArrayGOtoGOgjY[cp.SPAN_GO_TO_GO_GJ_Y + 1]();
	int xCoorsGOGOgj[cp.NUM_P_GO_TO_GO_GJ]();
	int yCoorsGOGOgj[cp. NUM_P_GO_TO_GO_GJ]();

	float gjPCon[cp.NUM_P_GO_TO_GO_GJ]();
	float gjCC[cp.NUM_P_GO_TO_GO_GJ]();

	bool gjConBool[cp.NUM_GO][cp.NUM_GO];
	std::fill(gjConBool[0], gjConBool[0] + (cp.NUM_GO * cp.NUM_GO), false);

	for (int i = 0; i < cp.SPAN_GO_TO_GO_GJ_X + 1; i++)
	{
		spanArrayGOtoGOgjX[i] = i - (cp.SPAN_GO_TO_GO_GJ_X / 2);
	}

	for (int i = 0; i < cp.SPAN_GO_TO_GO_GJ_Y + 1; i++)
	{
		spanArrayGOtoGOgjY[i] = i - (cp.SPAN_GO_TO_GO_GJ_Y / 2);
	}

	for (int i = 0; i < cp.NUM_P_GO_TO_GO_GJ; i++)
	{
		xCoorsGOGOgj[i] = spanArrayGOtoGOgjX[i % (cp.SPAN_GO_TO_GO_GJ_X + 1)];
		yCoorsGOGOgj[i] = spanArrayGOtoGOgjY[i / (cp.SPAN_GO_TO_GO_GJ_X + 1)];		
	}

	// "In Vivo additions"
	for (int i = 0; i < cp.NUM_P_GO_TO_GO_GJ; i++)
	{
		float gjPConX = exp(((abs(xCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );	
		float gjPConY = exp(((abs(yCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );
		gjPCon[i] = ((-1745.0 + (1836.0 / (1 + (gjPConX + gjPConY)))) * 0.01);
		
		float gjCCX = exp(abs(xCoorsGOGOgj[i] * 100.0) / 190.0);
		float gjCCY = exp(abs(yCoorsGOGOgj[i] * 100.0) / 190.0);
		gjCC[i] = (-2.3 + (23.0 / ((gjCCX + gjCCY)/2.0))) * 0.09;
	}

	// Remove self connection 
	for (int i = 0; i < cp.NUM_P_GO_TO_GO_GJ; i++)
	{
		if ((xCoorsGOGOgj[i] == 0) && (yCoorsGOGOgj[i] == 0))
		{
			gjPCon[i] = 0;
			gjCC[i] = 0;
		}
	}

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int i = 0; i < cp.NUM_GO; i++)
	{	
		srcPosX = i % goX;
		srcPosY = i / goX;	
		
		for (int j = 0; j < cp.NUM_P_GO_TO_GO_GJ; j++)
		{	
			destPosX = srcPosX + xCoorsGOGOgj[j]; 
			destPosY = srcPosY + yCoorsGOGOgj[j];	

			destPosX = (destPosX % goX + goX) % goX;
			destPosY = (destPosY % goY + goY) % goY;
					
			destIndex = destPosY * goX + destPosX;

			if ((randGen.Random()>= 1 - gjPCon[j]) && !gjConBool[i][destIndex]
					&& !gjConBool[destIndex][i])
			{	
				pGOCoupInGOGO[destIndex][numpGOCoupInGOGO[destIndex]] = i;
				pGOCoupInGOGOCCoeff[destIndex][numpGOCoupInGOGO[destIndex]] = gjCC[j];
				numpGOCoupInGOGO[destIndex]++;
					
				pGOCoupInGOGO[i][numpGOCoupInGOGO[i]] = destIndex;
				pGOCoupInGOGOCCoeff[i][numpGOCoupInGOGO[i]] = gjCC[j];
				numpGOCoupInGOGO[i]++;

				gjConBool[i][destIndex] = true;
			}
		}
	}
}

void InNetConnectivityState::translateMFGL(ConnectivityParams &cp)
{

	// Mossy fiber to Granule
	
	for (int i = 0; i < cp.NUM_GR; i++)
	{
		for (int j = 0; j < numpGRfromGLtoGR[i]; j++)
		{
			int glIndex = pGRfromGLtoGR[i][j];
			if (haspGLfromMFtoGL[glIndex])
			{
				int mfIndex = pGLfromMFtoGL[glIndex];

				pMFfromMFtoGR[mfIndex][numpMFfromMFtoGR[mfIndex]] = i; 
				numpMFfromMFtoGR[mfIndex]++;			

				pGRfromMFtoGR[i][numpGRfromMFtoGR[i]] = mfIndex;
				numpGRfromMFtoGR[i]++;
			}
		}
	}	

	int grMFInputCounter = 0;
	
	for (int i = 0; i < cp.NUM_GR; i++)
	{
		grMFInputCounter += numpGRfromMFtoGR[i];
	}

	std::cout << "Total MF inputs: " << grMFInputCounter << std::endl;

	// Mossy fiber to Golgi	
	
	for (int i = 0; i < cp.NUM_GO; i++)
	{
		for (int j = 0; j < numpGOfromGLtoGO[i]; j++)
		{
			int glIndex = pGOfromGLtoGO[i][j];
			
			if (haspGLfromMFtoGL[glIndex])
			{
				int mfIndex = pGLfromMFtoGL[glIndex];

				pMFfromMFtoGO[mfIndex][numpMFfromMFtoGO[mfIndex]] = i; 
				numpMFfromMFtoGO[mfIndex]++;			

				pGOfromMFtoGO[i][numpGOfromMFtoGO[i]] = mfIndex;
				numpGOfromMFtoGO[i]++;
			}
		}
	}
}

void InNetConnectivityState::translateGOGL(ConnectivityParams &cp)
{
	for (int i = 0; i < cp.NUM_GR; i++)
	{
		for (int j = 0; j < cp.MAX_NUM_P_GR_FROM_GO_TO_GR; j++)
		{
			for (int k = 0; k < cp.MAX_NUM_P_GL_FROM_GO_TO_GL; k++)
			{	
				if (numpGRfromGOtoGR[i] < cp.MAX_NUM_P_GR_FROM_GO_TO_GR)
				{	
					int glIndex = pGRfromGLtoGR[i][j];
					int goIndex = pGLfromGOtoGL[glIndex][k];
					
					pGOfromGOtoGR[goIndex][numpGOfromGOtoGR[goIndex]] = i; 
					numpGOfromGOtoGR[goIndex]++;			

					pGRfromGOtoGR[i][numpGRfromGOtoGR[i]] = goIndex;
					numpGRfromGOtoGR[i]++;
				}
			}
		}
	}
	int totalGOGR = 0;
	for (int i = 0; i < cp.NUM_GR; i++) totalGOGR += numpGRfromGOtoGR[i];
	
	std::cout << "total GO->GR: " << totalGOGR << std::endl;
	std::cout << "GO->GR Per GR: " << (float)totalGOGR / (float)cp.NUM_GR << std::endl;
}

void InNetConnectivityState::assignGRDelays(unsigned int msPerStep)
{
	for (int i = 0; i < cp.NUM_GR; i++)
	{
		//calculate x coordinate of GR position
		int grPosX = i % cp.GR_X;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		int grBCPCSCDist = abs((int)(cp.GR_X / 2 - grPosX));
		pGRDelayMaskfromGRtoBSP[i] = 0x1 << (int)((grBCPCSCDist / cp.GR_PF_VEL_IN_GR_X_PER_T_STEP
			+ cp.GR_AF_DELAY_IN_T_STEP) / msPerStep);

		for (int j = 0; j < numpGRfromGRtoGO[i]; j++)
		{
			int goPosX = (pGRfromGRtoGO[i][j] % cp.GO_X) * (((float)cp.GR_X) / cp.GO_X);
			int dfromGRtoGO = abs(goPosX - grPosX);

			if (dfromGRtoGO > cp.GR_X / 2)
			{
				if (goPosX < grPosX) dfromGRtoGO = goPosX + cp.GR_X - grPosX;
				else dfromGRtoGO = grPosX + cp.GR_X - goPosX;
			}
			pGRDelayMaskfromGRtoGO[i][j] = 0x1<< (int)((dfromGRtoGO / cp.GR_PF_VEL_IN_GR_X_PER_T_STEP
				+ cp.GR_AF_DELAY_IN_T_STEP) / msPerStep);
		}
	}
}

//void InNetConnectivityState::connectGLUBC()
//{
//	for (int i = 0; i < cp->spanGLtoUBCX + 1; i++)
//	{
//		int ind = cp->spanGLtoUBCX - i;
//		spanArrayGLtoUBCX[i] = (cp->spanGLtoUBCX / 2) - ind;
//	}
//	for (int i = 0; i < cp->spanGLtoUBCY + 1; i++)
//	{
//		int ind = cp->spanGLtoUBCY - i;
//		spanArrayGLtoUBCY[i] = (cp->spanGLtoUBCY / 2) - ind;
//	}
//		
//	for (int i = 0; i < cp->numpGLtoUBC; i++)
//	{
//		xCoorsGLUBC[i] = spanArrayGLtoUBCX[i % (cp->spanGLtoUBCX + 1)];
//		yCoorsGLUBC[i] = spanArrayGLtoUBCY[i / (cp->spanGLtoUBCX + 1)];		
//	}
//
//	// Grid Scale: Complete
//	float gridXScaleSrctoDest = (float)cp->ubcX / (float)cp->glX; 
//	float gridYScaleSrctoDest = (float)cp->ubcY / (float)cp->glY; 
//
//	// Make Random Mossy Fiber Index Array: Complete	
//	std::vector<int> rUBCInd;
//	rUBCInd.assign(cp->numUBC,0);
//	for (int i = 0; i < cp->numUBC; i++)
//	{
//		rUBCInd[i] = i;
//	}
//
//	std::random_shuffle(rUBCInd.begin(), rUBCInd.end());
//
//	//Make Random Span Array: Complete
//	std::vector<int> rUBCSpanInd;
//	rUBCSpanInd.assign(cp->numpGLtoUBC,0);
//	for (int ind = 0; ind < cp->numpGLtoUBC; ind++)
//	{
//		rUBCSpanInd[ind] = ind;
//	}
//	
//	std::random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	
//
//	int srcPosX;
//	int srcPosY;
//	int preDestPosX;
//	int preDestPosY;
//	int tempDestPosX;
//	int tempDestPosY;
//	int destInd;
//	int UBCInputs = 1;
//
//	int glX = cp->glX;
//	int glY = cp->glY;
//
//	for (int attempts = 0; attempts < 4; attempts++)
//	{
//		std::random_shuffle(rUBCInd.begin(), rUBCInd.end());	
//
//		for (int i = 0; i < cp->numUBC; i++)
//		{	
//			//Select MF Coordinates from random index array: Complete	
//			srcPosX = rUBCInd[i] % cp->ubcX;
//			srcPosY = rUBCInd[i] / cp->ubcX;		
//			
//			std::random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	
//
//			for (int j = 0; j < cp->numpGLtoUBC; j++)
//			{	
//				
//				if (numpUBCfromGLtoUBC[rUBCInd[i]] == UBCInputs) break; 
//				
//				preDestPosX = xCoorsGLUBC[rUBCSpanInd[j]]; 
//				preDestPosY = yCoorsGLUBC[rUBCSpanInd[j]];	
//
//				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
//				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;	
//				
//				tempDestPosX = (tempDestPosX%glX + glX) % glX;
//				tempDestPosY = (tempDestPosY%glY+glY) % glY;
//				
//				destInd = tempDestPosY * glX + tempDestPosX;
//					
//				pUBCfromGLtoUBC[rUBCInd[i]] = destInd;
//				numpUBCfromGLtoUBC[rUBCInd[i]]++;
//					
//				pGLfromGLtoUBC[destInd] = rUBCInd[i];
//				numpGLfromGLtoUBC[destInd]++;		
//			}
//		}
//	}
//
//	int totalGLtoUBC = 0;
//	
//	for (int i = 0; i < cp->numUBC; i++)
//	{
//		totalGLtoUBC += numpUBCfromGLtoUBC[i];
//	}
//	std::cout << "Total GL to UBC connections:" << totalGLtoUBC << std::endl; 
//}


//void InNetConnectivityState::connectUBCGL()
//{
//	for (int i = 0; i < cp->spanUBCtoGLX + 1; i++)
//	{
//		int ind = cp->spanUBCtoGLX - i;
//		spanArrayUBCtoGLX[i] = (cp->spanUBCtoGLX / 2) - ind;
//	}
//
//	for (int i = 0; i < cp->spanUBCtoGLY + 1; i++)
//	{
//		int ind = cp->spanUBCtoGLY - i;
//		spanArrayUBCtoGLY[i] = (cp->spanUBCtoGLY / 2) - ind;
//	}
//		
//	for(int i = 0; i < cp->numpUBCtoGL; i++)
//	{
//		xCoorsUBCGL[i] = spanArrayUBCtoGLX[i % (cp->spanUBCtoGLX + 1)];
//		yCoorsUBCGL[i] = spanArrayUBCtoGLY[i / (cp->spanUBCtoGLX + 1)];		
//	}
//
//	// Grid Scale: Complete
//	float gridXScaleSrctoDest = (float)cp->ubcX / (float)cp->glX; 
//	float gridYScaleSrctoDest = (float)cp->ubcY / (float)cp->glY; 
//
//	//Make Random Mossy Fiber Index Array: Complete	
//	std::vector<int> rUBCInd;
//	rUBCInd.assign(cp->numUBC,0);
//	
//	for (int i = 0; i < cp->numUBC; i++)
//	{
//		rUBCInd[i] = i;
//	}
//	std::random_shuffle(rUBCInd.begin(), rUBCInd.end());
//
//	//Make Random Span Array: Complete
//	std::vector<int> rUBCSpanInd;
//	rUBCSpanInd.assign(cp->numpUBCtoGL,0);
//	for (int ind = 0; ind < cp->numpUBCtoGL; ind++) rUBCSpanInd[ind] = ind;
//
//	int srcPosX;
//	int srcPosY;
//	int preDestPosX;
//	int preDestPosY;
//	int tempDestPosX;
//	int tempDestPosY;
//	int destInd;
//
//	int glX = cp->glX;
//	int glY = cp->glY;
//
//	int UBCOutput = 10;
//
//	for (int attempts = 0; attempts < 3; attempts++)
//	{
//		std::random_shuffle(rUBCInd.begin(), rUBCInd.end());	
//
//		for (int i = 0; i < cp->numUBC; i++)
//		{	
//			//Select MF Coordinates from random index array: Complete	
//			srcPosX = rUBCInd[i] % cp->ubcX;
//			srcPosY = rUBCInd[i] / cp->ubcX;		
//
//			std::random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	
//			
//			for (int j = 0; j < cp->numpUBCtoGL; j++)
//			{	
//				preDestPosX = xCoorsUBCGL[rUBCSpanInd[j]]; 
//				preDestPosY = yCoorsUBCGL[rUBCSpanInd[j]];	
//
//				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
//				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;	
//				
//				tempDestPosX = ((tempDestPosX%glX + glX) % glX);
//				tempDestPosY = ((tempDestPosY%glY + glY) % glY);
//				
//				destInd = tempDestPosY*glX+tempDestPosX;
//				
//				if (destInd == pUBCfromGLtoUBC[rUBCInd[i]] ||
//						numpUBCfromUBCtoGL[rUBCInd[i]] == UBCOutput) break; 
//				
//				if (numpGLfromUBCtoGL[destInd] == 0) 
//				{	
//					pUBCfromUBCtoGL[ rUBCInd[i] ][ numpUBCfromUBCtoGL[rUBCInd[i]] ] = destInd;
//					numpUBCfromUBCtoGL[ rUBCInd[i] ]++;	
//
//					pGLfromUBCtoGL[destInd][ numpGLfromUBCtoGL[destInd] ] = rUBCInd[i];
//					numpGLfromUBCtoGL[destInd]++;	
//				
//				}
//			}
//		}
//
//	}	
//}

//void InNetConnectivityState::connectMFGL_withUBC(CRandomSFMT *randGen)
//{
//
//	int initialMFOutput = 14;
//
//	//Make Span Array: Complete	
//	for (int i = 0; i < cp->spanMFtoGLX + 1;i++)
//	{
//		int ind = cp->spanMFtoGLX - i;
//		spanArrayMFtoGLX[i] = (cp->spanMFtoGLX / 2) - ind;
//	}
//
//	for(int i = 0; i < cp->spanMFtoGLY + 1; i++)
//	{
//		int ind = cp->spanMFtoGLY - i;
//		spanArrayMFtoGLY[i] = (cp->spanMFtoGLY / 2) - ind;
//	}
//		
//	for (int i = 0; i < cp->numpMFtoGL; i++)
//	{
//		xCoorsMFGL[i] = spanArrayMFtoGLX[i % (cp->spanMFtoGLX + 1)];
//		yCoorsMFGL[i] = spanArrayMFtoGLY[i / (cp->spanMFtoGLX + 1)];		
//	}
//
//	// Grid Scale: Complete
//	float gridXScaleSrctoDest = (float)cp->mfX / (float)cp->glX; 
//	float gridYScaleSrctoDest = (float)cp->mfY / (float)cp->glY; 
//
//	//Make Random Mossy Fiber Index Array: Complete	
//	std::vector<int> rMFInd;
//	rMFInd.assign(cp->numMF,0);
//	
//	for (int i = 0; i < cp->numMF; i++)
//	{
//		rMFInd[i] = i;	
//	}
//
//	std::random_shuffle(rMFInd.begin(), rMFInd.end());
//
//	//Make Random Span Array: Complete
//	std::vector<int> rMFSpanInd;
//	rMFSpanInd.assign(cp->numpMFtoGL,0);
//	
//	for (int ind = 0; ind < cp->numpMFtoGL; ind++) rMFSpanInd[ind] = ind;
//
//	int srcPosX;
//	int srcPosY;
//	int preDestPosX;
//	int preDestPosY;
//	int tempDestPosX;
//	int tempDestPosY;
//	int destInd;
//
//	int glX = cp->glX;
//	int glY = cp->glY;
//	int shitCounter;
//	
//	for(int attempts = 0; attempts < 3; attempts++)
//	{
//		std::random_shuffle(rMFInd.begin(), rMFInd.end());	
//
//		for (int i = 0; i < cp->numMF; i++)
//		{	
//			//Select MF Coordinates from random index array: Complete	
//			srcPosX = rMFInd[i] % cp->mfX;
//			srcPosY = rMFInd[i] / cp->mfX;		
//			
//			std::random_shuffle(rMFSpanInd.begin(), rMFSpanInd.end());	
//			
//			for (int j = 0; j < cp->numpMFtoGL; j++)
//			{	
//				preDestPosX = xCoorsMFGL[ rMFSpanInd[j] ]; 
//				preDestPosY = yCoorsMFGL[ rMFSpanInd[j] ];	
//
//				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
//				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;	
//				
//				tempDestPosX = (tempDestPosX%glX+glX) % glX;
//				tempDestPosY = (tempDestPosY%glY+glY) % glY;
//				
//				destInd = tempDestPosY * glX + tempDestPosX;
//				
//				if ( numpMFfromMFtoGL[rMFInd[i]] == initialMFOutput + attempts ) break;
//					
//				if ( !haspGLfromMFtoGL[destInd] && numpGLfromUBCtoGL[destInd] == 0 ) 
//				{	
//					pMFfromMFtoGL[rMFInd[i]][numpMFfromMFtoGL[rMFInd[i]]] = destInd;
//					numpMFfromMFtoGL[rMFInd[i]]++;
//					
//					pGLfromMFtoGL[destInd] = rMFInd[i];
//					haspGLfromMFtoGL[destInd] = true;	
//				}
//			}
//		}
//		
//		shitCounter = 0;
//		
//		for (int i = 0; i < cp->numGL; i++)
//		{
//			if (!haspGLfromMFtoGL[i]) shitCounter++;
//		}
//	}	
//	
//	std::cout << "Empty Glomeruli Counter: " << shitCounter << std::endl << std::endl;
//	
//	int count = 0;
//	
//	for (int i = 0; i < cp->numMF; i++)
//	{
//		count += numpMFfromMFtoGL[i];
//	}
//
//	std::cout << "Total number of Mossy Fiber to Glomeruli connections:	" << count << std::endl;
//	std::cout << "Correct number: " << cp->numGL << std::endl;
//
//	count = 0;
//
//	for (int i = 0; i < cp->numMF; i++)
//	{
//		for (int j = 0; j < numpMFfromMFtoGL[i]; j++)
//		{
//			for (int k = 0; k < numpMFfromMFtoGL[i]; k++)
//			{
//				if (pMFfromMFtoGL[i][j] == pMFfromMFtoGL[i][k] && j != k) count++; 
//			}
//		}
//	}
//
//	std::cout << "Double Mossy Fiber to Glomeruli connecitons: " << count << std::endl;
//}

//void InNetConnectivityState::translateUBCGL()
//{
//
//	int numUBCfromUBCtoGL = 10;
//	//UBC to GR 
//	int grIndex;
//	for (int i = 0; i < cp->numUBC; i++)
//	{
//		for (int j = 0; j < numUBCfromUBCtoGL; j++)
//		{
//			glIndex = pUBCfromUBCtoGL[i][j]; 
//			
//			for (int k = 0; k < numpGLfromGLtoGR[glIndex]; k++)
//			{
//				grIndex = pGLfromGLtoGR[glIndex][k];
//
//				pUBCfromUBCtoGR[i][numpUBCfromUBCtoGR[i]] = grIndex; 
//				numpUBCfromUBCtoGR[i]++;			
//
//				pGRfromUBCtoGR[grIndex][numpGRfromUBCtoGR[grIndex]] = i;
//				numpGRfromUBCtoGR[grIndex]++;
//			}
//		}
//	}
//
//	std::ofstream fileUBCGRconIn;
//	fileUBCGRconIn.open("UBCGRInputcon.txt");
//	
//	for (int i = 0; i < cp->numGR; i++)
//	{
//		for (int j = 0; j < cp->maxnumpGRfromGLtoGR; j++)
//		{
//			fileUBCGRconIn << pGRfromUBCtoGR[i][j] << " ";
//		}
//
//		fileUBCGRconIn << std::endl;
//	}
//
//	int grUBCInputCounter = 0;
//	
//	for (int i = 0; i < cp->numGR; i++) grUBCInputCounter += numpGRfromUBCtoGR[i];
//	std::cout << "Total UBC inputs: " << grUBCInputCounter << std::endl;
//
//	//UBC to GO
//	int goIndex;
//	
//	for (int i = 0; i < cp->numUBC; i++)
//	{
//		for (int j = 0; j < numUBCfromUBCtoGL; j++)
//		{
//			glIndex = pUBCfromUBCtoGL[i][j]; 
//			
//			for (int k = 0; k < numpGLfromGLtoGO[glIndex]; k++)
//			{
//				goIndex = pGLfromGLtoGO[glIndex][k];
//
//				pUBCfromUBCtoGO[i][numpUBCfromUBCtoGO[i]] = goIndex; 
//				numpUBCfromUBCtoGO[i]++;			
//
//				pGOfromUBCtoGO[goIndex][numpGOfromUBCtoGO[goIndex]] = i;
//				numpGOfromUBCtoGO[goIndex]++;
//			}
//		}
//	}
//	
//	std::ofstream fileUBCGOconIn;
//	fileUBCGOconIn.open("UBCGOInputcon.txt");
//	for (int i = 0; i < cp->numGO; i++)
//	{
//		for (int j = 0; j < 16; j++) fileUBCGOconIn << pGOfromUBCtoGO[i][j] << " ";
//		fileUBCGOconIn << std::endl;
//	}
//
//	//UBC to UBC
//	int ubcIndex;
//
//	for (int i = 0; i < cp->numUBC; i++)
//	{
//		for (int j = 0; j < numUBCfromUBCtoGL; j++)
//		{
//			glIndex = pUBCfromUBCtoGL[i][j]; 
//			
//			if (numpGLfromGLtoUBC[glIndex] == 1)
//			{
//				ubcIndex = pGLfromGLtoUBC[glIndex];
//
//				pUBCfromUBCOutUBC[i][numpUBCfromUBCOutUBC[i]] = ubcIndex; 
//				numpUBCfromUBCOutUBC[i]++;			
//
//				pUBCfromUBCInUBC[ubcIndex][numpUBCfromUBCInUBC[ubcIndex]] = i;
//				numpUBCfromUBCInUBC[ubcIndex]++;
//			}
//		}
//	}
//}

//void InNetConnectivityState::connectGOGO(CRandomSFMT *randGen)
//{
//	int spanGOtoGOsynX = 12;
//	int spanGOtoGOsynY = 12;
//	int numpGOtoGOsyn = (spanGOtoGOsynX + 1) * (spanGOtoGOsynY + 1); 
//
//	//Make Span Array: Complete	
//	for (int i = 0; i < spanGOtoGOsynX + 1;i++)
//	{
//		spanArrayGOtoGOsynX[i] = (spanGOtoGOsynX / 2) - (spanGOtoGOsynX - i);
//	}
//
//	for (int i = 0; i < spanGOtoGOsynY + 1; i++)
//	{
//		int ind = spanGOtoGOsynY - i;
//		spanArrayGOtoGOsynY[i] = (spanGOtoGOsynY / 2) - (spanGOtoGOsynY - i);
//	}
//		
//	for (int i = 0; i < numpGOtoGOsyn; i++)
//	{
//		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (spanGOtoGOsynX + 1)];
//		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (spanGOtoGOsynX + 1)];		
//	}
//
//	//Make Random Mossy Fiber Index Array: Complete	
//	std::vector<int> rGOInd;
//	rGOInd.assign(cp->numGO,0);
//	
//	for (int i = 0; i < cp->numGO; i++) rGOInd[i] = i;	
//	
//	std::random_shuffle(rGOInd.begin(), rGOInd.end());
//
//	//Make Random Span Array
//	std::vector<int> rGOGOSpanInd;
//	rGOGOSpanInd.assign(numpGOtoGOsyn,0);
//	for (int ind = 0; ind < numpGOtoGOsyn; ind++) rGOGOSpanInd[ind] = ind;
//
//	int srcPosX;
//	int srcPosY;
//	int preDestPosX;
//	int preDestPosY;
//	int tempDestPosX;
//	int tempDestPosY;
//	int destInd;
//
//	int numOutputs = cp->numConGOGO;
//	int conType = 2;
//
//	for (int attempt = 1; attempt < numOutputs + 1; attempt++)
//	{
//		for (int i = 0; i < cp->numGO; i++)
//		{	
//			srcPosX = rGOInd[i] % cp->goX;
//			srcPosY = rGOInd[i] / cp->goX;	
//			
//			std::random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
//			
//			for (int j = 0; j < numpGOtoGOsyn; j++)
//			{	
//				preDestPosX = xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
//				preDestPosY = yCoorsGOGOsyn[rGOGOSpanInd[j]];	
//
//				tempDestPosX = srcPosX + preDestPosX;
//				tempDestPosY = srcPosY + preDestPosY;
//
//				tempDestPosX = (tempDestPosX % cp->goX + cp->goX) % cp->goX;
//				tempDestPosY = (tempDestPosY % cp->goY + cp->goY) % cp->goY;
//
//				destInd = tempDestPosY * cp->goX + tempDestPosX;
//					
//				if (conType == 0)
//				{ 	
//					// Normal random connectivity		
//					if (numpGOGABAOutGOGO[rGOInd[i]] == numOutputs) break;
//
//					// conditional statment blocking the ability to make two outputs to the same cell
//					pGOGABAOutGOGO[rGOInd[i]][numpGOGABAOutGOGO[rGOInd[i]]] = destInd;
//					numpGOGABAOutGOGO[rGOInd[i]]++;
//					
//					pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = rGOInd[i];
//					numpGOGABAInGOGO[destInd]++;
//				}
//				else if (conType == 1){	
//					// 100% reciprocal	
//					if (!conGOGOBoolOut[rGOInd[i]][destInd]&&
//						(numpGOGABAOutGOGO[rGOInd[i]] < attempt ) &&
//						(numpGOGABAInGOGO[rGOInd[i]] < attempt) &&
//						(numpGOGABAOutGOGO[destInd] < attempt) &&
//						(numpGOGABAInGOGO[destInd] < attempt) &&
//						(destInd != rGOInd[i])) 
//					{	
//						pGOGABAOutGOGO[rGOInd[i]][numpGOGABAOutGOGO[rGOInd[i]]] = destInd;
//						numpGOGABAOutGOGO[rGOInd[i]]++;
//						pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = rGOInd[i];
//						numpGOGABAOutGOGO[destInd]++;
//					
//						pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = rGOInd[i];
//						numpGOGABAInGOGO[destInd]++;
//						pGOGABAInGOGO[rGOInd[i]][numpGOGABAInGOGO[rGOInd[i]]] = destInd;
//						numpGOGABAInGOGO[rGOInd[i]]++;
//				
//						conGOGOBoolOut[rGOInd[i]][destInd] = true;
//					}
//				}
//			
//				else if (conType == 2)
//				{	
//					// variable %reciprocal	
//					if (!conGOGOBoolOut[rGOInd[i]][destInd] &&
//						(numpGOGABAOutGOGO[rGOInd[i]] < attempt) &&
//						(numpGOGABAInGOGO[rGOInd[i]] < attempt) &&
//						(numpGOGABAOutGOGO[destInd] < attempt) &&
//						(numpGOGABAInGOGO[destInd] < attempt) &&
//						destInd != rGOInd[i]) 
//					{	
//						pGOGABAOutGOGO[rGOInd[i]][numpGOGABAOutGOGO[rGOInd[i]]] = destInd;
//						numpGOGABAOutGOGO[rGOInd[i]]++;
//					
//						pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = rGOInd[i];
//						numpGOGABAInGOGO[destInd]++;
//						
//						if (randGen->Random() >= 0.95)
//						{
//							pGOGABAInGOGO[rGOInd[i]][numpGOGABAInGOGO[rGOInd[i]]] = destInd;
//							numpGOGABAInGOGO[rGOInd[i]]++;
//							pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = rGOInd[i];
//							numpGOGABAOutGOGO[destInd]++;
//				
//							conGOGOBoolOut[rGOInd[i]][destInd] = true;
//						}
//					}
//				}
//			}
//		}
//	}	
//	
//	std::ofstream fileGOGOconIn;
//	fileGOGOconIn.open("GOGOInputcon.txt");
//	
//	for (int i = 0; i < cp->numGO; i++)
//	{
//		for (int j = 0; j < 30; j++)
//		{
//			fileGOGOconIn << pGOGABAInGOGO[i][j] << " ";
//		}
//
//		fileGOGOconIn << std::endl;
//	}
//	
//	std::ofstream fileGOGOconOut;
//	fileGOGOconOut.open("GOGOOutputcon.txt");
//	
//	for (int i = 0; i < cp->numGO; i++)
//	{
//		for (int j = 0; j < 30; j++)
//		{
//			fileGOGOconOut << pGOGABAOutGOGO[i][j] << " ";
//		}
//		fileGOGOconOut << std::endl;
//	}
//
//	if (conType == 0)
//	{
//		for (int k = 0; k < 20; k++)
//		{
//			std::cout << "numpGOGABAInGOGO[" << k << "]: " << numpGOGABAInGOGO[k] << std::endl;
//			for (int j = 0; j < numpGOGABAInGOGO[k]; j++)
//			{
//				std::cout << pGOGABAInGOGO[k][j] << " ";
//			}
//			std::cout << std::endl;
//		}
//	}
//
//	else if (conType == 1)
//	{	
//		int pNonRecip = 0;
//		int missedOutCon = 0;
//		int missedInCon = 0;
//
//		for (int i = 0; i < cp->numGO; i++)
//		{
//			if (numpGOGABAInGOGO[i] != numpGOGABAOutGOGO[i]) pNonRecip++;
//			if (numpGOGABAInGOGO[i] != numOutputs) missedInCon++;
//			if (numpGOGABAOutGOGO[i] != numOutputs) missedOutCon++;
//		}
//
//		std::cout << "Potential non-reciprocal connection: " << pNonRecip << std::endl;
//		std::cout << "Missing Input: " << missedInCon << std::endl;
//		std::cout << "Missing Output: " << missedOutCon << std::endl;
//	}
//}

//void InNetConnectivityState::connectGOGOBias(CRandomSFMT *randGen)
//{
//	int spanGOtoGOsynX = 12;
//	int spanGOtoGOsynY = 12;
//	int numpGOtoGOsyn = (spanGOtoGOsynX + 1) *(spanGOtoGOsynY + 1); 
//
//	//Make Span Array: Complete	
//	for (int i = 0; i < spanGOtoGOsynX +1;i++)
//	{
//		spanArrayGOtoGOsynX[i] = (spanGOtoGOsynX / 2) - (spanGOtoGOsynX - i);
//	}
//
//	for (int i =0; i <spanGOtoGOsynY +1; i++)
//	{
//		spanArrayGOtoGOsynY[i] = (spanGOtoGOsynY / 2) - (spanGOtoGOsynY - i);
//	}
//		
//	for (int i = 0; i < numpGOtoGOsyn; i++)
//	{
//		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (spanGOtoGOsynX + 1)];
//		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (spanGOtoGOsynX + 1)];		
//	}
//
//	//Make Random Mossy Fiber Index Array: Complete	
//	std::vector<int> rGOInd;
//	rGOInd.assign(cp->numGO, 0);
//	
//	for (int i = 0; i < cp->numGO; i++) rGOInd[i] = i;	
//	
//	std::random_shuffle(rGOInd.begin(), rGOInd.end());
//
//	//Make Random Span Array
//	std::vector<int> rGOGOSpanInd;
//	rGOGOSpanInd.assign(numpGOtoGOsyn,0);
//	for (int ind = 0; ind < numpGOtoGOsyn; ind++) rGOGOSpanInd[ind] = ind;
//
//	int srcPosX;
//	int srcPosY;
//	int preDestPosX;
//	int preDestPosY;
//	int tempDestPosX;
//	int tempDestPosY;
//	int destInd;
//
//	int numOutputs = cp->numConGOGO;
//	int conType = 0;
//
//	for (int i = 0; i < cp->numGO; i++)
//	{	
//		srcPosX = rGOInd[i] % cp->goX;
//		srcPosY = rGOInd[i] / cp->goX;	
//		
//		std::random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
//			
//		for (int j = 0; j < numpGOtoGOsyn; j++)
//		{	
//			preDestPosX = xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
//			preDestPosY = yCoorsGOGOsyn[rGOGOSpanInd[j]];	
//
//			tempDestPosX = srcPosX + preDestPosX;
//			tempDestPosY = srcPosY + preDestPosY;
//
//			tempDestPosX = (tempDestPosX % cp->goX + cp->goX) % cp->goX;
//			tempDestPosY = (tempDestPosY % cp->goY + cp->goY) % cp->goY;
//			
//			destInd = tempDestPosY * cp->goX + tempDestPosX;
//				
//			if (randGen->Random() >= 0.9695 && !conGOGOBoolOut[rGOInd[i]][destInd])
//			{
//				pGOGABAOutGOGO[rGOInd[i]][numpGOGABAOutGOGO[rGOInd[i]]] = destInd;
//				numpGOGABAOutGOGO[rGOInd[i]]++;
//					
//				pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = rGOInd[i];
//				numpGOGABAInGOGO[destInd]++;	
//					
//				conGOGOBoolOut[rGOInd[i]][destInd] = true;
//			
//				// conditional statement against making double output
//
//				if (randGen->Random() > 0 && !conGOGOBoolOut[destInd][rGOInd[i]])
//				{
//					pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = rGOInd[i];
//					numpGOGABAOutGOGO[destInd]++;
//
//					pGOGABAInGOGO[rGOInd[i]][numpGOGABAInGOGO[rGOInd[i]]] = destInd;
//					numpGOGABAInGOGO[rGOInd[i]]++;
//					
//					conGOGOBoolOut[destInd][rGOInd[i]] = true;
//				}
//			}	
//		}
//	}
//	
//	if (conType == 0)
//	{
//		for (int k = 0; k < 20; k++)
//		{
//			std::cout << "numpGOGABAInGOGO[" << k << "]: " << numpGOGABAInGOGO[k] << std::endl;
//			for (int j = 0; j < numpGOGABAInGOGO[k]; j++) std::cout << pGOGABAInGOGO[k][j] << " ";
//			std::cout << std::endl;
//		}
//	}
//	else if (conType == 1)
//	{	
//		int pNonRecip 	 = 0;
//		int missedOutCon = 0;
//		int missedInCon  = 0;
//
//		for (int i = 0; i < cp->numGO; i++)
//		{
//
//			if (numpGOGABAInGOGO[i]  != numpGOGABAOutGOGO[i]) pNonRecip++; 
//			if (numpGOGABAInGOGO[i]  != numOutputs) missedInCon++; 
//			if (numpGOGABAOutGOGO[i] != numOutputs) missedOutCon++; 
//		}
//
//		std::cout << "Potential non-reciprocal connection: " << pNonRecip << std::endl;
//		std::cout << "Missing Input: " << missedInCon << std::endl;
//		std::cout << "Missing Output: " << missedOutCon << std::endl;
//	}
//}

//void InNetConnectivityState::connectGOGODecay(CRandomSFMT *randGen)
//{
//	float A 		 = 0.01;
//	float pRecipGOGO = 1;
//	float PconX;
//	float PconY;
//
//	std::cout << cp->spanGOGOsynX << std::endl;
//	std::cout << cp->spanGOGOsynY << std::endl;
//	std::cout << cp->sigmaGOGOsynML << std::endl;
//	std::cout << cp->sigmaGOGOsynS << std::endl;
//
//	std::cout << cp->pRecipGOGOsyn << std::endl;
//	std::cout << cp->maxGOGOsyn << std::endl;
//
//	for (int i = 0; i < cp->spanGOGOsynX + 1; i++)
//	{
//		int ind = cp->spanGOGOsynX - i;
//		spanArrayGOtoGOsynX[i] = (cp->spanGOGOsynX / 2) - ind;
//	}
//
//	for (int i = 0; i < cp->spanGOGOsynY + 1; i++)
//	{
//		int ind = cp->spanGOGOsynY - i;
//		spanArrayGOtoGOsynY[i] = (cp->spanGOGOsynY / 2) - ind;
//	}
//		
//	for (int i = 0; i < cp->numpGOGOsyn; i++)
//	{
//		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (cp->spanGOGOsynX + 1)];
//		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (cp->spanGOGOsynX+1)];		
//	}
//
//	for (int i = 0; i < cp->numpGOGOsyn; i++)
//	{
//		PconX = (xCoorsGOGOsyn[i] * xCoorsGOGOsyn[i]) / (2 * cp->sigmaGOGOsynML * cp->sigmaGOGOsynML);
//		PconY = (yCoorsGOGOsyn[i] * yCoorsGOGOsyn[i]) / (2 * cp->sigmaGOGOsynS * cp->sigmaGOGOsynS);
//		Pcon[i] = A * exp(-(PconX + PconY));
//	}
//	
//	// Remove self connection 
//	for (int i = 0; i < cp->numpGOGOsyn; i++)
//	{
//		if ((xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0)) Pcon[i] = 0;
//	}
//	
//	std::vector<int> rGOGOSpanInd;
//	rGOGOSpanInd.assign(cp->numpGOGOsyn,0);
//	for (int ind = 0; ind < cp->numpGOGOsyn; ind++) rGOGOSpanInd[ind] = ind;
//	
//	int srcPosX;
//	int srcPosY;
//	int preDestPosX;
//	int preDestPosY;
//	int tempDestPosX;
//	int tempDestPosY;
//	int destInd;
//
//	for (int atmpt = 0; atmpt <200; atmpt++)
//	{
//		for (int i = 0; i < cp->numGO; i++)
//		{	
//			srcPosX = i % cp->goX;
//			srcPosY = i / cp->goX;	
//			
//			std::random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
//			
//			for (int j = 0; j < cp->numpGOGOsyn; j++)
//			{	
//				
//				preDestPosX = xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
//				preDestPosY = yCoorsGOGOsyn[rGOGOSpanInd[j]];	
//
//				tempDestPosX = srcPosX + preDestPosX;
//				tempDestPosY = srcPosY + preDestPosY;
//
//				tempDestPosX = (tempDestPosX % cp->goX + cp->goX) % cp->goX;
//				tempDestPosY = (tempDestPosY % cp->goY + cp->goY) % cp->goY;
//						
//				destInd = tempDestPosY * cp->goX + tempDestPosX;
//
//				if (randGen->Random()>=1 - Pcon[rGOGOSpanInd[j]] && !conGOGOBoolOut[i][destInd] &&
//					   numpGOGABAOutGOGO[i] < cp->maxGOGOsyn && numpGOGABAInGOGO[destInd] < cp->maxGOGOsyn)
//				{	
//					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destInd;
//					numpGOGABAOutGOGO[i]++;
//					
//					pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = i;
//					numpGOGABAInGOGO[destInd]++;
//					
//					conGOGOBoolOut[i][destInd] = true;
//					
//					if (randGen->Random() >= 1 - pRecipGOGO && !conGOGOBoolOut[destInd][i] &&
//							numpGOGABAOutGOGO[destInd] < cp->maxGOGOsyn && numpGOGABAInGOGO[i] < cp->maxGOGOsyn)
//					{
//						pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = i;
//						numpGOGABAOutGOGO[destInd]++;
//
//						pGOGABAInGOGO[i][numpGOGABAInGOGO[i]] = destInd;
//						numpGOGABAInGOGO[i]++;
//						
//						conGOGOBoolOut[destInd][i] = true;
//					}
//				}
//			}
//		}
//	}
//
//	std::ofstream fileGOGOconIn;
//	fileGOGOconIn.open("GOGOInputcon.txt");
//	
//	for (int i = 0; i < cp->numGO; i++)
//	{
//		for (int j = 0; j < cp->maxGOGOsyn; j++)
//		{
//			fileGOGOconIn << pGOGABAInGOGO[i][j] << " ";
//		}
//
//		fileGOGOconIn << std::endl;
//	}
//	
//	std::ofstream fileGOGOconOut;
//	fileGOGOconOut.open("GOGOOutputcon.txt");
//	
//	for (int i = 0; i < cp->numGO; i++)
//	{
//		for (int j = 0; j < cp->maxGOGOsyn; j++)
//		{
//			fileGOGOconOut << pGOGABAOutGOGO[i][j] << " ";
//		}
//
//		fileGOGOconOut << std::endl;
//	}
//
//	float totalGOGOcons = 0;
//	
//	for (int i = 0; i < cp->numGO; i++)
//	{
//		totalGOGOcons += numpGOGABAInGOGO[i];
//	}
//
//	std::cout << "Total GOGO connections: " << totalGOGOcons << std::endl;
//	std::cout << "Average GOGO connections:	" << totalGOGOcons / float(cp->numGO) << std::endl;
//	std::cout << cp->numGO << std::endl;
//	int recipCounter = 0;
//	
//	for (int i = 0; i < cp->numGO; i++)
//	{
//		for (int j = 0; j < numpGOGABAInGOGO[i]; j++)
//		{
//			for (int k = 0; k < numpGOGABAOutGOGO[i]; k++)
//			{
//				if (pGOGABAInGOGO[i][j] == pGOGABAOutGOGO[i][k] && pGOGABAInGOGO[i][j] != UINT_MAX
//						&& pGOGABAOutGOGO[i][k] != UINT_MAX) recipCounter++; 
//			}
//		}
//	}
//
//	float fracRecip = recipCounter / totalGOGOcons;
//	std::cout << "FracRecip: " << fracRecip << std::endl;
//	rGOGOSpanInd.clear();
//}


//void InNetConnectivityState::connectPFtoBC()
//{
//	int spanPFtoBCX = cp->grX;
//	int spanPFtoBCY = cp->grY / cp->numBC;
//	int numpPFtoBC = (spanPFtoBCX + 1) * (spanPFtoBCY + 1);
//
//	for (int i = 0; i < spanPFtoBCX + 1; i++)
//	{
//		spanArrayPFtoBCX[i] = (spanPFtoBCX / 2) - (spanPFtoBCX - i);
//	}
//
//	for (int i = 0; i < spanPFtoBCY + 1; i++)
//	{
//		int ind = spanPFtoBCY - i;
//		spanArrayPFtoBCY[i] = (spanPFtoBCY / 2) - (spanPFtoBCY - i);
//	}
//	
//	for (int i = 0; i < numpPFtoBC; i++)
//	{
//		xCoorsPFBC[i] = spanArrayPFtoBCX[i % (spanPFtoBCX + 1)];
//		yCoorsPFBC[i] = spanArrayPFtoBCY[i / (spanPFtoBCX + 1)];
//	}
//
//	//Random Span Array
//	std::vector<int> rPFBCSpanInd;
//	rPFBCSpanInd.assign(numpPFtoBC, 0);
//	
//	for (int ind = 0; ind < numpPFtoBC; ind++) rPFBCSpanInd[ind] = ind;
//
//	float gridXScaleSrctoDest = 1;
//	float gridYScaleSrctoDest = (float)cp->numBC / cp->grY;
//
//	int srcPosX;
//	int srcPosY;
//	int preDestPosX;
//	int preDestPosY;
//	int tempDestPosX;
//	int tempDestPosY;
//	int destInd;
//
//	for (int i =0; i < cp->numBC; i++)
//	{
//		srcPosX = cp->grX / 2;
//		srcPosY = i;
//
//		std::random_shuffle(rPFBCSpanInd.begin(), rPFBCSpanInd.end());
//		
//		for (int j = 0; j < 5000; j++)
//		{
//			preDestPosX = xCoorsPFBC[rPFBCSpanInd[j]];
//			preDestPosY = yCoorsPFBC[rPFBCSpanInd[j]];
//			
//			tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
//			tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;
//			
//			tempDestPosX = (tempDestPosX%cp->grX + cp->grX) % cp->grX;
//			tempDestPosY = (tempDestPosY%cp->grY + cp->grY) % cp->grY;
//
//			destInd = tempDestPosY * cp->grX + tempDestPosX;	
//
//			pBCfromPFtoBC[i][numpBCfromPFtoBC[i]] = destInd;
//			numpBCfromPFtoBC[i]++;
//
//			pGRfromPFtoBC[destInd][numpGRfromPFtoBC[destInd]] = i;
//			numpGRfromPFtoBC[destInd]++;
//		}
//	}
//}

//void InNetConnectivityState::assignPFtoBCDelays(unsigned int msPerStep)
//{
//	
//	for (int i = 0; i < cp->numGR; i++)
//	{
//		int grPosX;
//
//		//calculate x coordinate of GR position
//		grPosX=i%cp->grX;
//
//		for (int j = 0; j < numpGRfromPFtoBC[i]; j++)
//		{
//			int dfromGRtoBC;
//			int bcPosX;
//
//			bcPosX = cp->grX / 2; 
//			dfromGRtoBC=abs(bcPosX-grPosX);
//		}
//	}
//}

void InNetConnectivityState::connectCommon(int **srcConArr, int *srcNumCon,
		int **destConArr, int *destNumCon, int srcMaxNumCon, int numSrcCells,
		int destMaxNumCon, int destNormNumCon, int srcGridX, int srcGridY,
		int destGridX, int destGridY, int srcSpanOnDestGridX, int srcSpanOnDestGridY,
		int normConAttempts, int maxConAttempts, bool needUnique)
{
	float gridXScaleStoD;
	float gridYScaleStoD;

	gridXScaleStoD = (float)srcGridX / (float)destGridX;
	gridYScaleStoD = (float)srcGridY / (float)destGridY;

	bool *srcConnected = new bool[numSrcCells];

	std::cout << "srcMaxNumCon: " << srcMaxNumCon <<" numSrcCells: " << numSrcCells << std::endl;
	std::cout << "destMaxNumCon " << destMaxNumCon <<" destNormNumCon "<< destNormNumCon << std::endl;
	std::cout << "srcGridX: " << srcGridX <<" srcGridY: "<< srcGridY << " destGridX: "<< destGridX << 
		" destGridY: " << destGridY << std::endl;
	std::cout << "srcSpanOnDestGridX: " << srcSpanOnDestGridX << " srcSpanOnDestGridY: " <<
		srcSpanOnDestGridY << std::endl;
	std::cout << "gridXScaleStoD: " << gridXScaleStoD << " gridYScaleStoD: "<< gridYScaleStoD << std::endl;

	for (int i = 0; i < srcMaxNumCon; i++)
	{
		// NOTE: watch out with initialization with memset. There are better ways.	
		int srcNumConnected = 0;
		memset(srcConnected, false, numSrcCells*sizeof(bool));

		std::cout << "i: " << i << std::endl;

		while (srcNumConnected < numSrcCells)
		{
			int srcInd;
			int srcPosX;
			int srcPosY;
			int attempts;
			int tempDestNumConLim;
			bool complete;

			srcInd = randGen->IRandom(0, numSrcCells - 1);

			if (srcConnected[srcInd])
			{
				continue;
			}
			
			srcConnected[srcInd] = true;
			srcNumConnected++;

			srcPosX = srcInd % srcGridX;
			srcPosY = (int)(srcInd / srcGridX);

			tempDestNumConLim = destNormNumCon;

			for (attempts = 0; attempts < maxConAttempts; attempts++)
			{
				int tempDestPosX;
				int tempDestPosY;
				int derivedDestInd;

				if (attempts == normConAttempts) tempDestNumConLim = destMaxNumCon;

				tempDestPosX = (int)round(srcPosX / gridXScaleStoD);
				tempDestPosY = (int)round(srcPosY / gridXScaleStoD);

				tempDestPosX += round((randGen->Random() - 0.5) * srcSpanOnDestGridX);
				tempDestPosY += round((randGen->Random() - 0.5) * srcSpanOnDestGridY);

				tempDestPosX = ((tempDestPosX % destGridX + destGridX) % destGridX);
				tempDestPosY = ((tempDestPosY % destGridY + destGridY) % destGridY);

				derivedDestInd = tempDestPosY *destGridX + tempDestPosX;

				if (needUnique)
				{
					//NOTE: JUST USE A SET!	
					bool unique = true;
					
					for (int j = 0; j < i; j++)
					{
						if (derivedDestInd == srcConArr[srcInd][j])
						{
							unique = false;
							break;
						}
					}

					if (!unique) continue;
				}

				if (destNumCon[derivedDestInd] < tempDestNumConLim)
				{
					destConArr[derivedDestInd][destNumCon[derivedDestInd]] = srcInd;
					destNumCon[derivedDestInd]++;
					srcConArr[srcInd][i] = derivedDestInd;
					srcNumCon[srcInd]++;
					break;
				}
			}
		}
	}

	delete[] srcConnected;
}

//void InNetConnectivityState::translateCommon(int **pPreGLConArr, int *numpPreGLCon,
//		int **pGLPostGLConArr, int *numpGLPostGLCon, int **pPreConArr, int *numpPreCon,
//		int **pPostConArr, int *numpPostCon, int numPre)
//{
//	
//	for (int i = 0; i < numPre; i++)
//	{
//		numpPreCon[i] = 0;
//
//		for (int j = 0; j < numpPreGLCon[i]; j++)
//		{
//			int glInd;
//
//			glInd = pPreGLConArr[i][j];
//
//			for (int k = 0; k < numpGLPostGLCon[glInd]; k++)
//			{
//				int postInd;
//
//				postInd = pGLPostGLConArr[glInd][k];
//
//				pPreConArr[i][numpPreCon[i]] = postInd;
//				numpPreCon[i]++;
//
//				pPostConArr[postInd][numpPostCon[postInd]] = i;
//				numpPostCon[postInd]++;
//			}
//		}
//	}
//}

