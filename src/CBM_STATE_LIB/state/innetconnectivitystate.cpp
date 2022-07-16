/*
 * innetconnectivitystate.cpp
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */
#include "state/innetconnectivitystate.h"

//InNetConnectivityState::InNetConnectivityState(unsigned int msPerStep, )

InNetConnectivityState::InNetConnectivityState(unsigned int msPerStep, int randSeed)
{
	CRandomSFMT0 randGen(randSeed);

	std::cout << "[INFO]: allocating and initializing connectivity arrays..." << std::endl;
	allocateMemory();
	initializeVals();

	std::cout << "[INFO]: Initializing innet connections..." << std::endl;

	std::cout << "[INFO]: Connecting MF and GL" << std::endl;
	connectMFGL_noUBC();

	std::cout << "[INFO]: Connecting GR and GL" << std::endl;
	connectGLGR(randGen);

	std::cout << "[INFO]: Connecting GR to GO" << std::endl;
	connectGRGO();

	std::cout << "[INFO]: Connecting GO and GL" << std::endl;
	connectGOGL(randGen);

	std::cout << "[INFO]: Connecting GO to GO" << std::endl;
	connectGOGODecayP(randGen);

	std::cout << "[INFO]: Connecting GO to GO gap junctions" << std::endl;
	connectGOGO_GJ(randGen);

	std::cout << "[INFO]: Translating MF GL" << std::endl;
	translateMFGL();

	std::cout << "[INFO]: Translating GO and GL" << std::endl;
	translateGOGL();

	std::cout << "[INFO]: Assigning GR delays" << std::endl;
	assignGRDelays(msPerStep);

	std::cout << "[INFO]: Finished making innet connections." << std::endl;
}

InNetConnectivityState::InNetConnectivityState(std::fstream &infile)
{
	allocateMemory();
	stateRW(true, infile);
}

//InNetConnectivityState::InNetConnectivityState(const InNetConnectivityState &state)
//{
//	allocateMemory();
//
//	arrayCopy<int>(haspGLfromMFtoGL, state.haspGLfromMFtoGL, numGL);
//	arrayCopy<int>(pGLfromMFtoGL, state.pGLfromMFtoGL, numGL);
//
//	arrayCopy<int>(numpGLfromGLtoGO, state.numpGLfromGLtoGO, numGL);
//	arrayCopy<int>(pGLfromGLtoGO[0], state.pGLfromGLtoGO[0],
//			numGL*maxnumpGLfromGLtoGO);
//
//	arrayCopy<int>(numpGLfromGOtoGL, state.numpGLfromGOtoGL, numGL);
//
//	arrayCopy<int>(numpGLfromGLtoGR, state.numpGLfromGLtoGR, numGL);
//	arrayCopy<int>(pGLfromGLtoGR[0], state.pGLfromGLtoGR[0],
//			numGL*maxnumpGLfromGOtoGL);
//
//	arrayCopy<int>(numpMFfromMFtoGL, state.numpMFfromMFtoGL, numMF);
//	arrayCopy<int>(pMFfromMFtoGL[0], state.pMFfromMFtoGL[0],
//			numMF*20);
//
//	arrayCopy<int>(numpMFfromMFtoGR, state.numpMFfromMFtoGR, numMF);
//	arrayCopy<int>(pMFfromMFtoGR[0], state.pMFfromMFtoGR[0],
//			numMF*maxnumpMFfromMFtoGR);
//
//	arrayCopy<int>(numpMFfromMFtoGO, state.numpMFfromMFtoGO, numMF);
//	arrayCopy<int>(pMFfromMFtoGO[0], state.pMFfromMFtoGO[0],
//			numMF*maxnumpMFfromMFtoGO);
//
//	arrayCopy<int>(numpGOfromGLtoGO, state.numpGOfromGLtoGO, numGO);
//	arrayCopy<int>(pGOfromGLtoGO[0], state.pGOfromGLtoGO[0],
//			numGO*maxnumpGOfromGLtoGO);
//
//	arrayCopy<int>(numpGOfromGOtoGL, state.numpGOfromGOtoGL, numGO);
//	arrayCopy<int>(pGOfromGOtoGL[0], state.pGOfromGOtoGL[0],
//			numGO*maxnumpGOfromGOtoGL);
//
//	arrayCopy<int>(numpGOfromMFtoGO, state.numpGOfromMFtoGO, numGO);
//	arrayCopy<int>(pGOfromMFtoGO[0], state.pGOfromMFtoGO[0],
//			numGO*16);
//
//	arrayCopy<int>(numpGOfromGOtoGR, state.numpGOfromGOtoGR, numGO);
//	arrayCopy<int>(pGOfromGOtoGR[0], state.pGOfromGOtoGR[0],
//			numGO*maxnumpGOfromGOtoGR);
//
//	arrayCopy<int>(numpGOfromGRtoGO, state.numpGOfromGRtoGO, numGO);
//	arrayCopy<int>(pGOfromGRtoGO[0], state.pGOfromGRtoGO[0],
//			numGO*maxnumpGOfromGRtoGO);
//
//	arrayCopy<int>(numpGOGABAInGOGO, state.numpGOGABAInGOGO, numGO);
//	arrayCopy<int>(pGOGABAInGOGO[0], state.pGOGABAInGOGO[0],
//			numGO*maxnumpGOGABAInGOGO);
//
//	arrayCopy<int>(numpGOGABAOutGOGO, state.numpGOGABAOutGOGO, numGO);
//	arrayCopy<int>(pGOGABAOutGOGO[0], state.pGOGABAOutGOGO[0],
//			numGO*maxGOGOsyn);
//
//	arrayCopy<int>(numpGOCoupInGOGO, state.numpGOCoupInGOGO, numGO);
//	arrayCopy<int>(pGOCoupInGOGO[0], state.pGOCoupInGOGO[0],
//			numGO*49);
//
//	arrayCopy<int>(numpGOCoupOutGOGO, state.numpGOCoupOutGOGO, numGO);
//	arrayCopy<int>(pGOCoupOutGOGO[0], state.pGOCoupOutGOGO[0],
//			numGO*49);
//
//	arrayCopy<ct_uint32_t>(pGRDelayMaskfromGRtoBSP, state.pGRDelayMaskfromGRtoBSP, numGR);
//
//	arrayCopy<int>(numpGRfromGLtoGR, state.numpGRfromGLtoGR, numGR);
//	arrayCopy<int>(pGRfromGLtoGR[0], state.pGRfromGLtoGR[0],
//			numGR*maxnumpGRfromGLtoGR);
//
//	arrayCopy<int>(numpGRfromGRtoGO, state.numpGRfromGRtoGO, numGR);
//	arrayCopy<int>(pGRfromGRtoGO[0], state.pGRfromGRtoGO[0],
//			numGR*maxnumpGRfromGRtoGO);
//	arrayCopy<int>(pGRDelayMaskfromGRtoGO[0], state.pGRDelayMaskfromGRtoGO[0],
//			numGR*maxnumpGRfromGRtoGO);
//
//	arrayCopy<int>(numpGRfromGOtoGR, state.numpGRfromGOtoGR, numGR);
//	arrayCopy<int>(pGRfromGOtoGR[0], state.pGRfromGOtoGR[0],
//			numGR*maxnumpGRfromGOtoGR);
//
//	arrayCopy<int>(numpGRfromMFtoGR, state.numpGRfromMFtoGR, numGR);
//	arrayCopy<int>(pGRfromMFtoGR[0], state.pGRfromMFtoGR[0],
//			numGR*maxnumpGRfromMFtoGR);
//}

InNetConnectivityState::~InNetConnectivityState() {deallocMemory();}

void InNetConnectivityState::readState(std::fstream &infile)
{
	stateRW(true, infile);
}

void InNetConnectivityState::writeState(std::fstream &outfile)
{
	stateRW(false, outfile);
}

void InNetConnectivityState::allocateMemory()
{
	haspGLfromMFtoGL = new bool[NUM_GL];
	numpGLfromGLtoGO = new int[NUM_GL];
	pGLfromGLtoGO    = allocate2DArray<int>(NUM_GL, MAX_NUM_P_GL_FROM_GL_TO_GO);
	numpGLfromGOtoGL = new int[NUM_GL];
	pGLfromGOtoGL    = allocate2DArray<int>(NUM_GL, MAX_NUM_P_GL_FROM_GO_TO_GL);
	numpGLfromGLtoGR = new int[NUM_GL];
	pGLfromGLtoGR    = allocate2DArray<int>(NUM_GL, MAX_NUM_P_GL_FROM_GL_TO_GR);
	pGLfromMFtoGL    = new int[NUM_GL];
	numpMFfromMFtoGL = new int[NUM_MF];
	pMFfromMFtoGL    = allocate2DArray<int>(NUM_MF, MAX_NUM_P_MF_FROM_MF_TO_GL);
	numpMFfromMFtoGR = new int[NUM_MF];
	pMFfromMFtoGR    = allocate2DArray<int>(NUM_MF, MAX_NUM_P_MF_FROM_MF_TO_GR);
	numpMFfromMFtoGO = new int[NUM_MF];
	pMFfromMFtoGO    = allocate2DArray<int>(NUM_MF, MAX_NUM_P_MF_FROM_MF_TO_GO);

	//golgi
	numpGOfromGLtoGO = new int[NUM_GO];
	pGOfromGLtoGO    = allocate2DArray<int>(NUM_GO, MAX_NUM_P_GO_FROM_GL_TO_GO);
	numpGOfromGOtoGL = new int[NUM_GO];
	pGOfromGOtoGL    = allocate2DArray<int>(NUM_GO, MAX_NUM_P_GO_FROM_GO_TO_GL);
	numpGOfromMFtoGO = new int[NUM_GO];
	pGOfromMFtoGO    = allocate2DArray<int>(NUM_GO, MAX_NUM_P_GO_FROM_MF_TO_GO);
	numpGOfromGOtoGR = new int[NUM_GO];
	pGOfromGOtoGR    = allocate2DArray<int>(NUM_GO, MAX_NUM_P_GO_FROM_GO_TO_GR);
	numpGOfromGRtoGO = new int[NUM_GO];
	pGOfromGRtoGO    = allocate2DArray<int>(NUM_GO, MAX_NUM_P_GO_FROM_GR_TO_GO);

	// coincidentally, numcongotogo == maxnumpgogabaingogo
	numpGOGABAInGOGO  = new int[NUM_GO];
	pGOGABAInGOGO     = allocate2DArray<int>(NUM_GO, NUM_CON_GO_TO_GO);
	numpGOGABAOutGOGO = new int[NUM_GO];
	pGOGABAOutGOGO    = allocate2DArray<int>(NUM_GO, NUM_CON_GO_TO_GO);

	// go <-> go gap junctions
	numpGOCoupInGOGO     = new int[NUM_GO];
	pGOCoupInGOGO        = allocate2DArray<int>(NUM_GO, NUM_P_GO_TO_GO_GJ);
	numpGOCoupOutGOGO    = new int[NUM_GO];
	pGOCoupOutGOGO       = allocate2DArray<int>(NUM_GO, NUM_P_GO_TO_GO_GJ);
	pGOCoupOutGOGOCCoeff = allocate2DArray<float>(NUM_GO, NUM_P_GO_TO_GO_GJ);
	pGOCoupInGOGOCCoeff  = allocate2DArray<float>(NUM_GO, NUM_P_GO_TO_GO_GJ);

	//granule
	pGRDelayMaskfromGRtoBSP = new ct_uint32_t[NUM_GR];
	numpGRfromGLtoGR        = new int[NUM_GR];
	pGRfromGLtoGR           = allocate2DArray<int>(NUM_GR, MAX_NUM_P_GR_FROM_GL_TO_GR);
	numpGRfromGRtoGO        = new int[NUM_GR];
	pGRfromGRtoGO           = allocate2DArray<int>(NUM_GR, MAX_NUM_P_GR_FROM_GR_TO_GO);
	pGRDelayMaskfromGRtoGO  = allocate2DArray<int>(NUM_GR, MAX_NUM_P_GR_FROM_GR_TO_GO);
	numpGRfromGOtoGR        = new int[NUM_GR];
	pGRfromGOtoGR           = allocate2DArray<int>(NUM_GR, MAX_NUM_P_GR_FROM_GO_TO_GR);
	numpGRfromMFtoGR        = new int[NUM_GR];
	pGRfromMFtoGR           = allocate2DArray<int>(NUM_GR, MAX_NUM_P_GR_FROM_MF_TO_GR);
}

void InNetConnectivityState::initializeVals()
{
	std::fill(haspGLfromMFtoGL, haspGLfromMFtoGL + NUM_GL, false);
	std::fill(numpGLfromGLtoGO, numpGLfromGLtoGO + NUM_GL, 0);
	std::fill(pGLfromGLtoGO[0], pGLfromGLtoGO[0]
		+ NUM_GL * MAX_NUM_P_GL_FROM_GL_TO_GO, 0);
	std::fill(numpGLfromGOtoGL, numpGLfromGOtoGL + NUM_GL, 0);
	std::fill(pGLfromGOtoGL[0], pGLfromGOtoGL[0]
		+ NUM_GL * MAX_NUM_P_GL_FROM_GO_TO_GL, 0);
	std::fill(numpGLfromGLtoGR, numpGLfromGLtoGR + NUM_GL, 0);
	std::fill(pGLfromGLtoGR[0], pGLfromGLtoGR[0]
		+ NUM_GL * MAX_NUM_P_GL_FROM_GL_TO_GR, 0);
	std::fill(pGLfromMFtoGL, pGLfromMFtoGL + NUM_GL, 0);
	std::fill(numpMFfromMFtoGL, numpMFfromMFtoGL + NUM_MF, 0);
	std::fill(pMFfromMFtoGL[0], pMFfromMFtoGL[0]
		+ NUM_MF * MAX_NUM_P_MF_FROM_MF_TO_GL, 0);
	std::fill(numpMFfromMFtoGR, numpMFfromMFtoGR + NUM_MF, 0);
	std::fill(pMFfromMFtoGR[0], pMFfromMFtoGR[0]
		+ NUM_MF * MAX_NUM_P_MF_FROM_MF_TO_GR, 0);
	std::fill(numpMFfromMFtoGO, numpMFfromMFtoGO + NUM_MF, 0);
	std::fill(pMFfromMFtoGO[0], pMFfromMFtoGO[0]
		+ NUM_MF * MAX_NUM_P_MF_FROM_MF_TO_GO, 0);

	std::fill(numpGOfromGLtoGO, numpGOfromGLtoGO + NUM_GO, 0);
	std::fill(pGOfromGLtoGO[0], pGOfromGLtoGO[0]
		+ NUM_GO * MAX_NUM_P_GO_FROM_GL_TO_GO, 0);
	std::fill(numpGOfromGOtoGL, numpGOfromGOtoGL + NUM_GO, 0);
	std::fill(pGOfromGOtoGL[0], pGOfromGOtoGL[0]
		+ NUM_GO * MAX_NUM_P_GO_FROM_GO_TO_GL, 0);

	std::fill(numpGOfromMFtoGO, numpGOfromMFtoGO + NUM_GO, 0);
	std::fill(pGOfromMFtoGO[0], pGOfromMFtoGO[0]
		+ NUM_GO * MAX_NUM_P_GO_FROM_MF_TO_GO, 0);
	std::fill(numpGOfromGOtoGR, numpGOfromGOtoGR + NUM_GO, 0);
	std::fill(pGOfromGOtoGR[0], pGOfromGOtoGR[0]
		+ NUM_GO * MAX_NUM_P_GO_FROM_GO_TO_GR, 0);
	std::fill(numpGOfromGRtoGO, numpGOfromGRtoGO + NUM_GO, 0);
	std::fill(pGOfromGRtoGO[0], pGOfromGRtoGO[0]
		+ NUM_GO * MAX_NUM_P_GO_FROM_GR_TO_GO, 0);

	std::fill(numpGOGABAInGOGO, numpGOGABAInGOGO + NUM_GO, 0);
	std::fill(pGOGABAInGOGO[0], pGOGABAInGOGO[0]
		+ NUM_GO * NUM_CON_GO_TO_GO, INT_MAX);
	std::fill(numpGOGABAOutGOGO, numpGOGABAOutGOGO + NUM_GO, 0);
	std::fill(pGOGABAOutGOGO[0], pGOGABAOutGOGO[0]
		+ NUM_GO * NUM_CON_GO_TO_GO, INT_MAX);

	std::fill(numpGOCoupInGOGO, numpGOCoupInGOGO + NUM_GO, 0);
	std::fill(pGOCoupInGOGO[0], pGOCoupInGOGO[0]
		+ NUM_GO * NUM_P_GO_TO_GO_GJ, 0);
	std::fill(numpGOCoupOutGOGO, numpGOCoupOutGOGO + NUM_GO, 0);
	std::fill(pGOCoupOutGOGO[0], pGOCoupOutGOGO[0]
		+ NUM_GO * NUM_P_GO_TO_GO_GJ, 0);
	std::fill(pGOCoupOutGOGOCCoeff[0], pGOCoupOutGOGOCCoeff[0]
		+ NUM_GO * NUM_P_GO_TO_GO_GJ, 0);
	std::fill(pGOCoupInGOGOCCoeff[0], pGOCoupInGOGOCCoeff[0]
		+ NUM_GO * NUM_P_GO_TO_GO_GJ, 0);

	std::fill(pGRDelayMaskfromGRtoBSP, pGRDelayMaskfromGRtoBSP + NUM_GR, 0);
	std::fill(numpGRfromGLtoGR, numpGRfromGLtoGR + NUM_GR, 0);
	std::fill(pGRfromGLtoGR[0], pGRfromGLtoGR[0]
		+ NUM_GR * MAX_NUM_P_GR_FROM_GL_TO_GR, 0);
	std::fill(numpGRfromGRtoGO, numpGRfromGRtoGO + NUM_GR, 0);
	std::fill(pGRfromGRtoGO[0], pGRfromGRtoGO[0]
		+ NUM_GR * MAX_NUM_P_GR_FROM_GR_TO_GO, 0);
	std::fill(pGRDelayMaskfromGRtoGO[0], pGRDelayMaskfromGRtoGO[0]
		+ NUM_GR * MAX_NUM_P_GR_FROM_GR_TO_GO, 0);
	std::fill(numpGRfromGOtoGR, numpGRfromGOtoGR + NUM_GR, 0);
	std::fill(pGRfromGOtoGR[0], pGRfromGOtoGR[0]
		+ NUM_GR * MAX_NUM_P_GR_FROM_GO_TO_GR, 0);
	std::fill(numpGRfromMFtoGR, numpGRfromMFtoGR + NUM_GR, 0);
	std::fill(pGRfromMFtoGR[0], pGRfromMFtoGR[0]
		+ NUM_GR * MAX_NUM_P_GR_FROM_MF_TO_GR, 0);
}

void InNetConnectivityState::deallocMemory()
{
	// mf
	delete[] haspGLfromMFtoGL;
	delete[] numpGLfromGLtoGO;
	delete2DArray<int>(pGLfromGLtoGO);	
	delete[] numpGLfromGOtoGL;
	delete2DArray<int>(pGLfromGOtoGL);
	delete[] numpGLfromGLtoGR;
	delete2DArray<int>(pGLfromGLtoGR);
	delete[] pGLfromMFtoGL;
	delete[] numpMFfromMFtoGL;
	delete2DArray<int>(pMFfromMFtoGL);
	delete[] numpMFfromMFtoGR;
	delete2DArray<int>(pMFfromMFtoGR);
	delete[] numpMFfromMFtoGO;
	delete2DArray<int>(pMFfromMFtoGO);

	// golgi
	delete[] numpGOfromGLtoGO;
	delete2DArray<int>(pGOfromGLtoGO);
	delete[] numpGOfromGOtoGL;
	delete2DArray<int>(pGOfromGOtoGL);
	delete[] numpGOfromMFtoGO;
	delete2DArray<int>(pGOfromMFtoGO);
	delete[] numpGOfromGOtoGR;
	delete2DArray<int>(pGOfromGOtoGR);
	delete[] numpGOfromGRtoGO;
	delete2DArray<int>(pGOfromGRtoGO);

	// go gaba
	delete[] numpGOGABAInGOGO;
	delete2DArray<int>(pGOGABAInGOGO);
	delete[] numpGOGABAOutGOGO;
	delete2DArray<int>(pGOGABAOutGOGO);

	// go gap junction
	delete[] numpGOCoupInGOGO;
	delete2DArray<int>(pGOCoupInGOGO);
	delete[] numpGOCoupOutGOGO;
	delete2DArray<int>(pGOCoupOutGOGO);
	delete2DArray<float>(pGOCoupOutGOGOCCoeff);
	delete2DArray<float>(pGOCoupInGOGOCCoeff);

	// granule
	delete[] pGRDelayMaskfromGRtoBSP;
	delete[] numpGRfromGLtoGR;
	delete2DArray<int>(pGRfromGLtoGR);
	delete[] numpGRfromGRtoGO;
	delete2DArray<int>(pGRfromGRtoGO);
	delete2DArray<int>(pGRDelayMaskfromGRtoGO);
	delete[] numpGRfromGOtoGR;
	delete2DArray<int>(pGRfromGOtoGR);
	delete[] numpGRfromMFtoGR;
	delete2DArray<int>(pGRfromMFtoGR);
}

void InNetConnectivityState::stateRW(bool read, std::fstream &file)
{
	//glomerulus
	rawBytesRW((char *)haspGLfromMFtoGL, NUM_GL * sizeof(bool), read, file);
	rawBytesRW((char *)numpGLfromGLtoGO, NUM_GL * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGO[0],
		NUM_GL * MAX_NUM_P_GL_FROM_GL_TO_GO * sizeof(int), read, file);
	rawBytesRW((char *)numpGLfromGOtoGL, NUM_GL * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGOtoGL[0],
		NUM_GL * MAX_NUM_P_GL_FROM_GO_TO_GL * sizeof(int), read, file);
	rawBytesRW((char *)numpGLfromGLtoGR, NUM_GL * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGR[0],
		NUM_GL * MAX_NUM_P_GL_FROM_GL_TO_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromMFtoGL, NUM_GL * sizeof(int), read, file);

	//mossy fibers
	rawBytesRW((char *)numpMFfromMFtoGL, NUM_MF * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGL[0],
		NUM_MF * MAX_NUM_P_MF_FROM_MF_TO_GL * sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGR, NUM_MF * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGR[0],
		NUM_MF * MAX_NUM_P_MF_FROM_MF_TO_GR * sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGO, NUM_MF * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGO[0],
		NUM_MF * MAX_NUM_P_MF_FROM_MF_TO_GO * sizeof(int), read, file);

	//golgi
	rawBytesRW((char *)numpGOfromGLtoGO, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGLtoGO[0],
		NUM_GO * MAX_NUM_P_GO_FROM_GL_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGL, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGL[0],
		NUM_GO * MAX_NUM_P_GO_FROM_GO_TO_GL * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromMFtoGO, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromMFtoGO[0],
		NUM_GO * MAX_NUM_P_GO_FROM_MF_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGR, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGR[0],
		NUM_GO * MAX_NUM_P_GO_FROM_GO_TO_GR * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGRtoGO, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGRtoGO[0],
		NUM_GO * MAX_NUM_P_GO_FROM_GR_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAInGOGO, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAInGOGO[0],
		NUM_GO * NUM_CON_GO_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAOutGOGO, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAOutGOGO[0],
		NUM_GO * NUM_CON_GO_TO_GO * sizeof(int), read, file);
	
	rawBytesRW((char *)numpGOCoupInGOGO, NUM_GO*sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupInGOGO[0],
		NUM_GO * NUM_P_GO_TO_GO_GJ * sizeof(int), read, file);

	rawBytesRW((char *)numpGOCoupOutGOGO, NUM_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupOutGOGO[0],
		NUM_GO * NUM_P_GO_TO_GO_GJ * sizeof(int), read, file);

	rawBytesRW((char *)pGOCoupOutGOGOCCoeff[0],
		NUM_GO * NUM_P_GO_TO_GO_GJ * sizeof(float), read, file);
	rawBytesRW((char *)pGOCoupInGOGOCCoeff[0],
		NUM_GO * NUM_P_GO_TO_GO_GJ * sizeof(float), read, file);

	//granule
	rawBytesRW((char *)pGRDelayMaskfromGRtoBSP, NUM_GR * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromGLtoGR, NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGLtoGR[0],
		NUM_GR * MAX_NUM_P_GR_FROM_GL_TO_GR * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGRtoGO, NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGRtoGO[0],
		NUM_GR * MAX_NUM_P_GR_FROM_GR_TO_GO * sizeof(int), read, file);
	rawBytesRW((char *)pGRDelayMaskfromGRtoGO[0],
		NUM_GR * MAX_NUM_P_GR_FROM_GR_TO_GO * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGOtoGR, NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGOtoGR[0],
		NUM_GR * MAX_NUM_P_GR_FROM_GO_TO_GR * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromMFtoGR, NUM_GR * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromMFtoGR[0],
		NUM_GR * MAX_NUM_P_GR_FROM_MF_TO_GR * sizeof(int), read, file);
}


void InNetConnectivityState::connectMFGL_noUBC()
{
	// shorter convenience var names
	int glX = GL_X;
	int glY = GL_Y;
	int mfX = MF_X;
	int mfY = MF_Y;

	// define span and coord arrays locally
	int spanArrayMFtoGLX[SPAN_MF_TO_GL_X + 1] = {0};
	int spanArrayMFtoGLY[SPAN_MF_TO_GL_Y + 1] = {0};
	int xCoorsMFGL[NUM_P_MF_TO_GL] = {0};
	int yCoorsMFGL[NUM_P_MF_TO_GL] = {0};

	// fill span arrays and coord arrays
	for (int i = 0; i < SPAN_MF_TO_GL_X + 1; i++)
	{
		spanArrayMFtoGLX[i] = i - (SPAN_MF_TO_GL_X / 2);
	}

	for (int i = 0; i < SPAN_MF_TO_GL_Y + 1; i++)
	{
		spanArrayMFtoGLY[i] = i - (SPAN_MF_TO_GL_Y / 2);
	}
		
	for (int i = 0; i < NUM_P_MF_TO_GL; i++)
	{
		xCoorsMFGL[i] = spanArrayMFtoGLX[i % (SPAN_MF_TO_GL_X + 1)];
		yCoorsMFGL[i] = spanArrayMFtoGLY[i / (SPAN_MF_TO_GL_Y + 1)];		
	}

	// scale factors from one cell coord to another
	float gridXScaleSrctoDest = (float)mfX / (float)glX; 
	float gridYScaleSrctoDest = (float)mfY/ (float)glY; 

	// random mf index array, supposedly to even out distribution of connections
	int rMFInd[NUM_MF] = {0};	
	for (int i = 0; i < NUM_MF; i++) rMFInd[i] = i;	
	std::random_shuffle(rMFInd, rMFInd + NUM_MF);

	// fill random span array with linear indices
	int rMFSpanInd[NUM_P_MF_TO_GL] = {0};
	for (int i = 0; i < NUM_P_MF_TO_GL; i++) rMFSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	// attempt to make connections
	for (int attempts = 0; attempts < MAX_MF_TO_GL_ATTEMPTS; attempts++)
	{
		std::random_shuffle(rMFInd, rMFInd + NUM_MF);	
		// for each attempt, loop through all presynaptic cells
		for (int i = 0; i < NUM_MF; i++)
		{	
			//Select MF Coordinates from random index array: Complete	
			srcPosX = rMFInd[i] % mfX;
			srcPosY = rMFInd[i] / mfX;		
			
			std::random_shuffle(rMFSpanInd, rMFSpanInd + NUM_P_MF_TO_GL);	
			// for each presynaptic cell, attempt to make up to initial output + max attempts
			// connections.	
			for (int j = 0; j < NUM_P_MF_TO_GL; j++)
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
				if (numpMFfromMFtoGL[rMFInd[i]] == (INITIAL_MF_OUTPUT + attempts)) break;
				
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
	for (int i = 0; i < NUM_MF; i++) count += numpMFfromMFtoGL[i];
	
	std::cout << "Total number of Mossy Fiber to Glomeruli connections:	" << count << std::endl;
	std::cout << "Correct number: " << NUM_GL << std::endl;
}

void InNetConnectivityState::connectGLGR(CRandomSFMT &randGen)
{
	int grX = GR_X;
	// int grY = GR_Y; /* unused */
	int glX = GL_X;
	int glY = GL_Y;

	float gridXScaleStoD = (float)grX / (float)glX;
	// float gridYScaleStoD = (float)grY / (float)glY; /* unused actually :/ */

	bool srcConnected[NUM_GR] = {false};

	for (int i = 0; i < MAX_NUM_P_GR_FROM_GL_TO_GR; i++)
	{
		int srcNumConnected = 0;
		std::fill(srcConnected, srcConnected + NUM_GR, false);

		while (srcNumConnected < NUM_GR)
		{
			int srcIndex = randGen.IRandom(0, NUM_GR - 1);
			if (!srcConnected[srcIndex])
			{
				int srcPosX = srcIndex % grX;
				int srcPosY = (int)(srcIndex / grX);

				int tempDestNumConLim = LOW_NUM_P_GL_FROM_GL_TO_GR;

				for (int attempts = 0; attempts < MAX_GL_TO_GR_ATTEMPTS; attempts++)
				{
					if (attempts == LOW_GL_TO_GR_ATTEMPTS) tempDestNumConLim = MAX_NUM_P_GL_FROM_GL_TO_GR;

					int destPosX = (int)round(srcPosX / gridXScaleStoD);
					int destPosY = (int)round(srcPosY / gridXScaleStoD);

					// again, should add 1 to spans
					destPosX += round((randGen.Random() - 0.5) * SPAN_GL_TO_GR_X);
					destPosY += round((randGen.Random() - 0.5) * SPAN_GL_TO_GR_Y);

					destPosX = (destPosX % glX + glX) % glX;
					destPosY = (destPosY % glY + glY) % glY;

					int destIndex = destPosY * glX + destPosX;

					// for gl -> gr, we set needUnique to true
					bool unique = true;
					for (int j = 0; j < i; j++)
					{
						if (destIndex == pGRfromGLtoGR[srcIndex][j])
						{
							unique = false;
							break;
						}
					}

					if (unique && numpGLfromGLtoGR[destIndex] < tempDestNumConLim)
					{
						pGLfromGLtoGR[destIndex][numpGLfromGLtoGR[destIndex]] = srcIndex;
						numpGLfromGLtoGR[destIndex]++;
						pGRfromGLtoGR[srcIndex][i] = destIndex;
						numpGRfromGLtoGR[srcIndex]++;
						break;
					}
				}
				srcConnected[srcIndex] = true;
				srcNumConnected++;
			}
		}
	}

	int count = 0;
	for (int i = 0; i < NUM_GL; i++)
	{
		count += numpGLfromGLtoGR[i];
	}

	std::cout << "Total number of Glomeruli to Granule connections:	" << count << std::endl; 
	std::cout << "Correct number: " << NUM_GR * MAX_NUM_P_GR_FROM_GL_TO_GR << std::endl;
	// for now, no empty counter
}

void InNetConnectivityState::connectGRGO()
{
	int grX = GR_X;
	int grY = GR_Y;
	int goX = GO_X;
	int goY = GO_Y;

	int spanArrayPFtoGOX[SPAN_PF_TO_GO_X + 1] = {0};
	int spanArrayPFtoGOY[SPAN_PF_TO_GO_Y + 1] = {0};
	int xCoorsPFGO[NUM_P_PF_TO_GO] = {0};
	int yCoorsPFGO[NUM_P_PF_TO_GO] = {0};

	//PARALLEL FIBER TO GOLGI 
	for (int i = 0; i < SPAN_PF_TO_GO_X + 1; i++)
	{
		spanArrayPFtoGOX[i] = i - (SPAN_PF_TO_GO_X / 2);
	}

	for (int i = 0; i < SPAN_PF_TO_GO_Y + 1; i++)
	{
		spanArrayPFtoGOY[i] = i - (SPAN_PF_TO_GO_Y / 2);
	}
		
	for (int i = 0; i < NUM_P_PF_TO_GO; i++)
	{
		xCoorsPFGO[i] = spanArrayPFtoGOX[i % (SPAN_PF_TO_GO_X + 1)];
		yCoorsPFGO[i] = spanArrayPFtoGOY[i / (SPAN_PF_TO_GO_X + 1)];	
	}

	// Grid Scale: Complete
	float gridXScaleSrctoDest = (float)goX / (float)grX; 
	float gridYScaleSrctoDest = (float)goY / (float)grY; 

	//Make Random Span Array: Complete
	int rPFSpanInd[NUM_P_PF_TO_GO] = {0};
	for (int i = 0; i < NUM_P_PF_TO_GO; i++) rPFSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int attempts = 0; attempts < MAX_PF_TO_GO_ATTEMPTS; attempts++)
	{
		for (int i = 0; i < NUM_GO; i++)
		{	
			srcPosX = i % goX;
			srcPosY = i / goX;

			std::random_shuffle(rPFSpanInd, rPFSpanInd + NUM_P_PF_TO_GO);		
			for (int j = 0; j < MAX_PF_TO_GO_INPUT; j++)
			{	
				destPosX = xCoorsPFGO[rPFSpanInd[j]]; 
				destPosY = yCoorsPFGO[rPFSpanInd[j]];	
				
				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);
				
				destPosX = (destPosX % grX + grX) % grX;
				destPosY = (destPosY % grY + grY) % grY;
						
				destIndex = destPosY * grX + destPosX;

				if (numpGOfromGRtoGO[i] < MAX_PF_TO_GO_INPUT)
				{	
					pGOfromGRtoGO[i][numpGOfromGRtoGO[i]] = destIndex;
					numpGOfromGRtoGO[i]++;

					pGRfromGRtoGO[destIndex][numpGRfromGRtoGO[destIndex]] = i;
					numpGRfromGRtoGO[destIndex]++;	
				}
			}
		}
	}

	int spanArrayAAtoGOX[SPAN_AA_TO_GO_X + 1] = {0};
	int spanArrayAAtoGOY[SPAN_AA_TO_GO_Y + 1] = {0};
	int xCoorsAAGO[NUM_P_AA_TO_GO] = {0};
	int yCoorsAAGO[NUM_P_AA_TO_GO] = {0};

	for (int i = 0; i < SPAN_AA_TO_GO_X + 1;i++)
	{
		spanArrayAAtoGOX[i] = i - (SPAN_AA_TO_GO_X / 2);
	}

	for (int i = 0; i < SPAN_AA_TO_GO_Y + 1;i++)
	{
		spanArrayAAtoGOY[i] = i - (SPAN_AA_TO_GO_Y / 2);
	}
		
	for (int i = 0; i < NUM_P_AA_TO_GO; i++)
	{
		xCoorsAAGO[i] = spanArrayAAtoGOX[i % (SPAN_AA_TO_GO_X + 1)];
		yCoorsAAGO[i] = spanArrayAAtoGOY[i / (SPAN_AA_TO_GO_Y + 1)];	
	}
	
	int rAASpanInd[NUM_P_AA_TO_GO] = {0};
	for (int i = 0; i < NUM_P_AA_TO_GO; i++) rAASpanInd[i] = i;

	for (int attempts = 0; attempts < MAX_AA_TO_GO_ATTEMPTS; attempts++)
	{
		for (int i = 0; i < NUM_GO; i++)
		{	
			srcPosX = i % goX;
			srcPosY = i / goX;

			std::random_shuffle(rAASpanInd, rAASpanInd + NUM_P_AA_TO_GO);		
			
			for (int j = 0; j < MAX_AA_TO_GO_INPUT; j++)
			{	
				destPosX = xCoorsAAGO[rAASpanInd[j]]; 
				destPosY = yCoorsAAGO[rAASpanInd[j]];	
				
				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);

				destPosX = (destPosX % grX + grX) % grX;
				destPosY = (destPosY % grY + grY) % grY;
						
				destIndex = destPosY * grX + destPosX;
				
				if (numpGOfromGRtoGO[i] < MAX_AA_TO_GO_INPUT + MAX_PF_TO_GO_INPUT)
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
	
	for (int i = 0; i < NUM_GO; i++)
	{
		sumGOGR_GO += numpGOfromGRtoGO[i];
	}

	std::cout << "GRtoGO_GO: " << sumGOGR_GO << std::endl;

	int sumGOGR_GR = 0;

	for (int i = 0; i < NUM_GR; i++)
	{
		sumGOGR_GR += numpGRfromGRtoGO[i];
	}

	std::cout << "GRtoGO_GR: " << sumGOGR_GR << std::endl;
}

void InNetConnectivityState::connectGOGL(CRandomSFMT &randGen)
{
	// using old connectivity alg for now , cannot generalize (do not always know
	// at least both array bounds for 2D arrays at compile time)
	// gl -> go
	int goX = GO_X;
	int goY = GO_Y;
	int glX = GL_X;
	int glY = GL_Y;

	float gridXScaleSrctoDest = (float)goX / (float)glX;
	float gridYScaleSrctoDest = (float)goY / (float)glY;

	bool srcConnected[NUM_GO] = {false};

	for (int i = 0; i < MAX_NUM_P_GO_FROM_GL_TO_GO; i++)
	{
		int srcNumConnected = 0;
		while (srcNumConnected < NUM_GO)
		{
			int srcIndex = randGen.IRandom(0, NUM_GO - 1);
			if (!srcConnected[srcIndex])
			{
				int srcPosX = srcIndex % goX;
				int srcPosY = (int)(srcIndex / goX);

				int tempDestNumConLim = LOW_NUM_P_GL_FROM_GL_TO_GO;

				for (int attempts = 0; attempts < MAX_GL_TO_GO_ATTEMPTS; attempts++)
				{
					int destPosX;
					int destPosY;
					int destIndex;

					if (attempts == LOW_GL_TO_GO_ATTEMPTS) tempDestNumConLim = MAX_NUM_P_GL_FROM_GL_TO_GO;

					destPosX = (int)round(srcPosX / gridXScaleSrctoDest);
					destPosY = (int)round(srcPosY / gridXScaleSrctoDest);

					// should multiply spans by 1 for full coverage
					destPosX += round((randGen.Random() - 0.5) * SPAN_GL_TO_GO_X);
					destPosY += round((randGen.Random() - 0.5) * SPAN_GL_TO_GO_Y);

					destPosX = (destPosX % glX + glX) % glX;
					destPosY = (destPosY % glY + glY) % glY;

					destIndex = destPosY * glX + destPosX;

					if (numpGLfromGLtoGO[destIndex] < tempDestNumConLim)
					{
						pGLfromGLtoGO[destIndex][numpGLfromGLtoGO[destIndex]] = srcIndex;
						numpGLfromGLtoGO[destIndex]++;
						pGOfromGLtoGO[srcIndex][numpGOfromGLtoGO[srcIndex]] = destIndex;
						numpGOfromGLtoGO[srcIndex]++;
						break;
					}
				}
				srcConnected[srcIndex] = true;
				srcNumConnected++;
			}
		}
		std::fill(srcConnected, srcConnected + NUM_GO, false);
	}

	std::cout << "[INFO]: Finished making gl go connections." << std::endl;
	std::cout << "[INFO]: Starting on go gl connections..." << std::endl;

	// go --> gl

	int spanArrayGOtoGLX[SPAN_GO_TO_GL_X + 1] = {0};
	int spanArrayGOtoGLY[SPAN_GO_TO_GL_Y + 1] = {0};
	int xCoorsGOGL[NUM_P_GO_TO_GL] = {0};
	int yCoorsGOGL[NUM_P_GO_TO_GL] = {0};
	float pConGOGL[NUM_P_GO_TO_GL] = {0.0};

	// Make span Array
	for (int i = 0; i < SPAN_GO_TO_GL_X + 1; i++)
	{
		spanArrayGOtoGLX[i] = i - (SPAN_GO_TO_GL_X / 2);
	}

	for (int i = 0; i < SPAN_GO_TO_GL_Y + 1; i++)
	{
		spanArrayGOtoGLY[i] = i - (SPAN_GO_TO_GL_Y / 2);
	}
		
	for (int i = 0; i < NUM_P_GO_TO_GL; i++)
	{
		xCoorsGOGL[i] = spanArrayGOtoGLX[i % (SPAN_GO_TO_GL_X + 1)];
		yCoorsGOGL[i] = spanArrayGOtoGLY[i / (SPAN_GO_TO_GL_Y + 1)];	
	}

	// Probability of connection as a function of distance
	for (int i = 0; i < NUM_P_GO_TO_GL; i++)
	{
		float PconX = (xCoorsGOGL[i] * xCoorsGOGL[i])
			/ (2 * STD_DEV_GO_TO_GL_ML * STD_DEV_GO_TO_GL_ML);
		float PconY = (yCoorsGOGL[i] * yCoorsGOGL[i])
			/ (2 * STD_DEV_GO_TO_GL_S * STD_DEV_GO_TO_GL_S);
		pConGOGL[i] = AMPL_GO_TO_GL * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < NUM_P_GO_TO_GL; i++)
	{
		if ((xCoorsGOGL[i] == 0) && (yCoorsGOGL[i] == 0)) pConGOGL[i] = 0;
	}

	//Make Random Golgi cell Index Array	
	int rGOInd[NUM_GO] = {0};
	for (int i = 0; i < NUM_GO; i++) rGOInd[i] = i;

	//Make Random Span Array
	int rGOSpanInd[NUM_P_GO_TO_GL] = {0};
	for (int i = 0; i < NUM_P_GO_TO_GL; i++) rGOSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int attempts = 0; attempts < MAX_GO_TO_GL_ATTEMPTS; attempts++)
	{
		std::random_shuffle(rGOInd, rGOInd + NUM_GO);	
		
		// Go through each golgi cell 
		for (int i = 0; i < NUM_GO; i++)
		{
			//Select GO Coordinates from random index array: Complete	
			srcPosX = rGOInd[i] % goX;
			srcPosY = rGOInd[i] / goX;	
			
			std::random_shuffle(rGOSpanInd, rGOSpanInd + NUM_P_GO_TO_GL);	
			
			for (int j = 0; j < NUM_P_GO_TO_GL; j++)   
			{	
				// relative position of connection
				destPosX = xCoorsGOGL[rGOSpanInd[j]];
				destPosY = yCoorsGOGL[rGOSpanInd[j]];	

				destPosX += (int)round(srcPosX / gridXScaleSrctoDest); 
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);

				destPosX = (destPosX % glX + glX) % glX;  
				destPosY = (destPosY % glY + glY) % glY;
				
				// Change position to Index	
				destIndex = destPosY * glX + destPosX;
					
				if (numpGOfromGOtoGL[rGOInd[i]] >= INITIAL_GO_INPUT + attempts) break; 
				if (randGen.Random() >= 1 - pConGOGL[rGOSpanInd[j]] && 
						numpGLfromGOtoGL[destIndex] < MAX_NUM_P_GL_FROM_GO_TO_GL) 
				{	
					pGOfromGOtoGL[rGOInd[i]][numpGOfromGOtoGL[rGOInd[i]]] = destIndex;
					numpGOfromGOtoGL[rGOInd[i]]++;

					pGLfromGOtoGL[destIndex][numpGLfromGOtoGL[destIndex]] = rGOInd[i];
					numpGLfromGOtoGL[destIndex]++;
				}
			}
		}
	}

	std::cout << "[INFO]: Finished making go gl connections." << std::endl;

	int shitCounter = 0;
	int totalGOGL = 0;

	for (int i = 0; i < NUM_GL; i++)
	{
		if (numpGLfromGOtoGL[i] < MAX_NUM_P_GL_FROM_GO_TO_GL) shitCounter++;
		totalGOGL += numpGLfromGOtoGL[i];
	}

	std::cout << "Empty Glomeruli Counter: " << shitCounter << std::endl;
	std::cout << "Total GO -> GL: " << totalGOGL << std::endl;
	std::cout << "avg Num  GO -> GL Per GL: " << (float)totalGOGL / (float)NUM_GL << std::endl;
}

void InNetConnectivityState::connectGOGODecayP(CRandomSFMT &randGen)
{
	int goX = GO_X;
	int goY = GO_Y;

	int spanArrayGOtoGOsynX[SPAN_GO_TO_GO_X + 1] = {0};
	int spanArrayGOtoGOsynY[SPAN_GO_TO_GO_Y + 1] = {0};
	int xCoorsGOGOsyn[NUM_P_GO_TO_GO] = {0};
	int yCoorsGOGOsyn[NUM_P_GO_TO_GO] = {0};
	float Pcon[NUM_P_GO_TO_GO] = {0}; 				

	bool **conGOGOBoolOut = allocate2DArray<bool>(NUM_GO, NUM_GO);
	std::fill(conGOGOBoolOut[0], conGOGOBoolOut[0] + NUM_GO * NUM_GO, false);

	for (int i = 0; i < SPAN_GO_TO_GO_X + 1; i++)
   	{
		spanArrayGOtoGOsynX[i] = i - (SPAN_GO_TO_GO_X / 2);
	}

	for (int i = 0; i < SPAN_GO_TO_GO_Y + 1; i++)
   	{
		spanArrayGOtoGOsynY[i] = i - (SPAN_GO_TO_GO_Y / 2);
	}
		
	for (int i = 0; i < NUM_P_GO_TO_GO; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (SPAN_GO_TO_GO_X + 1)];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (SPAN_GO_TO_GO_Y + 1)];		
	}

	for (int i = 0; i < NUM_P_GO_TO_GO; i++)
	{
		float PconX = (xCoorsGOGOsyn[i] * xCoorsGOGOsyn[i])
			/ (2 * (STD_DEV_GO_TO_GO * STD_DEV_GO_TO_GO));
		float PconY = (yCoorsGOGOsyn[i] * yCoorsGOGOsyn[i])
			/ (2 * (STD_DEV_GO_TO_GO * STD_DEV_GO_TO_GO));
		Pcon[i] = AMPL_GO_TO_GO * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < NUM_P_GO_TO_GO; i++) 
	{
		if ((xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0)) Pcon[i] = 0;
	}
	
	int rGOGOSpanInd[NUM_P_GO_TO_GO] = {0};
	for (int i = 0; i < NUM_P_GO_TO_GO; i++) rGOGOSpanInd[i] = i;

	for (int attempts = 0; attempts < MAX_GO_TO_GO_ATTEMPTS; attempts++) 
	{
		for (int i = 0; i < NUM_GO; i++) 
		{	
			int srcPosX = i % goX;
			int srcPosY = i / goX;	
			
			std::random_shuffle(rGOGOSpanInd, rGOGOSpanInd + NUM_P_GO_TO_GO);		
			
			for (int j = 0; j < NUM_P_GO_TO_GO; j++)
		   	{	
				int destPosX = srcPosX + xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
				int destPosY = srcPosY + yCoorsGOGOsyn[rGOGOSpanInd[j]];	

				destPosX = (destPosX % goX + goX) % goX;
				destPosY = (destPosY % goY + goY) % goY;
						
				int destIndex = destPosY * goX + destPosX;
			
				// Normal One	
				if (GO_GO_RECIP_CONS && !REDUCE_BASE_RECIP_GO_GO
						&& randGen.Random()>= 1 - Pcon[rGOGOSpanInd[j]]
						&& !conGOGOBoolOut[i][destIndex]
						&& numpGOGABAOutGOGO[i] < NUM_CON_GO_TO_GO 
						&& numpGOGABAInGOGO[destIndex] < NUM_CON_GO_TO_GO) 
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;
							
					if (randGen.Random() <= P_RECIP_GO_GO 
							&& !conGOGOBoolOut[destIndex][i]
							&& numpGOGABAOutGOGO[destIndex] < NUM_CON_GO_TO_GO 
							&& numpGOGABAInGOGO[i] < NUM_CON_GO_TO_GO) 
					{
						pGOGABAOutGOGO[destIndex][numpGOGABAOutGOGO[destIndex]] = i;
						numpGOGABAOutGOGO[destIndex]++;
						
						pGOGABAInGOGO[i][numpGOGABAInGOGO[i]] = destIndex;
						numpGOGABAInGOGO[i]++;
						
						conGOGOBoolOut[destIndex][i] = true;
					}
				}
			
				if (GO_GO_RECIP_CONS && REDUCE_BASE_RECIP_GO_GO
						&& randGen.Random() >= 1 - Pcon[rGOGOSpanInd[j]]
						&& !conGOGOBoolOut[i][destIndex] && (!conGOGOBoolOut[destIndex][i] ||
							randGen.Random() <= P_RECIP_LOWER_BASE_GO_GO)
						&& numpGOGABAOutGOGO[i] < NUM_CON_GO_TO_GO 
						&& numpGOGABAInGOGO[destIndex] < NUM_CON_GO_TO_GO) 
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;	
				}

				if (!GO_GO_RECIP_CONS && !REDUCE_BASE_RECIP_GO_GO &&
						randGen.Random() >= 1 - Pcon[rGOGOSpanInd[j]]
						&& (!conGOGOBoolOut[i][destIndex]) && !conGOGOBoolOut[destIndex][i]
						&& numpGOGABAOutGOGO[i] < NUM_CON_GO_TO_GO 
						&& numpGOGABAInGOGO[destIndex] < NUM_CON_GO_TO_GO)
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

	for (int i = 0; i < NUM_GO; i++)
	{
		totalGOGOcons += numpGOGABAInGOGO[i];
	}

	std::cout << "Total GOGO connections: " << totalGOGOcons << std::endl;
	std::cout << "Average GOGO connections:	" << (float)totalGOGOcons / float(NUM_GO) << std::endl;
	std::cout << NUM_GO << std::endl;

	int recipCounter = 0;

	for (int i = 0; i < NUM_GO; i++)
	{
		for (int j = 0; j < numpGOGABAInGOGO[i]; j++)
		{
			for (int k = 0; k < numpGOGABAOutGOGO[i]; k++)
			{
				if (pGOGABAInGOGO[i][j] == pGOGABAOutGOGO[i][k] && pGOGABAInGOGO[i][j] != INT_MAX 
						&& pGOGABAOutGOGO[i][k] != INT_MAX)
				{
					recipCounter++;	
				}
			}
		}
	}

	float fracRecip = (float)recipCounter / (float)totalGOGOcons;
	std::cout << "FracRecip: " << fracRecip << std::endl;
	delete2DArray<bool>(conGOGOBoolOut);
}

void InNetConnectivityState::connectGOGO_GJ(CRandomSFMT &randGen)
{
	int goX = GO_X;
	int goY = GO_Y;

	int spanArrayGOtoGOgjX[SPAN_GO_TO_GO_GJ_X + 1] = {0};
	int spanArrayGOtoGOgjY[SPAN_GO_TO_GO_GJ_Y + 1] = {0};
	int xCoorsGOGOgj[NUM_P_GO_TO_GO_GJ] = {0};
	int yCoorsGOGOgj[ NUM_P_GO_TO_GO_GJ] = {0};

	float gjPCon[NUM_P_GO_TO_GO_GJ] = {0.0};
	float gjCC[NUM_P_GO_TO_GO_GJ] = {0.0};

	bool **gjConBool = allocate2DArray<bool>(NUM_GO, NUM_GO);
	std::fill(gjConBool[0], gjConBool[0] + NUM_GO * NUM_GO, false);

	for (int i = 0; i < SPAN_GO_TO_GO_GJ_X + 1; i++)
	{
		spanArrayGOtoGOgjX[i] = i - (SPAN_GO_TO_GO_GJ_X / 2);
	}

	for (int i = 0; i < SPAN_GO_TO_GO_GJ_Y + 1; i++)
	{
		spanArrayGOtoGOgjY[i] = i - (SPAN_GO_TO_GO_GJ_Y / 2);
	}

	for (int i = 0; i < NUM_P_GO_TO_GO_GJ; i++)
	{
		xCoorsGOGOgj[i] = spanArrayGOtoGOgjX[i % (SPAN_GO_TO_GO_GJ_X + 1)];
		yCoorsGOGOgj[i] = spanArrayGOtoGOgjY[i / (SPAN_GO_TO_GO_GJ_X + 1)];		
	}

	// "In Vivo additions"
	for (int i = 0; i < NUM_P_GO_TO_GO_GJ; i++)
	{
		float gjPConX = exp(((abs(xCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );	
		float gjPConY = exp(((abs(yCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );
		gjPCon[i] = ((-1745.0 + (1836.0 / (1 + (gjPConX + gjPConY)))) * 0.01);
		
		float gjCCX = exp(abs(xCoorsGOGOgj[i] * 100.0) / 190.0);
		float gjCCY = exp(abs(yCoorsGOGOgj[i] * 100.0) / 190.0);
		gjCC[i] = (-2.3 + (23.0 / ((gjCCX + gjCCY)/2.0))) * 0.09;
	}

	// Remove self connection 
	for (int i = 0; i < NUM_P_GO_TO_GO_GJ; i++)
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

	for (int i = 0; i < NUM_GO; i++)
	{	
		srcPosX = i % goX;
		srcPosY = i / goX;	
		
		for (int j = 0; j < NUM_P_GO_TO_GO_GJ; j++)
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
	delete2DArray<bool>(gjConBool);
}

void InNetConnectivityState::translateMFGL()
{

	// Mossy fiber to Granule
	
	for (int i = 0; i < NUM_GR; i++)
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
	
	for (int i = 0; i < NUM_GR; i++)
	{
		grMFInputCounter += numpGRfromMFtoGR[i];
	}

	std::cout << "Total MF inputs: " << grMFInputCounter << std::endl;

	// Mossy fiber to Golgi	
	
	for (int i = 0; i < NUM_GO; i++)
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

void InNetConnectivityState::translateGOGL()
{
	for (int i = 0; i < NUM_GR; i++)
	{
		for (int j = 0; j < MAX_NUM_P_GR_FROM_GO_TO_GR; j++)
		{
			for (int k = 0; k < MAX_NUM_P_GL_FROM_GO_TO_GL; k++)
			{	
				if (numpGRfromGOtoGR[i] < MAX_NUM_P_GR_FROM_GO_TO_GR)
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
	for (int i = 0; i < NUM_GR; i++) totalGOGR += numpGRfromGOtoGR[i];
	
	std::cout << "total GO->GR: " << totalGOGR << std::endl;
	std::cout << "GO->GR Per GR: " << (float)totalGOGR / (float)NUM_GR << std::endl;
}

void InNetConnectivityState::assignGRDelays(unsigned int msPerStep)
{
	for (int i = 0; i < NUM_GR; i++)
	{
		//calculate x coordinate of GR position
		int grPosX = i % GR_X;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		int grBCPCSCDist = abs((int)(GR_X / 2 - grPosX));
		pGRDelayMaskfromGRtoBSP[i] = 0x1 << (int)((grBCPCSCDist / GR_PF_VEL_IN_GR_X_PER_T_STEP
			+ GR_AF_DELAY_IN_T_STEP) / msPerStep);

		for (int j = 0; j < numpGRfromGRtoGO[i]; j++)
		{
			int goPosX = (pGRfromGRtoGO[i][j] % GO_X) * (((float)GR_X) / GO_X);
			int dfromGRtoGO = abs(goPosX - grPosX);

			if (dfromGRtoGO > GR_X / 2)
			{
				if (goPosX < grPosX) dfromGRtoGO = goPosX + GR_X - grPosX;
				else dfromGRtoGO = grPosX + GR_X - goPosX;
			}
			pGRDelayMaskfromGRtoGO[i][j] = 0x1<< (int)((dfromGRtoGO / GR_PF_VEL_IN_GR_X_PER_T_STEP
				+ GR_AF_DELAY_IN_T_STEP) / msPerStep);
		}
	}
}

