/*
 * innetconnectivitystate.cpp
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#include "state/innetconnectivitystate.h"

InNetConnectivityState::InNetConnectivityState(ConnectivityParams *parameters,
		unsigned int msPerStep, int randSeed, int goRecipParam, int simNum)
{
	cp = parameters;

	CRandomSFMT *randGen = new CRandomSFMT0(randSeed);
	
	std::cout << "Input net state construction..." << std::endl;
	std::cout << "allocating memory..." << std::endl;
	allocateMemory();
	
	std::cout << "Initializing connections..." << std::endl;
	initializeVals();
	
	srand(time(0));	

	std::cout << "connecting GO to GO" << std::endl;
	// NOTE: there are different versions of this function	
	connectGOGODecayP(randGen, goRecipParam, simNum);	
	
	std::cout << "connecting GR and GL" << std::endl;
	connectGRGL(randGen);
	
	std::cout << "connecting GO and GL" << std::endl;
	connectGOGL(randGen);
	
	std::cout << "connecting MF and GL" << std::endl;
	connectMFGL_noUBC(randGen);
	
	std::cout << "translating MF GL" << std::endl;
	translateMFGL();
	
	std::cout << "translating GO and GL" << std::endl;
	translateGOGL(randGen);
	
	std::cout << "connecting GR to GO" << std::endl;
	connectGRGO(randGen, goRecipParam);
	
	std::cout << "connecting GO to GO gap junctions" << std::endl;
	connectGOGO_GJ(randGen);
	
	std::cout << "assigning GR delays" << std::endl;
	assignGRDelays(msPerStep);
	
	std::cout << "assigning PF to BC delays" << std::endl;
	std::cout << "done" << std::endl;

	delete randGen;
}

InNetConnectivityState::InNetConnectivityState(ConnectivityParams *parameters, std::fstream &infile)
{
	cp = parameters;

	allocateMemory();

	stateRW(true, infile);
}

InNetConnectivityState::InNetConnectivityState(const InNetConnectivityState &state)
{
	cp = state.cp;

	allocateMemory();

	arrayCopy<int>(haspGLfromMFtoGL, state.haspGLfromMFtoGL, cp->numGL);
	arrayCopy<int>(pGLfromMFtoGL, state.pGLfromMFtoGL, cp->numGL);

	arrayCopy<int>(numpGLfromGLtoGO, state.numpGLfromGLtoGO, cp->numGL);
	arrayCopy<int>(pGLfromGLtoGO[0], state.pGLfromGLtoGO[0],
			cp->numGL*cp->maxnumpGLfromGLtoGO);

	arrayCopy<int>(numpGLfromGOtoGL, state.numpGLfromGOtoGL, cp->numGL);

	arrayCopy<int>(numpGLfromGLtoGR, state.numpGLfromGLtoGR, cp->numGL);
	arrayCopy<int>(pGLfromGLtoGR[0], state.pGLfromGLtoGR[0],
			cp->numGL*cp->maxnumpGLfromGOtoGL);

	arrayCopy<int>(numpMFfromMFtoGL, state.numpMFfromMFtoGL, cp->numMF);
	arrayCopy<int>(pMFfromMFtoGL[0], state.pMFfromMFtoGL[0],
			cp->numMF*20);

	arrayCopy<int>(numpMFfromMFtoGR, state.numpMFfromMFtoGR, cp->numMF);
	arrayCopy<int>(pMFfromMFtoGR[0], state.pMFfromMFtoGR[0],
			cp->numMF*cp->maxnumpMFfromMFtoGR);

	arrayCopy<int>(numpMFfromMFtoGO, state.numpMFfromMFtoGO, cp->numMF);
	arrayCopy<int>(pMFfromMFtoGO[0], state.pMFfromMFtoGO[0],
			cp->numMF*cp->maxnumpMFfromMFtoGO);

	arrayCopy<int>(numpGOfromGLtoGO, state.numpGOfromGLtoGO, cp->numGO);
	arrayCopy<int>(pGOfromGLtoGO[0], state.pGOfromGLtoGO[0],
			cp->numGO*cp->maxnumpGOfromGLtoGO);

	arrayCopy<int>(numpGOfromGOtoGL, state.numpGOfromGOtoGL, cp->numGO);
	arrayCopy<int>(pGOfromGOtoGL[0], state.pGOfromGOtoGL[0],
			cp->numGO*cp->maxnumpGOfromGOtoGL);

	arrayCopy<int>(numpGOfromMFtoGO, state.numpGOfromMFtoGO, cp->numGO);
	arrayCopy<int>(pGOfromMFtoGO[0], state.pGOfromMFtoGO[0],
			cp->numGO*16);

	arrayCopy<int>(numpGOfromGOtoGR, state.numpGOfromGOtoGR, cp->numGO);
	arrayCopy<int>(pGOfromGOtoGR[0], state.pGOfromGOtoGR[0],
			cp->numGO*cp->maxnumpGOfromGOtoGR);

	arrayCopy<int>(numpGOfromGRtoGO, state.numpGOfromGRtoGO, cp->numGO);
	arrayCopy<int>(pGOfromGRtoGO[0], state.pGOfromGRtoGO[0],
			cp->numGO*cp->maxnumpGOfromGRtoGO);

	arrayCopy<int>(numpGOGABAInGOGO, state.numpGOGABAInGOGO, cp->numGO);
	arrayCopy<int>(pGOGABAInGOGO[0], state.pGOGABAInGOGO[0],
			cp->numGO*cp->maxnumpGOGABAInGOGO);

	arrayCopy<int>(numpGOGABAOutGOGO, state.numpGOGABAOutGOGO, cp->numGO);
	arrayCopy<int>(pGOGABAOutGOGO[0], state.pGOGABAOutGOGO[0],
			cp->numGO*cp->maxGOGOsyn);

	arrayCopy<int>(numpGOCoupInGOGO, state.numpGOCoupInGOGO, cp->numGO);
	arrayCopy<int>(pGOCoupInGOGO[0], state.pGOCoupInGOGO[0],
			cp->numGO*49);

	arrayCopy<int>(numpGOCoupOutGOGO, state.numpGOCoupOutGOGO, cp->numGO);
	arrayCopy<int>(pGOCoupOutGOGO[0], state.pGOCoupOutGOGO[0],
			cp->numGO*49);

	arrayCopy<ct_uint32_t>(pGRDelayMaskfromGRtoBSP, state.pGRDelayMaskfromGRtoBSP, cp->numGR);

	arrayCopy<int>(numpGRfromGLtoGR, state.numpGRfromGLtoGR, cp->numGR);
	arrayCopy<int>(pGRfromGLtoGR[0], state.pGRfromGLtoGR[0],
			cp->numGR*cp->maxnumpGRfromGLtoGR);

	arrayCopy<int>(numpGRfromGRtoGO, state.numpGRfromGRtoGO, cp->numGR);
	arrayCopy<int>(pGRfromGRtoGO[0], state.pGRfromGRtoGO[0],
			cp->numGR*cp->maxnumpGRfromGRtoGO);
	arrayCopy<int>(pGRDelayMaskfromGRtoGO[0], state.pGRDelayMaskfromGRtoGO[0],
			cp->numGR*cp->maxnumpGRfromGRtoGO);

	arrayCopy<int>(numpGRfromGOtoGR, state.numpGRfromGOtoGR, cp->numGR);
	arrayCopy<int>(pGRfromGOtoGR[0], state.pGRfromGOtoGR[0],
			cp->numGR*cp->maxnumpGRfromGOtoGR);

	arrayCopy<int>(numpGRfromMFtoGR, state.numpGRfromMFtoGR, cp->numGR);
	arrayCopy<int>(pGRfromMFtoGR[0], state.pGRfromMFtoGR[0],
			cp->numGR*cp->maxnumpGRfromMFtoGR);
}

InNetConnectivityState::~InNetConnectivityState()
{
	delete[] numpGLfromGLtoGO;
	delete2DArray<int>(pGLfromGLtoGO);

	delete[] haspGLfromGOtoGL;
	delete[] numpGLfromGOtoGL;
	delete2DArray<int>(pGLfromGOtoGL);
	delete[] numpGLfromGLtoGR;
	delete2DArray<int>(pGLfromGLtoGR);
	delete[] spanArrayGRtoGLX;
	delete[] spanArrayGRtoGLY;
	delete[] xCoorsGRGL;
	delete[] yCoorsGRGL;
	std::cout << "check2" << std::endl;

	//mossy fibers
	delete[] haspGLfromMFtoGL;
	delete[] pGLfromMFtoGL;	
	delete[] numpMFfromMFtoGL;
	delete2DArray<int>(pMFfromMFtoGL);
	delete[] spanArrayMFtoGLX;
	delete[] spanArrayMFtoGLY;
	delete[] xCoorsMFGL;		
	delete[] yCoorsMFGL;		
	delete[] numpMFfromMFtoGR;
	delete2DArray<int>(pMFfromMFtoGR);
	delete[] numpMFfromMFtoGO;
	delete2DArray<int>(pMFfromMFtoGO); 	

	//golgi
	delete[] numpGOfromGLtoGO;
	delete2DArray<int>(pGOfromGLtoGO);
	delete[] numpGOfromGOtoGL;
	delete2DArray<int>(pGOfromGOtoGL);
	delete[] spanArrayGOtoGLY;
	delete[] spanArrayGOtoGLX;
	delete[] xCoorsGOGL;
	delete[] yCoorsGOGL;
	delete[] numpGOfromMFtoGO;
	delete2DArray<int>(pGOfromMFtoGO);
	delete[] numpGOfromGOtoGR;
	delete2DArray<int>(pGOfromGOtoGR);

	delete[] numpGOfromGRtoGO;
	delete2DArray<int>(pGOfromGRtoGO);
	delete[] spanArrayPFtoGOX;
	delete[] spanArrayPFtoGOY;
	delete[] spanArrayAAtoGOX;
	delete[] spanArrayAAtoGOY;
	delete[] xCoorsPFGO;
	delete[] yCoorsPFGO;
	delete[] xCoorsAAGO;
	delete[] yCoorsAAGO;

	delete[] numpGOGABAInGOGO;
	delete2DArray<int>(pGOGABAInGOGO);
	delete[] numpGOGABAOutGOGO;
	delete2DArray<int>(pGOGABAOutGOGO);
	delete2DArray<bool>(conGOGOBoolOut);
	delete[] spanArrayGOtoGOsynX;
	delete[] spanArrayGOtoGOsynY;
	delete[] xCoorsGOGOsyn;
	delete[] yCoorsGOGOsyn;

	delete[] numpGOCoupInGOGO;
	delete2DArray<int>(pGOCoupInGOGO);
	delete2DArray<float>(pGOCoupInGOGOCCoeff);
	delete[] numpGOCoupOutGOGO;
	delete2DArray<int>(pGOCoupOutGOGO);
	delete2DArray<float>(pGOCoupOutGOGOCCoeff);
	delete2DArray<bool>(gjConBool);
	
	delete[] spanArrayGOtoGOgjX;
	delete[] spanArrayGOtoGOgjY;
	delete[] xCoorsGOGOgj;
	delete[] yCoorsGOGOgj;
	delete[] gjPcon;
	delete[] gjCC;

	//granule
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

void InNetConnectivityState::writeState(std::fstream &outfile)
{
	std::cout << "Writing input network connectivity state to disk..." << std::endl;
	stateRW(false, (std::fstream &)outfile);
	std::cout << "finished writing input network connectivity to disk." << std::endl;
}

bool InNetConnectivityState::operator==(const InNetConnectivityState &compState)
{
	bool eq = true;
	//NOTE: haspGLfromMFtoGL and the like are excellent candidates for std::arrays,
	//      so that we can run STL algs on them	
	for (int i = 0; i < cp->numGL; i++)
	{
		eq = eq && (haspGLfromMFtoGL[i] == compState.haspGLfromMFtoGL[i]);
	}

	for(int i = 0; i < cp->numGO; i++)
	{
		eq = eq && (numpGOfromGOtoGR[i] == compState.numpGOfromGOtoGR[i]);
	}

	for (int i = 0; i < cp->numGR; i++)
	{
		eq = eq && (numpGRfromGOtoGR[i] == compState.numpGRfromGOtoGR[i]);
	}

	return eq;
}

bool InNetConnectivityState::operator!=(const InNetConnectivityState &compState)
{
	return !(*this == compState);
}

std::vector<int> InNetConnectivityState::getpGOfromGOtoGLCon(int goN)
{
	return getConCommon(goN, numpGOfromGOtoGL, pGOfromGOtoGL);
}

std::vector<int> InNetConnectivityState::getpGOfromGLtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromGLtoGO, pGOfromGLtoGO);
}

std::vector<int> InNetConnectivityState::getpMFfromMFtoGLCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGL, pMFfromMFtoGL);
}

std::vector<int> InNetConnectivityState::getpGLfromGLtoGRCon(int glN)
{
	return getConCommon(glN, numpGLfromGLtoGR, pGLfromGLtoGR);
}

std::vector<int> InNetConnectivityState::getpGRfromMFtoGR(int grN)
{
	return getConCommon(grN, numpGRfromMFtoGR, pGRfromMFtoGR);
}

std::vector<std::vector<int> > InNetConnectivityState::getpGRPopfromMFtoGR()
{
	std::vector<std::vector<int>> retVect;

	retVect.resize(cp->numGR);

	for (int i = 0; i < cp->numGR; i++)
	{
		retVect[i]=getpGRfromMFtoGR(i);
	}

	return retVect;
}

std::vector<int> InNetConnectivityState::getpGRfromGOtoGRCon(int grN)
{
	return getConCommon(grN, numpGRfromGOtoGR, pGRfromGOtoGR);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGRPopfromGOtoGRCon()
{
	return getPopConCommon(cp->numGR, numpGRfromGOtoGR, pGRfromGOtoGR);
}

std::vector<int> InNetConnectivityState::getpGRfromGRtoGOCon(int grN)
{
	return getConCommon(grN, numpGRfromGRtoGO, pGRfromGRtoGO);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGRPopfromGRtoGOCon()
{
	return getPopConCommon(cp->numGR, numpGRfromGRtoGO, pGRfromGRtoGO);
}

std::vector<int> InNetConnectivityState::getpGOfromGRtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromGRtoGO, pGOfromGRtoGO);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGOPopfromGRtoGOCon()
{
	return getPopConCommon(cp->numGO, numpGOfromGRtoGO, pGOfromGRtoGO);
}

std::vector<int> InNetConnectivityState::getpGOfromGOtoGRCon(int goN)
{
	return getConCommon(goN, numpGOfromGOtoGR, pGOfromGOtoGR);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGOPopfromGOtoGRCon()
{
	return getPopConCommon(cp->numGO, numpGOfromGOtoGR, pGOfromGOtoGR);
}

std::vector<int> InNetConnectivityState::getpGOOutGOGOCon(int goN)
{
	return getConCommon(goN, numpGOGABAOutGOGO, pGOGABAOutGOGO);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGOPopOutGOGOCon()
{
	return getPopConCommon(cp->numGO, numpGOGABAOutGOGO, pGOGABAOutGOGO);
}

std::vector<int> InNetConnectivityState::getpGOInGOGOCon(int goN)
{
	return getConCommon(goN, numpGOGABAInGOGO, pGOGABAInGOGO);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGOPopInGOGOCon()
{
	return getPopConCommon(cp->numGO, numpGOGABAInGOGO, pGOGABAInGOGO);
}

std::vector<int> InNetConnectivityState::getpGOCoupOutGOGOCon(int goN)
{
	return getConCommon(goN, numpGOCoupOutGOGO, pGOCoupOutGOGO);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGOPopCoupOutGOGOCon()
{
	return getPopConCommon(cp->numGO, numpGOCoupOutGOGO, pGOCoupOutGOGO);
}

std::vector<int> InNetConnectivityState::getpGOCoupInGOGOCon(int goN)
{
	return getConCommon(goN, numpGOCoupInGOGO, pGOCoupInGOGO);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGOPopCoupInGOGOCon()
{
	return getPopConCommon(cp->numGO, numpGOCoupInGOGO, pGOCoupInGOGO);
}

std::vector<int> InNetConnectivityState::getpMFfromMFtoGRCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGR, pMFfromMFtoGR);
}

std::vector<int> InNetConnectivityState::getpMFfromMFtoGOCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGO, pMFfromMFtoGO);
}

std::vector<int> InNetConnectivityState::getpGOfromMFtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromMFtoGO, pGOfromMFtoGO);
}

std::vector<std::vector<int>> InNetConnectivityState::getpGOPopfromMFtoGOCon()
{
	std::vector<std::vector<int>> con;

	con.resize(cp->numGO);
	for (int i = 0; i < cp->numGO; i++)
	{
		con[i] = getpGOfromMFtoGOCon(i);
	}

	return con;
}

std::vector<ct_uint32_t> InNetConnectivityState::getConCommon(int cellN, ct_int32_t *numpCellCon,
		ct_uint32_t **pCellCon)
{
	std::vector<ct_uint32_t> inds;
	inds.resize(numpCellCon[cellN]);
	for (int i = 0; i < numpCellCon[cellN]; i++)
	{
		inds[i] = pCellCon[cellN][i];
	}

	return inds;
}

std::vector<std::vector<ct_uint32_t>> InNetConnectivityState::getPopConCommon(int numCells,
		ct_int32_t *numpCellCon, ct_uint32_t **pCellCon)
{
	std::vector<std::vector<ct_uint32_t>> con;
	con.resize(numCells);
	for (int i = 0; i < numCells; i++)
	{
		con[i].insert(con[i].end(), &pCellCon[i][0], &pCellCon[i][numpCellCon[i]]);
	}

	return con;
}

std::vector<int> InNetConnectivityState::getConCommon(int cellN, int *numpCellCon, int **pCellCon)
{
	std::vector<int> inds;
	inds.resize(numpCellCon[cellN]);
	for (int i = 0; i < numpCellCon[cellN]; i++)
	{
		inds[i] = pCellCon[cellN][i];
	}

	return inds;
}

std::vector<std::vector<int>> InNetConnectivityState::getPopConCommon(int numCells, int *numpCellCon,
		int **pCellCon)
{
	std::vector<std::vector<int>> con;
	con.resize(numCells);
	for (int i = 0; i < numCells; i++)
	{
		con[i].insert(con[i].end(), &pCellCon[i][0], &pCellCon[i][numpCellCon[i]]);
	}

	return con;
}

std::vector<ct_uint32_t> InNetConnectivityState::getGOIncompIndfromGRtoGO()
{
	std::vector<ct_uint32_t> goInds;

	for (int i = 0; i < cp->numGO; i++)
	{
		if (numpGOfromGRtoGO[i] < cp->maxnumpGOfromGRtoGO)
		{
			goInds.push_back(i);
		}
	}

	return goInds;
}

std::vector<ct_uint32_t> InNetConnectivityState::getGRIncompIndfromGRtoGO()
{
	std::vector<ct_uint32_t> grInds;

	for (int i = 0; i < cp->numGR; i++)
	{
		if (numpGRfromGRtoGO[i] < cp->maxnumpGRfromGRtoGO)
		{
			grInds.push_back(i);
		}
	}

	return grInds;
}

bool InNetConnectivityState::deleteGOGOConPair(int srcGON, int destGON)
{
	bool hasCon = false;
	int conN;
	
	for (int i = 0; i < numpGOGABAOutGOGO[srcGON]; i++)
	{
		if (pGOGABAOutGOGO[srcGON][i] == destGON)
		{
			hasCon = true;
			conN = i;
			break;
		}
	}

	if (!hasCon)
	{
		return hasCon;
	}

	for (int i = conN; i < numpGOGABAOutGOGO[srcGON] - 1; i++)
	{
		pGOGABAOutGOGO[srcGON][i] = pGOGABAOutGOGO[srcGON][i + 1];
	}
	numpGOGABAOutGOGO[srcGON]--;

	for (int i = 0; i < numpGOGABAInGOGO[destGON]; i++)
	{
		if (pGOGABAInGOGO[destGON][i] == srcGON)
		{
			conN = i;
		}
	}

	for(int i = conN; i < numpGOGABAInGOGO[destGON] - 1; i++)
	{
		pGOGABAInGOGO[destGON][i] = pGOGABAInGOGO[destGON][i + 1];
	}
	numpGOGABAInGOGO[destGON]--;

	return hasCon;
}

bool InNetConnectivityState::addGOGOConPair(int srcGON, int destGON)
{
	if (numpGOGABAOutGOGO[srcGON] >= cp->maxnumpGOGABAOutGOGO ||
			numpGOGABAInGOGO[destGON] >= cp->maxnumpGOGABAInGOGO)
	{
		return false;
	}

	pGOGABAOutGOGO[srcGON][numpGOGABAOutGOGO[srcGON]] = destGON;
	numpGOGABAOutGOGO[srcGON]++;

	pGOGABAInGOGO[destGON][numpGOGABAInGOGO[destGON]] = srcGON;
	numpGOGABAInGOGO[destGON]++;

	return true;
}

void InNetConnectivityState::allocateMemory()
{

	numpGLfromGLtoGO = new int[cp->numGL];
	pGLfromGLtoGO = allocate2DArray<int>(cp->numGL, cp->maxnumpGLfromGLtoGO);

	haspGLfromGOtoGL = new int[cp->numGL];
	numpGLfromGOtoGL = new int[cp->numGL];
	pGLfromGOtoGL = allocate2DArray<int>(cp->numGL, cp->maxnumpGLfromGOtoGL);

	numpGLfromGLtoGR = new int[cp->numGL];
	pGLfromGLtoGR = allocate2DArray<int>(cp->numGL, cp->maxnumpGLfromGLtoGR);
	spanArrayGRtoGLX = new int[5];
	spanArrayGRtoGLY = new int[5];
	xCoorsGRGL = new int[25];
	yCoorsGRGL = new int[25];
	
	//mf
	haspGLfromMFtoGL = new int[cp->numGL];
	pGLfromMFtoGL = new int[cp->numGL];
	numpMFfromMFtoGL = new int[cp->numMF];
	pMFfromMFtoGL = allocate2DArray<int>(cp->numMF, 40);
	spanArrayMFtoGLX = new int[cp->glX];
	spanArrayMFtoGLY = new int[cp->glY];
	xCoorsMFGL = new int[cp->numpMFtoGL];
	yCoorsMFGL = new int[cp->numpMFtoGL];

	numpMFfromMFtoGR = new int[cp->numMF];
	pMFfromMFtoGR=allocate2DArray<int>(cp->numMF, cp->maxnumpMFfromMFtoGR);
	numpMFfromMFtoGO = new int[cp->numMF];
	pMFfromMFtoGO = allocate2DArray<int>(cp->numMF, cp->maxnumpMFfromMFtoGO);

	//golgi
	numpGOfromGLtoGO = new int[cp->numGO];
	pGOfromGLtoGO = allocate2DArray<int>(cp->numGO, cp->maxnumpGOfromGLtoGO);

	numpGOfromGOtoGL = new int[cp->numGO];
	pGOfromGOtoGL = allocate2DArray<int>(cp->numGO, cp->maxnumpGOfromGOtoGL);
	spanArrayGOtoGLX = new int[cp->spanGOtoGLX+1];
	spanArrayGOtoGLY = new int [cp->spanGOtoGLY+1];
	xCoorsGOGL = new int[cp->numpGOGL];
	yCoorsGOGL = new int[cp->numpGOGL];
	PconGOGL = new float[cp->numpGOGL];

	numpGOfromMFtoGO = new int[cp->numGO];
	pGOfromMFtoGO = allocate2DArray<int>(cp->numGO, 40);

	numpGOfromGOtoGR = new int[cp->numGO];
	pGOfromGOtoGR = allocate2DArray<int>(cp->numGO, cp->maxnumpGOfromGOtoGR);

	numpGOfromGRtoGO = new int[cp->numGO];
	pGOfromGRtoGO = allocate2DArray<int>(cp->numGO, cp->maxnumpGOfromGRtoGO);
	spanArrayPFtoGOX = new int[cp->grX + 1];
	spanArrayPFtoGOY = new int[150 + 1];
	xCoorsPFGO = new int[(cp->grX + 1)*(151)];
	yCoorsPFGO = new int[(cp->grX + 1)*(151)];	
	spanArrayAAtoGOX = new int[202];
	spanArrayAAtoGOY = new int[202];
	xCoorsAAGO = new int[202*202];
	yCoorsAAGO = new int[202*202];

	numpGOCoupInGOGO =new int[cp->numGO];
	pGOCoupInGOGO = allocate2DArray<int>(cp->numGO, 81);
	pGOCoupInGOGOCCoeff = allocate2DArray<float>(cp->numGO, 81);
	numpGOCoupOutGOGO = new int[cp->numGO];
	pGOCoupOutGOGO = allocate2DArray<int>(cp->numGO, 81);
	pGOCoupOutGOGOCCoeff = allocate2DArray<float>(cp->numGO, 81);
	gjConBool = allocate2DArray<bool>(cp->numGO, cp->numGO);
	spanArrayGOtoGOgjX = new int[9];
	spanArrayGOtoGOgjY = new int[9];
	xCoorsGOGOgj = new int[9*9];
	yCoorsGOGOgj = new int[9*9];
	gjPcon = new float[9*9];
	gjCC = new float[9*9];

	//granule
	pGRDelayMaskfromGRtoBSP = new ct_uint32_t[cp->numGR];

	numpGRfromGLtoGR = new int[cp->numGR];
	pGRfromGLtoGR = allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromGLtoGR);

	numpGRfromGRtoGO = new int[cp->numGR];
	pGRfromGRtoGO = allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromGRtoGO);
	pGRDelayMaskfromGRtoGO = allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromGRtoGO);

	numpGRfromGOtoGR = new int[cp->numGR];
	pGRfromGOtoGR = allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromGOtoGR);

	numpGRfromMFtoGR = new int[cp->numGR];
	pGRfromMFtoGR = allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromMFtoGR);
}

void InNetConnectivityState::stateRW(bool read, std::fstream &file)
{
	std::cout << "glomerulus" << std::endl;
	//glomerulus
	rawBytesRW((char *)haspGLfromMFtoGL, cp->numGL*sizeof(int), read, file);
	std::cout << "glomerulus 1.1" << std::endl;
	rawBytesRW((char *)pGLfromMFtoGL, cp->numGL*sizeof(int), read, file);

	std::cout << "glomerulus 2" << std::endl;
	rawBytesRW((char *)numpGLfromGLtoGO, cp->numGL*sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGO[0], cp->numGL*cp->maxnumpGLfromGLtoGO*sizeof(int), read, file);

	std::cout << "glomerulus 3" << std::endl;
	rawBytesRW((char *)numpGLfromGOtoGL, cp->numGL*sizeof(int), read, file);
	rawBytesRW((char *)haspGLfromGOtoGL, cp->numGL*sizeof(int), read, file);
	//rawBytesRW((char *)pGLfromGOtoGL, cp->numGL*sizeof(int), read, file);

	std::cout << "glomerulus 4" << std::endl;
	rawBytesRW((char *)numpGLfromGLtoGR, cp->numGL*sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGR[0], cp->numGL*cp->maxnumpGLfromGLtoGR*sizeof(int), read, file);

	std::cout << "mf" << std::endl;
	//mossy fibers
	rawBytesRW((char *)numpMFfromMFtoGL, cp->numMF*sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGL[0], cp->numMF*20*sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGR, cp->numMF*sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGR[0], cp->numMF*cp->maxnumpMFfromMFtoGR*sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGO, cp->numMF*sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGO[0], cp->numMF*cp->maxnumpMFfromMFtoGO*sizeof(int), read, file);

	std::cout << "golgi" << std::endl;
	//golgi
	rawBytesRW((char *)numpGOfromGLtoGO, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGLtoGO[0], cp->numGO*cp->maxnumpGOfromGLtoGO*sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGL, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGL[0], cp->numGO*cp->maxnumpGOfromGOtoGL*sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromMFtoGO, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOfromMFtoGO[0], cp->numGO*16*sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGR, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGR[0], cp->numGO*cp->maxnumpGOfromGOtoGR*sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGRtoGO, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGRtoGO[0], cp->numGO*cp->maxnumpGOfromGRtoGO*sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAInGOGO, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAInGOGO[0], cp->numGO*cp->maxnumpGOGABAInGOGO*sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAOutGOGO, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAOutGOGO[0], cp->numGO*cp->maxnumpGOGABAOutGOGO*sizeof(int), read, file);
	
	rawBytesRW((char *)numpGOCoupInGOGO, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupInGOGO[0], cp->numGO*49*sizeof(int), read, file);

	rawBytesRW((char *)numpGOCoupOutGOGO, cp->numGO*sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupOutGOGO[0], cp->numGO*49*sizeof(int), read, file);

	std::cout << "granule" << std::endl;
	//granule
	rawBytesRW((char *)pGRDelayMaskfromGRtoBSP, cp->numGR*sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromGLtoGR, cp->numGR*sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGLtoGR[0], cp->numGR*cp->maxnumpGRfromGLtoGR*sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGRtoGO, cp->numGR*sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGRtoGO[0], cp->maxnumpGRfromGRtoGO*cp->numGR*sizeof(int), read, file);
	rawBytesRW((char *)pGRDelayMaskfromGRtoGO[0], cp->maxnumpGRfromGRtoGO*cp->numGR*sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGOtoGR, cp->numGR*sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGOtoGR[0], cp->maxnumpGRfromGOtoGR*cp->numGR*sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromMFtoGR, cp->numGR*sizeof(int), read, file);
	rawBytesRW((char *)pGRfromMFtoGR[0], cp->maxnumpGRfromMFtoGR*cp->numGR*sizeof(int), read, file);
}

void InNetConnectivityState::initializeVals()
{
	std::fill(haspGLfromMFtoGL, haspGLfromMFtoGL + cp->numGL, 0);
	std::fill(pGLfromMFtoGL, pGLfromMFtoGL + cp->numGL, UINT_MAX);

	std::fill(numpGLfromGLtoGO, numpGLfromGLtoGO + cp->numGL, 0);
	std::fill(pGLfromGLtoGO[0], pGLfromGLtoGO[0] +
			cp->numGL*cp->maxnumpGLfromGLtoGO, UINT_MAX);

	std::fill(haspGLfromGOtoGL, haspGLfromGOtoGL + cp->numGL, 0);
	std::fill(numpGLfromGOtoGL, numpGLfromGOtoGL + cp->numGL, 0);
	std::fill(pGLfromGOtoGL[0], pGLfromGOtoGL[0] +
			cp->numGL*cp->maxnumpGLfromGOtoGL, UINT_MAX);

	std::fill(numpGLfromGLtoGR, numpGLfromGLtoGR + cp->numGL, 0);
	std::fill(pGLfromGLtoGR[0], pGLfromGLtoGR[0] +
			cp->numGL*cp->maxnumpGLfromGOtoGL, UINT_MAX);

	std::fill(spanArrayGRtoGLX, spanArrayGRtoGLX + 5, 0);
	std::fill(spanArrayGRtoGLY, spanArrayGRtoGLY + 5, 0);

	std::fill(xCoorsGRGL, xCoorsGRGL + 25, 0);
	std::fill(yCoorsGRGL, yCoorsGRGL + 25, 0);

	//mf
	std::fill(haspGLfromMFtoGL, haspGLfromMFtoGL + cp->numGL, 0);

	std::fill(numpMFfromMFtoGL, numpMFfromMFtoGL + cp->numMF, 0);
	std::fill(pMFfromMFtoGL[0], pMFfromMFtoGL[0] + cp->numMF * 20, UINT_MAX);
	std::fill(spanArrayMFtoGLX, spanArrayMFtoGLX + cp->glX, 0);
	std::fill(spanArrayMFtoGLY, spanArrayMFtoGLY + cp->glY, 0);
	std::fill(xCoorsMFGL, xCoorsMFGL + cp->numpMFtoGL, 0);
	std::fill(yCoorsMFGL, yCoorsMFGL + cp->numpMFtoGL, 0);

	std::fill(numpMFfromMFtoGR, numpMFfromMFtoGR + cp->numMF, 0);
	std::fill(pMFfromMFtoGR[0], pMFfromMFtoGR[0] +
			cp->numMF * cp->maxnumpMFfromMFtoGR, UINT_MAX);
	std::fill(numpMFfromMFtoGO, numpMFfromMFtoGO + cp->numMF, 0);
	std::fill(pMFfromMFtoGO[0], pMFfromMFtoGO[0] +
			cp->numMF * cp->maxnumpMFfromMFtoGO, UINT_MAX);

	std::fill(numpGOfromGLtoGO, numpGOfromGLtoGO + cp->numGO, 0);
	std::fill(pGOfromGLtoGO[0], pGOfromGLtoGO[0] +
			cp->numGO * cp->maxnumpGOfromGLtoGO, UINT_MAX);

	std::fill(numpGOfromGOtoGL, numpGOfromGOtoGL + cp->numGO, 0);
	std::fill(pGOfromGOtoGL[0], pGOfromGOtoGL[0] +
			cp->numGO * cp->maxnumpGOfromGOtoGL, UINT_MAX);
	std::fill(PconGOGL, PconGOGL + cp->numpGOGL, 0);

	std::fill(numpGOfromMFtoGO, numpGOfromMFtoGO + cp->numGO, 0);
	std::fill(pGOfromMFtoGO[0], pGOfromMFtoGO[0] + cp->numGO * 16, UINT_MAX);

	std::fill(numpGOfromGOtoGR, numpGOfromGOtoGR + cp->numGO, 0);
	std::fill(pGOfromGOtoGR[0], pGOfromGOtoGR[0] +
			cp->numGO * cp->maxnumpGOfromGOtoGR, UINT_MAX);

	std::fill(numpGOfromGRtoGO, numpGOfromGRtoGO + cp->numGO, 0);
	std::fill(pGOfromGRtoGO[0], pGOfromGRtoGO[0] +
			cp->numGO * cp->maxnumpGOfromGRtoGO, UINT_MAX);
	
	std::fill(spanArrayPFtoGOX, spanArrayPFtoGOX + cp->grX + 1, 0);
	std::fill(spanArrayPFtoGOY, spanArrayPFtoGOY + 121, 0);
	std::fill(xCoorsPFGO, xCoorsPFGO + (cp->grX + 1) * 121, 0);
	std::fill(yCoorsPFGO, yCoorsPFGO + (cp->grX + 1) * 121, 0);
	
	std::fill(spanArrayAAtoGOX, spanArrayAAtoGOX + 202, 0);
	std::fill(spanArrayAAtoGOY, spanArrayAAtoGOY + 202, 0);
	std::fill(xCoorsAAGO, xCoorsAAGO + 202 * 202, 0);
	std::fill(yCoorsAAGO, yCoorsAAGO + 202 * 202, 0);

	std::fill(numpGOCoupInGOGO, numpGOCoupInGOGO + cp->numGO, 0);
	std::fill(pGOCoupInGOGO[0], pGOCoupInGOGO[0] + cp->numGO * 81, UINT_MAX);
	std::fill(pGOCoupInGOGOCCoeff[0], pGOCoupInGOGOCCoeff[0] + cp->numGO * 81, UINT_MAX);
	std::fill(numpGOCoupOutGOGO, numpGOCoupOutGOGO + cp->numGO, 0);
	std::fill(pGOCoupOutGOGO[0], pGOCoupOutGOGO[0] + cp->numGO * 81, UINT_MAX);
	std::fill(pGOCoupOutGOGOCCoeff[0], pGOCoupOutGOGOCCoeff[0] + cp->numGO * 81, UINT_MAX);
	std::fill(gjConBool[0], gjConBool[0] + cp->numGO * cp->numGO, false);
	
	std::fill(spanArrayGOtoGOgjX, spanArrayGOtoGOgjX + 9, 0);
	std::fill(spanArrayGOtoGOgjY, spanArrayGOtoGOgjY + 9, 0);
	std::fill(xCoorsGOGOgj, xCoorsGOGOgj + 9 * 9, 0);
	std::fill(yCoorsGOGOgj, yCoorsGOGOgj + 9 * 9, 0);
	std::fill(gjPcon, gjPcon + 9 * 9, 0);
	std::fill(gjCC, gjCC + 9 * 9, 0);
	
	std::fill(pGRDelayMaskfromGRtoBSP, pGRDelayMaskfromGRtoBSP + cp->numGR, 0);

	std::fill(numpGRfromGLtoGR, numpGRfromGLtoGR + cp->numGR, 0);
	std::fill(pGRfromGLtoGR[0], pGRfromGLtoGR[0] +
			cp->numGR * cp->maxnumpGRfromGLtoGR, UINT_MAX);

	std::fill(numpGRfromGRtoGO, numpGRfromGRtoGO + cp->numGR, 0);
	std::fill(pGRfromGRtoGO[0], pGRfromGRtoGO[0] +
			cp->numGR * cp->maxnumpGRfromGRtoGO, UINT_MAX);
	std::fill(pGRDelayMaskfromGRtoGO[0], pGRDelayMaskfromGRtoGO[0] +
			cp->numGR * cp->maxnumpGRfromGRtoGO, UINT_MAX);

	std::fill(numpGRfromGOtoGR, numpGRfromGOtoGR + cp->numGR, 0);
	std::fill(pGRfromGOtoGR[0], pGRfromGOtoGR[0] +
			cp->numGR * cp->maxnumpGRfromGOtoGR, UINT_MAX);

	std::fill(numpGRfromMFtoGR, numpGRfromMFtoGR + cp->numGR, 0);
	std::fill(pGRfromMFtoGR[0], pGRfromMFtoGR[0] +
			cp->numGR * cp->maxnumpGRfromMFtoGR, UINT_MAX);
}

void InNetConnectivityState::connectGLUBC()
{
	for (int i = 0; i < cp->spanGLtoUBCX + 1; i++)
	{
		int ind = cp->spanGLtoUBCX - i;
		spanArrayGLtoUBCX[i] = (cp->spanGLtoUBCX / 2) - ind;
	}
	for (int i = 0; i < cp->spanGLtoUBCY + 1; i++)
	{
		int ind = cp->spanGLtoUBCY - i;
		spanArrayGLtoUBCY[i] = (cp->spanGLtoUBCY / 2) - ind;
	}
		
	for (int i = 0; i < cp->numpGLtoUBC; i++)
	{
		xCoorsGLUBC[i] = spanArrayGLtoUBCX[i % (cp->spanGLtoUBCX + 1)];
		yCoorsGLUBC[i] = spanArrayGLtoUBCY[i / (cp->spanGLtoUBCX + 1)];		
	}

	// Grid Scale: Complete
	float gridXScaleSrctoDest = (float)cp->ubcX / (float)cp->glX; 
	float gridYScaleSrctoDest = (float)cp->ubcY / (float)cp->glY; 

	// Make Random Mossy Fiber Index Array: Complete	
	std::vector<int> rUBCInd;
	rUBCInd.assign(cp->numUBC,0);
	for (int i = 0; i < cp->numUBC; i++)
	{
		rUBCInd[i] = i;
	}

	std::random_shuffle(rUBCInd.begin(), rUBCInd.end());

	//Make Random Span Array: Complete
	std::vector<int> rUBCSpanInd;
	rUBCSpanInd.assign(cp->numpGLtoUBC,0);
	for (int ind = 0; ind < cp->numpGLtoUBC; ind++)
	{
		rUBCSpanInd[ind] = ind;
	}
	
	std::random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;
	int UBCInputs = 1;

	int glX = cp->glX;
	int glY = cp->glY;

	for (int attempts = 0; attempts < 4; attempts++)
	{
		std::random_shuffle(rUBCInd.begin(), rUBCInd.end());	

		for (int i = 0; i < cp->numUBC; i++)
		{	
			//Select MF Coordinates from random index array: Complete	
			srcPosX = rUBCInd[i] % cp->ubcX;
			srcPosY = rUBCInd[i] / cp->ubcX;		
			
			std::random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	

			for (int j = 0; j < cp->numpGLtoUBC; j++)
			{	
				
				if (numpUBCfromGLtoUBC[rUBCInd[i]] == UBCInputs) break; 
				
				preDestPosX = xCoorsGLUBC[rUBCSpanInd[j]]; 
				preDestPosY = yCoorsGLUBC[rUBCSpanInd[j]];	

				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;	
				
				tempDestPosX = (tempDestPosX%glX + glX) % glX;
				tempDestPosY = (tempDestPosY%glY+glY) % glY;
				
				destInd = tempDestPosY * glX + tempDestPosX;
					
				pUBCfromGLtoUBC[rUBCInd[i]] = destInd;
				numpUBCfromGLtoUBC[rUBCInd[i]]++;
					
				pGLfromGLtoUBC[destInd] = rUBCInd[i];
				numpGLfromGLtoUBC[destInd]++;		
			}
		}
	}

	int totalGLtoUBC = 0;
	
	for (int i = 0; i < cp->numUBC; i++)
	{
		totalGLtoUBC += numpUBCfromGLtoUBC[i];
	}
	std::cout << "Total GL to UBC connections:" << totalGLtoUBC << std::endl; 
}

void InNetConnectivityState::connectGRGL(CRandomSFMT *randGen)
{
	connectCommon(pGRfromGLtoGR, numpGRfromGLtoGR,
			pGLfromGLtoGR, numpGLfromGLtoGR,
			cp->maxnumpGRfromGLtoGR, cp->numGR,
			cp->maxnumpGLfromGLtoGR, cp->lownumpGLfromGLtoGR,
			cp->grX, cp->grY, cp->glX, cp->glY,
			cp->spanGRDenOnGLX, cp->spanGRDenOnGLY,
			20000, 50000, true,
			randGen);

	int count = 0;

	for (int i = 0; i < cp->numGL; i++)
	{
		count += numpGLfromGLtoGR[i];
	}

	std::cout << "Total number of Glomeruli to Granule connections:	" << count << std::endl; 
	std::cout << "Correct number: " << cp->numGR*cp->maxnumpGRfromGLtoGR << std::endl;
}

void InNetConnectivityState::connectGOGL(CRandomSFMT *randGen)
{
	connectCommon(pGOfromGLtoGO, numpGOfromGLtoGO,
			pGLfromGLtoGO, numpGLfromGLtoGO,
			cp->maxnumpGOfromGLtoGO, cp->numGO,
			cp->maxnumpGLfromGLtoGO, cp->maxnumpGLfromGLtoGO,
			cp->goX, cp->goY, cp->glX, cp->glY,
			cp->spanGODecDenOnGLX, cp->spanGODecDenOnGLY,
			20000, 50000, false,
			randGen);

	int initialGOOutput = 1;
	float sigmaML = 100;//10.5;
	float sigmaS = 100;//10.5;
	float A = 0.01;//0.095;
	float PconX;
	float PconY;

	// Make span Array
	for (int i = 0; i < cp->spanGOGLX + 1; i++)
	{
		int ind = cp->spanGOGLX - i;
		spanArrayGOtoGLX[i] = (cp->spanGOGLX / 2) - ind;
	}

	for (int i = 0; i < cp->spanGOGLY + 1; i++)
	{
		int ind = cp->spanGOGLY - i;
		spanArrayGOtoGLY[i] = (cp->spanGOGLY / 2) - ind;
	}
		
	for (int i = 0; i < cp->numpGOGL; i++)
	{
		xCoorsGOGL[i] = spanArrayGOtoGLX[i % (cp->spanGOGLX + 1)];
		yCoorsGOGL[i] = spanArrayGOtoGLY[i / (cp->spanGOGLY + 1)];	
	}

	//Make Random Golgi cell Index Array	
	std::vector<int> rGOInd;
	rGOInd.assign(cp->numGO,0);
	for (int i = 0; i < cp->numGO; i++)
	{
		rGOInd[i] = i;
	}

	std::random_shuffle(rGOInd.begin(), rGOInd.end());

	//Make Random Span Array
	std::vector<int> rGOSpanInd;
	rGOSpanInd.assign(cp->numpGOGL,0);
	for (int ind = 0; ind < cp->numpGOGL; ind++) rGOSpanInd[ind] = ind;
	
	// Probability of connection as a function of distance
	for (int i = 0; i < cp->numpGOGL; i++)
	{
		PconX = (xCoorsGOGL[i] * xCoorsGOGL[i]) / (2 * sigmaML*sigmaML);
		PconY = (yCoorsGOGL[i] * yCoorsGOGL[i]) / (2 * sigmaS*sigmaS);
		PconGOGL[i] = A * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < cp->numpGOGL; i++)
	{
		if ((xCoorsGOGL[i] == 0) && (yCoorsGOGL[i] == 0))
		{
			PconGOGL[i] = 0;
		}
	}

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	float gridXScaleSrctoDest = (float)cp->goX / (float)cp->glX;
	float gridYScaleSrctoDest = (float)cp->goY / (float)cp->glY;
	
	int shitCounter;

	for (int attempts = 0; attempts < 100; attempts++)
	{
		std::random_shuffle(rGOSpanInd.begin(), rGOSpanInd.end());	
		
		if (shitCounter == 0 ) break; 

		// Go through each golgi cell 
		for (int i = 0; i < cp->numGO; i++)
		{
			//Select GO Coordinates from random index array: Complete	
			srcPosX = rGOInd[i] % cp->goX;
			srcPosY = rGOInd[i] / cp->goX;	
			
			std::random_shuffle(rGOSpanInd.begin(), rGOSpanInd.end());	
			
			for (int j = 0; j < cp->numpGOGL; j++)   
			{	
				// relative position of connection
				preDestPosX = xCoorsGOGL[rGOSpanInd[j]];
				preDestPosY = yCoorsGOGL[rGOSpanInd[j]];	

				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX; 
				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;

				tempDestPosX = (tempDestPosX % cp->glX + cp->glX) % cp->glX;  
				tempDestPosY = (tempDestPosY % cp->glY + cp->glY) % cp->glY;
				
				// Change position to Index	
				destInd = tempDestPosY*cp->glX+tempDestPosX;
					
				if (numpGOfromGOtoGL[rGOInd[i]] >= initialGOOutput + attempts) break; 
				if ( randGen->Random() >= 1 - PconGOGL[rGOSpanInd[j]] && 
						numpGLfromGOtoGL[destInd] < cp->maxnumpGLfromGOtoGL) 
				{	
					pGOfromGOtoGL[rGOInd[i]][numpGOfromGOtoGL[rGOInd[i]]] = destInd;
					numpGOfromGOtoGL[rGOInd[i]]++;

					pGLfromGOtoGL[destInd][numpGLfromGOtoGL[destInd]] = rGOInd[i];
					numpGLfromGOtoGL[destInd]++;
				}
			}
		}

		shitCounter = 0;
		totalGOGL = 0;

		for (int i = 0; i < cp->numGL; i++)
		{
			if (numpGLfromGOtoGL[i] < cp->maxnumpGLfromGOtoGL) shitCounter++;
			totalGOGL += numpGLfromGOtoGL[i];
		}
	}

	for (int i = 0; i < 20; i++)
	{
		for (int j = 0; j < numpGLfromGOtoGL[i]; j++)
		{
			std::cout << pGLfromGOtoGL[i][j] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "Empty Glomeruli Counter: " << shitCounter << std::endl;
	std::cout << "Total GO -> GL: " << totalGOGL << std::endl;
	std::cout << "avg Num  GO -> GL Per GL: " << (float)totalGOGL / (float)cp->numGL << std::endl;
}

void InNetConnectivityState::connectUBCGL()
{
	for (int i = 0; i < cp->spanUBCtoGLX + 1; i++)
	{
		int ind = cp->spanUBCtoGLX - i;
		spanArrayUBCtoGLX[i] = (cp->spanUBCtoGLX / 2) - ind;
	}

	for (int i = 0; i < cp->spanUBCtoGLY + 1; i++)
	{
		int ind = cp->spanUBCtoGLY - i;
		spanArrayUBCtoGLY[i] = (cp->spanUBCtoGLY / 2) - ind;
	}
		
	for(int i = 0; i < cp->numpUBCtoGL; i++)
	{
		xCoorsUBCGL[i] = spanArrayUBCtoGLX[i % (cp->spanUBCtoGLX + 1)];
		yCoorsUBCGL[i] = spanArrayUBCtoGLY[i / (cp->spanUBCtoGLX + 1)];		
	}

	// Grid Scale: Complete
	float gridXScaleSrctoDest = (float)cp->ubcX / (float)cp->glX; 
	float gridYScaleSrctoDest = (float)cp->ubcY / (float)cp->glY; 

	//Make Random Mossy Fiber Index Array: Complete	
	std::vector<int> rUBCInd;
	rUBCInd.assign(cp->numUBC,0);
	
	for (int i = 0; i < cp->numUBC; i++)
	{
		rUBCInd[i] = i;
	}
	std::random_shuffle(rUBCInd.begin(), rUBCInd.end());

	//Make Random Span Array: Complete
	std::vector<int> rUBCSpanInd;
	rUBCSpanInd.assign(cp->numpUBCtoGL,0);
	for (int ind = 0; ind < cp->numpUBCtoGL; ind++) rUBCSpanInd[ind] = ind;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	int glX = cp->glX;
	int glY = cp->glY;

	int UBCOutput = 10;

	for (int attempts = 0; attempts < 3; attempts++)
	{
		std::random_shuffle(rUBCInd.begin(), rUBCInd.end());	

		for (int i = 0; i < cp->numUBC; i++)
		{	
			//Select MF Coordinates from random index array: Complete	
			srcPosX = rUBCInd[i] % cp->ubcX;
			srcPosY = rUBCInd[i] / cp->ubcX;		

			std::random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	
			
			for (int j = 0; j < cp->numpUBCtoGL; j++)
			{	
				preDestPosX = xCoorsUBCGL[rUBCSpanInd[j]]; 
				preDestPosY = yCoorsUBCGL[rUBCSpanInd[j]];	

				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;	
				
				tempDestPosX = ((tempDestPosX%glX + glX) % glX);
				tempDestPosY = ((tempDestPosY%glY + glY) % glY);
				
				destInd = tempDestPosY*glX+tempDestPosX;
				
				if (destInd == pUBCfromGLtoUBC[rUBCInd[i]] ||
						numpUBCfromUBCtoGL[rUBCInd[i]] == UBCOutput) break; 
				
				if (numpGLfromUBCtoGL[destInd] == 0) 
				{	
					pUBCfromUBCtoGL[ rUBCInd[i] ][ numpUBCfromUBCtoGL[rUBCInd[i]] ] = destInd;
					numpUBCfromUBCtoGL[ rUBCInd[i] ]++;	

					pGLfromUBCtoGL[destInd][ numpGLfromUBCtoGL[destInd] ] = rUBCInd[i];
					numpGLfromUBCtoGL[destInd]++;	
				
				}
			}
		}

	}	
}

void InNetConnectivityState::connectMFGL_withUBC(CRandomSFMT *randGen)
{

	int initialMFOutput = 14;

	//Make Span Array: Complete	
	for (int i = 0; i < cp->spanMFtoGLX + 1;i++)
	{
		int ind = cp->spanMFtoGLX - i;
		spanArrayMFtoGLX[i] = (cp->spanMFtoGLX / 2) - ind;
	}

	for(int i = 0; i < cp->spanMFtoGLY + 1; i++)
	{
		int ind = cp->spanMFtoGLY - i;
		spanArrayMFtoGLY[i] = (cp->spanMFtoGLY / 2) - ind;
	}
		
	for (int i = 0; i < cp->numpMFtoGL; i++)
	{
		xCoorsMFGL[i] = spanArrayMFtoGLX[i % (cp->spanMFtoGLX + 1)];
		yCoorsMFGL[i] = spanArrayMFtoGLY[i / (cp->spanMFtoGLX + 1)];		
	}

	// Grid Scale: Complete
	float gridXScaleSrctoDest = (float)cp->mfX / (float)cp->glX; 
	float gridYScaleSrctoDest = (float)cp->mfY / (float)cp->glY; 

	//Make Random Mossy Fiber Index Array: Complete	
	std::vector<int> rMFInd;
	rMFInd.assign(cp->numMF,0);
	
	for (int i = 0; i < cp->numMF; i++)
	{
		rMFInd[i] = i;	
	}

	std::random_shuffle(rMFInd.begin(), rMFInd.end());

	//Make Random Span Array: Complete
	std::vector<int> rMFSpanInd;
	rMFSpanInd.assign(cp->numpMFtoGL,0);
	
	for (int ind = 0; ind < cp->numpMFtoGL; ind++) rMFSpanInd[ind] = ind;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	int glX = cp->glX;
	int glY = cp->glY;
	int shitCounter;
	
	for(int attempts = 0; attempts < 3; attempts++)
	{
		std::random_shuffle(rMFInd.begin(), rMFInd.end());	

		for (int i = 0; i < cp->numMF; i++)
		{	
			//Select MF Coordinates from random index array: Complete	
			srcPosX = rMFInd[i] % cp->mfX;
			srcPosY = rMFInd[i] / cp->mfX;		
			
			std::random_shuffle(rMFSpanInd.begin(), rMFSpanInd.end());	
			
			for (int j = 0; j < cp->numpMFtoGL; j++)
			{	
				preDestPosX = xCoorsMFGL[ rMFSpanInd[j] ]; 
				preDestPosY = yCoorsMFGL[ rMFSpanInd[j] ];	

				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;	
				
				tempDestPosX = (tempDestPosX%glX+glX) % glX;
				tempDestPosY = (tempDestPosY%glY+glY) % glY;
				
				destInd = tempDestPosY * glX + tempDestPosX;
				
				if ( numpMFfromMFtoGL[rMFInd[i]] == initialMFOutput + attempts ) break;
					
				if ( !haspGLfromMFtoGL[destInd] && numpGLfromUBCtoGL[destInd] == 0 ) 
				{	
					pMFfromMFtoGL[rMFInd[i]][numpMFfromMFtoGL[rMFInd[i]]] = destInd;
					numpMFfromMFtoGL[rMFInd[i]]++;
					
					pGLfromMFtoGL[destInd] = rMFInd[i];
					haspGLfromMFtoGL[destInd] = true;	
				}
			}
		}
		
		shitCounter = 0;
		
		for (int i = 0; i < cp->numGL; i++)
		{
			if (!haspGLfromMFtoGL[i]) shitCounter++;
		}
	}	
	
	std::cout << "Empty Glomeruli Counter: " << shitCounter << std::endl << std::endl;
	
	int count = 0;
	
	for (int i = 0; i < cp->numMF; i++)
	{
		count += numpMFfromMFtoGL[i];
	}

	std::cout << "Total number of Mossy Fiber to Glomeruli connections:	" << count << std::endl;
	std::cout << "Correct number: " << cp->numGL << std::endl;

	count = 0;

	for (int i = 0; i < cp->numMF; i++)
	{
		for (int j = 0; j < numpMFfromMFtoGL[i]; j++)
		{
			for (int k = 0; k < numpMFfromMFtoGL[i]; k++)
			{
				if (pMFfromMFtoGL[i][j] == pMFfromMFtoGL[i][k] && j != k) count++; 
			}
		}
	}

	std::cout << "Double Mossy Fiber to Glomeruli connecitons: " << count << std::endl;
}

void InNetConnectivityState::connectMFGL_noUBC(CRandomSFMT *randGen)
{

	std::cout << cp->mfX   << std::endl;
	std::cout << cp->mfY   << std::endl;
	std::cout << cp->numMF << std::endl;

	int initialMFOutput = 14;

	//Make Span Array: Complete	
	for(int i = 0; i < cp->spanMFtoGLX + 1;i++)
	{
		int ind = cp->spanMFtoGLX - i;
		spanArrayMFtoGLX[i] = (cp->spanMFtoGLX / 2) - ind;
	}

	for(int i = 0; i < cp->spanMFtoGLY + 1;i++)
	{
		int ind = cp->spanMFtoGLY - i;
		spanArrayMFtoGLY[i] = (cp->spanMFtoGLY / 2) - ind;
	}
		
	for(int i = 0; i < cp->numpMFtoGL; i++)
	{
		xCoorsMFGL[i] = spanArrayMFtoGLX[i % (cp->spanMFtoGLX+1)];
		yCoorsMFGL[i] = spanArrayMFtoGLY[i / (cp->spanMFtoGLX+1)];		
	}

	// Grid Scale: Complete
	float gridXScaleSrctoDest = (float)cp->mfX / (float)cp->glX; 
	float gridYScaleSrctoDest = (float)cp->mfY / (float)cp->glY; 

	//Make Random Mossy Fiber Index Array: Complete	
	std::vector<int> rMFInd;
	rMFInd.assign(cp->numMF,0);
	
	for(int i = 0; i < cp->numMF; i++) rMFInd[i] = i;	
	
	std::random_shuffle(rMFInd.begin(), rMFInd.end());

	//Make Random Span Array: Complete
	std::vector<int> rMFSpanInd;
	rMFSpanInd.assign(cp->numpMFtoGL,0);
	
	for (int ind = 0; ind < cp->numpMFtoGL; ind++) rMFSpanInd[ind] = ind;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	int glX = cp->glX;
	int glY = cp->glY;

	for (int attempts = 0; attempts < 4; attempts++)
	{
		std::random_shuffle(rMFInd.begin(), rMFInd.end());	

		for (int i = 0; i < cp->numMF; i++)
		{	
			//Select MF Coordinates from random index array: Complete	
			srcPosX = rMFInd[i] % cp->mfX;
			srcPosY = rMFInd[i] / cp->mfX;		
			
			std::random_shuffle(rMFSpanInd.begin(), rMFSpanInd.end());	

			for (int j = 0; j < cp->numpMFtoGL; j++)
			{	
				preDestPosX = xCoorsMFGL[rMFSpanInd[j]]; 
				preDestPosY = yCoorsMFGL[rMFSpanInd[j]];	

				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;	
				
				tempDestPosX = (tempDestPosX%glX + glX) % glX;
				tempDestPosY = (tempDestPosY%glY + glY) % glY;
				
				destInd = tempDestPosY * glX + tempDestPosX;
				
				if (numpMFfromMFtoGL[rMFInd[i]] == (initialMFOutput + attempts)) break;
					
				if (!haspGLfromMFtoGL[destInd]) 
				{	
					pMFfromMFtoGL[rMFInd[i]][numpMFfromMFtoGL[rMFInd[i]]] = destInd;
					numpMFfromMFtoGL[rMFInd[i]]++;
					
					pGLfromMFtoGL[destInd] = rMFInd[i];
					haspGLfromMFtoGL[destInd] = true;	
				}
			}
		}
	}	

	int count = 0;
	
	for (int i = 0; i < cp->numMF; i++) count += numpMFfromMFtoGL[i];
	
	std::cout << "Total number of Mossy Fiber to Glomeruli connections:	" << count << std::endl;
	std::cout << "Correct number: " << cp->numGL << std::endl;
}

void InNetConnectivityState::translateUBCGL()
{

	int numUBCfromUBCtoGL = 10;
	//UBC to GR 
	int grIndex;
	for (int i = 0; i < cp->numUBC; i++)
	{
		for (int j = 0; j < numUBCfromUBCtoGL; j++)
		{
			glIndex = pUBCfromUBCtoGL[i][j]; 
			
			for (int k = 0; k < numpGLfromGLtoGR[glIndex]; k++)
			{
				grIndex = pGLfromGLtoGR[glIndex][k];

				pUBCfromUBCtoGR[i][numpUBCfromUBCtoGR[i]] = grIndex; 
				numpUBCfromUBCtoGR[i]++;			

				pGRfromUBCtoGR[grIndex][numpGRfromUBCtoGR[grIndex]] = i;
				numpGRfromUBCtoGR[grIndex]++;
			}
		}
	}

	std::ofstream fileUBCGRconIn;
	fileUBCGRconIn.open("UBCGRInputcon.txt");
	
	for (int i = 0; i < cp->numGR; i++)
	{
		for (int j = 0; j < cp->maxnumpGRfromGLtoGR; j++)
		{
			fileUBCGRconIn << pGRfromUBCtoGR[i][j] << " ";
		}

		fileUBCGRconIn << std::endl;
	}

	int grUBCInputCounter = 0;
	
	for (int i = 0; i < cp->numGR; i++) grUBCInputCounter += numpGRfromUBCtoGR[i];
	std::cout << "Total UBC inputs: " << grUBCInputCounter << std::endl;

	//UBC to GO
	int goIndex;
	
	for (int i = 0; i < cp->numUBC; i++)
	{
		for (int j = 0; j < numUBCfromUBCtoGL; j++)
		{
			glIndex = pUBCfromUBCtoGL[i][j]; 
			
			for (int k = 0; k < numpGLfromGLtoGO[glIndex]; k++)
			{
				goIndex = pGLfromGLtoGO[glIndex][k];

				pUBCfromUBCtoGO[i][numpUBCfromUBCtoGO[i]] = goIndex; 
				numpUBCfromUBCtoGO[i]++;			

				pGOfromUBCtoGO[goIndex][numpGOfromUBCtoGO[goIndex]] = i;
				numpGOfromUBCtoGO[goIndex]++;
			}
		}
	}
	
	std::ofstream fileUBCGOconIn;
	fileUBCGOconIn.open("UBCGOInputcon.txt");
	for (int i = 0; i < cp->numGO; i++)
	{
		for (int j = 0; j < 16; j++) fileUBCGOconIn << pGOfromUBCtoGO[i][j] << " ";
		fileUBCGOconIn << std::endl;
	}

	//UBC to UBC
	int ubcIndex;

	for (int i = 0; i < cp->numUBC; i++)
	{
		for (int j = 0; j < numUBCfromUBCtoGL; j++)
		{
			glIndex = pUBCfromUBCtoGL[i][j]; 
			
			if (numpGLfromGLtoUBC[glIndex] == 1)
			{
				ubcIndex = pGLfromGLtoUBC[glIndex];

				pUBCfromUBCOutUBC[i][numpUBCfromUBCOutUBC[i]] = ubcIndex; 
				numpUBCfromUBCOutUBC[i]++;			

				pUBCfromUBCInUBC[ubcIndex][numpUBCfromUBCInUBC[ubcIndex]] = i;
				numpUBCfromUBCInUBC[ubcIndex]++;
			}
		}
	}
}

void InNetConnectivityState::translateMFGL()
{

	// Mossy fiber to Granule
	
	for (int i = 0; i<cp->numGR; i++)
	{
		for (int j = 0; j < numpGRfromGLtoGR[i]; j++)
		{
			glIndex = pGRfromGLtoGR[i][j];
			if (haspGLfromMFtoGL[glIndex])
			{
				mfIndex = pGLfromMFtoGL[glIndex];

				pMFfromMFtoGR[mfIndex][numpMFfromMFtoGR[mfIndex]] = i; 
				numpMFfromMFtoGR[mfIndex]++;			

				pGRfromMFtoGR[i][numpGRfromMFtoGR[i]] = mfIndex;
				numpGRfromMFtoGR[i]++;
			}
		}
	}	

	int grMFInputCounter = 0;
	
	std::cout << cp->numGR << std::endl;

	for (int i = 0; i < 100; i++)
	{
		std::cout << numpGRfromMFtoGR[i] << " ";
	}

	std::cout << std::endl;

	for (int i = 0; i < cp->numGR; i++)
	{
		grMFInputCounter += numpGRfromMFtoGR[i];
	}

	std::cout << "Total MF inputs: " << grMFInputCounter << std::endl;

	// Mossy fiber to Golgi	
	
	for (int i = 0; i < cp->numGO; i++)
	{
		for (int j = 0; j < numpGOfromGLtoGO[i]; j++)
		{
			glIndex = pGOfromGLtoGO[i][j];
			
			if (haspGLfromMFtoGL[glIndex])
			{
				mfIndex = pGLfromMFtoGL[glIndex];

				pMFfromMFtoGO[mfIndex][numpMFfromMFtoGO[mfIndex]] = i; 
				numpMFfromMFtoGO[mfIndex]++;			

				pGOfromMFtoGO[i][numpGOfromMFtoGO[i]] = mfIndex;
				numpGOfromMFtoGO[i]++;
			}
		}
	}
}

void InNetConnectivityState::translateGOGL(CRandomSFMT *randGen)
{
	for (int i = 0; i < cp->numGR; i++)
	{
		for (int j = 0; j < cp->maxnumpGRfromGOtoGR; j++)
		{
			for (int k = 0; k < cp->maxnumpGLfromGOtoGL; k++)
			{	
				if (numpGRfromGOtoGR[i] < cp->maxnumpGRfromGOtoGR)
				{	
					glIndex = pGRfromGLtoGR[i][j];
					goIndex = pGLfromGOtoGL[glIndex][k];
					
					pGOfromGOtoGR[goIndex][numpGOfromGOtoGR[goIndex]] = i; 
					numpGOfromGOtoGR[goIndex]++;			

					pGRfromGOtoGR[i][numpGRfromGOtoGR[i]] = goIndex;
					numpGRfromGOtoGR[i]++;
				}
			}
		}
	}

	for (int i = 0; i < cp->numGR; i++) totalGOGR += numpGRfromGOtoGR[i];
	
	std::cout << "total GO->GR: " << totalGOGR << std::endl;
	std::cout << "GO->GR Per GR: " << (float)totalGOGR / (float)cp->numGR << std::endl;
}

// NOTE: considerable bottleneck here
void InNetConnectivityState::connectGRGO(CRandomSFMT *randGen, int goRecipParam)
{

	int pfConv[8] = {3750, 3000, 2250, 1500, 750, 375, 188, 93}; 

	int spanPFtoGOX = cp->grX;
	int spanPFtoGOY = 150;
	int numpPFtoGO = (spanPFtoGOX + 1) * (spanPFtoGOY + 1);
	int maxPFtoGOInput = pfConv[goRecipParam];

	//PARALLEL FIBER TO GOLGI 
	for (int i = 0; i < spanPFtoGOX + 1;i++)
	{
		int ind = spanPFtoGOX - i;
		spanArrayPFtoGOX[i] = (spanPFtoGOX / 2) - ind;
	}

	for (int i = 0; i < spanPFtoGOY + 1;i++)
	{
		int ind = spanPFtoGOY - i;
		spanArrayPFtoGOY[i] = (spanPFtoGOY / 2) - ind;
	}
		
	for (int i = 0; i < numpPFtoGO; i++)
	{
		xCoorsPFGO[i] = spanArrayPFtoGOX[i % (spanPFtoGOX + 1)];
		yCoorsPFGO[i] = spanArrayPFtoGOY[i / (spanPFtoGOX + 1)];	
	}

	// Grid Scale: Complete
	float gridXScaleSrctoDest = (float)cp->goX / (float)cp->grX; 
	float gridYScaleSrctoDest = (float)cp->goY / (float)cp->grY; 

	//Make Random Span Array: Complete
	std::vector<int> rPFSpanInd;
	rPFSpanInd.assign(numpPFtoGO,0);

	for (int ind = 0; ind < numpPFtoGO; ind++) rPFSpanInd[ind] = ind;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	int grX = cp->grX;
	int grY = cp->grY;

	for (int atmps = 0; atmps < 5; atmps++)
	{
		for (int i = 0; i < cp->numGO; i++)
		{	
			srcPosX = i % cp->goX;
			srcPosY = i / cp->goX;
			
			std::random_shuffle(rPFSpanInd.begin(), rPFSpanInd.end());		
			
			for (int j = 0; j < maxPFtoGOInput; j++)
			{	
				preDestPosX = xCoorsPFGO[rPFSpanInd[j]]; 
				preDestPosY = yCoorsPFGO[rPFSpanInd[j]];	
				
				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;
				
				tempDestPosX = (tempDestPosX%grX + grX) % grX;
				tempDestPosY = (tempDestPosY%grY + grY) % grY;
						
				destInd = tempDestPosY * cp->grX + tempDestPosX;

				if (numpGOfromGRtoGO[i] < maxPFtoGOInput)
				{	
					pGOfromGRtoGO[i][numpGOfromGRtoGO[i]] = destInd;
					numpGOfromGRtoGO[i]++;

					pGRfromGRtoGO[destInd][numpGRfromGRtoGO[destInd]] = i;
					numpGRfromGRtoGO[destInd]++;	
				}
			}
		}
	}

	//ASCENDING AXON TO GOLGI 
	int aaConv[8] = {1250, 1000, 750, 500, 250, 125, 62, 32}; 

	int spanAAtoGOX = 150;
	int spanAAtoGOY = 150;
	int numpAAtoGO = (spanAAtoGOX+1)*(spanAAtoGOY+1);
	int maxAAtoGOInput = aaConv[goRecipParam];

	//Make Span Array: Complete	
	for (int i = 0; i < spanAAtoGOX + 1;i++)
	{
		int ind = spanAAtoGOX - i;
		spanArrayAAtoGOX[i] = (spanAAtoGOX / 2) - ind;
	}

	for (int i = 0; i < spanAAtoGOY + 1;i++)
	{
		int ind = spanAAtoGOY - i;
		spanArrayAAtoGOY[i] = (spanAAtoGOY / 2) - ind;
	}
		
	for (int i = 0; i < numpAAtoGO; i++)
	{
		xCoorsAAGO[i] = spanArrayAAtoGOX[i%(spanAAtoGOX+1)];
		yCoorsAAGO[i] = spanArrayAAtoGOY[i/(spanAAtoGOX+1)];	
	}
	
	//Make Random Span Array: Complete
	std::vector<int> rAASpanInd;
	rAASpanInd.assign(numpAAtoGO,0);

	for (int ind = 0; ind < numpAAtoGO; ind++) rAASpanInd[ind] = ind;

	for (int atmps = 0; atmps < 15; atmps++)
	{
		for (int i = 0; i < cp->numGO; i++)
		{	
			srcPosX = i % cp->goX;
			srcPosY = i / cp->goX;

			std::random_shuffle(rAASpanInd.begin(), rAASpanInd.end());		
			
			for (int j = 0; j < maxAAtoGOInput; j++)
			{	
				preDestPosX = xCoorsAAGO[rAASpanInd[j]]; 
				preDestPosY = yCoorsAAGO[rAASpanInd[j]];	
				
				tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
				tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;

				tempDestPosX = (tempDestPosX % grX + grX) % grX;
				tempDestPosY = (tempDestPosY % grY + grY) % grY;
						
				destInd = tempDestPosY * cp->grX + tempDestPosX;
				
				if (numpGOfromGRtoGO[i] < maxAAtoGOInput + maxPFtoGOInput)
				{	
					pGOfromGRtoGO[i][numpGOfromGRtoGO[i]] = destInd;	
					numpGOfromGRtoGO[i]++;

					pGRfromGRtoGO[destInd][numpGRfromGRtoGO[destInd]] = i;
					numpGRfromGRtoGO[destInd]++;	
				}
			}
		}
	}	

	int sumGOGR_GO = 0;
	
	for (int i = 0; i < cp->numGO; i++)
	{
		sumGOGR_GO += numpGOfromGRtoGO[i];
	}

	std::cout << "GRtoGO_GO: " << sumGOGR_GO << std::endl;

	int sumGOGR_GR = 0;

	for (int i = 0; i < cp->numGR; i++)
	{
		sumGOGR_GR += numpGRfromGRtoGO[i];
	}

	std::cout << "GRtoGO_GR: " << sumGOGR_GR << std::endl;
}

void InNetConnectivityState::connectGOGODecayP(CRandomSFMT *randGen, int goRecipParam, int simNum)
{
	int numberConnections[26] = { 1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50 };

	int span = 4;
	int numCon = 12;
	int numP = (span + 1) * (span + 1); 

	//Allocate Dat Shit Boi
	numpGOGABAInGOGO    = new int[cp->numGO];
	pGOGABAInGOGO	    = allocate2DArray<int>(cp->numGO, numCon);
	numpGOGABAOutGOGO   = new int[cp->numGO];
	pGOGABAOutGOGO	    = allocate2DArray<int>(cp->numGO, numCon);	
	conGOGOBoolOut	    = allocate2DArray<bool>(cp->numGO, cp->numGO);
	spanArrayGOtoGOsynX = new int[span + 1];
	spanArrayGOtoGOsynY = new int[span + 1];
	xCoorsGOGOsyn 		= new int[numP];
	yCoorsGOGOsyn 		= new int[numP];
	Pcon 				= new float[numP];
	
	//Initialize MaaFuka
	std::fill(numpGOGABAInGOGO, numpGOGABAInGOGO + cp->numGO, 0);
	std::fill(pGOGABAInGOGO[0], pGOGABAInGOGO[0] +
			cp->numGO * numCon, UINT_MAX);
	std::fill(numpGOGABAOutGOGO, numpGOGABAOutGOGO + cp->numGO, 0);
	std::fill(pGOGABAOutGOGO[0], pGOGABAOutGOGO[0] + cp->numGO * numCon, UINT_MAX);
	std::fill(conGOGOBoolOut[0], conGOGOBoolOut[0] + cp->numGO * cp->numGO, false);
	std::fill(spanArrayGOtoGOsynX, spanArrayGOtoGOsynX + span + 1, 0);
	std::fill(spanArrayGOtoGOsynY, spanArrayGOtoGOsynY + span + 1, 0);
	std::fill(xCoorsGOGOsyn, xCoorsGOGOsyn + numP, 0);
	std::fill(yCoorsGOGOsyn, yCoorsGOGOsyn + numP, 0);
	std::fill(Pcon, Pcon + numP, 0);

	float A = 0.35;
	float sig = 1000;

	float recipParamArray_P[11] = { 1.0, 0.925, 0.78, 0.62, 0.47, 0.32, 0.19, 0.07, 0.0, 0.0, 0.0 };
	float recipParamArray_LowerP[11] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.35, 0.0 }; 
	bool recipParamArray_ReduceBase[11] = { false, false, false, false, false, false, false, false, true, true, false };
	bool recipParamArray_noRecip[11] = { false, false, false, false, false, false, false, false, false, false, true };
	 
	float pRecipGOGO = 1.0;
	bool noRecip = false;
	float pRecipLowerBase = 0.0;
	bool reduceBaseRecip = false;

	float PconX;
	float PconY;

	for (int i = 0; i < span + 1; i++)
   	{
		spanArrayGOtoGOsynX[i] = (span / 2) - (span - i);
	}
	
	for (int i = 0; i < span + 1; i++) 
	{
		spanArrayGOtoGOsynY[i] = (span / 2) - (span - i);
	}
		
	for (int i = 0; i < numP; i++) {
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (span + 1)];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (span + 1)];		
	}

	for (int i = 0; i < numP; i++) {
		PconX   = (xCoorsGOGOsyn[i] * xCoorsGOGOsyn[i]) / (2 * (sig * sig));
		PconY   = (yCoorsGOGOsyn[i] * yCoorsGOGOsyn[i]) / (2 * (sig * sig));
		Pcon[i] = A * exp(-(PconX + PconY));
	
	}
	
	// Remove self connection 
	for (int i = 0; i < numP; i++) 
	{
		if ((xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0)) Pcon[i] = 0;
	}
	
	std::vector<int> rGOGOSpanInd;
	rGOGOSpanInd.assign(numP, 0);
	
	for (int ind = 0; ind < numP; ind++) rGOGOSpanInd[ind] = ind;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	for (int atmpt = 0; atmpt < 200; atmpt++) 
	{
		for (int i = 0; i < cp->numGO; i++) 
		{	
			srcPosX = i % cp->goX;
			srcPosY = i / cp->goX;	
			
			std::random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
			
			for (int j = 0; j < numP; j++)
		   	{	
				preDestPosX = xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
				preDestPosY = yCoorsGOGOsyn[rGOGOSpanInd[j]];	

				tempDestPosX = srcPosX + preDestPosX;
				tempDestPosY = srcPosY + preDestPosY;

				tempDestPosX = (tempDestPosX % cp->goX + cp->goX) % cp->goX;
				tempDestPosY = (tempDestPosY % cp->goY + cp->goY) % cp->goY;
						
				destInd = tempDestPosY * cp->goX + tempDestPosX;
			
				// Normal One	
				if (!noRecip && !reduceBaseRecip && randGen->Random()>= 1 - Pcon[rGOGOSpanInd[j]]
						&& !conGOGOBoolOut[i][destInd] && numpGOGABAOutGOGO[i] < numCon
						&& numpGOGABAInGOGO[destInd] < numCon) 
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destInd;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = i;
					numpGOGABAInGOGO[destInd]++;
					
					conGOGOBoolOut[i][destInd] = true;
							
					if (randGen->Random() >= 1 - pRecipGOGO && !conGOGOBoolOut[destInd][i]
							&& numpGOGABAOutGOGO[destInd] < numCon && numpGOGABAInGOGO[i] < numCon) 
					{
						pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = i;
						numpGOGABAOutGOGO[destInd]++;
						
						pGOGABAInGOGO[i][numpGOGABAInGOGO[i]] = destInd;
						numpGOGABAInGOGO[i]++;
						
						conGOGOBoolOut[destInd][i] = true;
					}
				}
			
				if (!noRecip && reduceBaseRecip && randGen->Random()>=1 - Pcon[rGOGOSpanInd[j]]
						&& !conGOGOBoolOut[i][destInd] && (!conGOGOBoolOut[destInd][i] ||
							randGen->Random() >= 1 - pRecipLowerBase) && numpGOGABAOutGOGO[i] < numCon
						&& numpGOGABAInGOGO[destInd] < numCon) 
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destInd;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = i;
					numpGOGABAInGOGO[destInd]++;
					
					conGOGOBoolOut[i][destInd] = true;	
				}

				if (noRecip && !reduceBaseRecip && randGen->Random() >= 1 - Pcon[rGOGOSpanInd[j]]
						&& (!conGOGOBoolOut[i][destInd]) && !conGOGOBoolOut[destInd][i] && 
						numpGOGABAOutGOGO[i] < numCon && numpGOGABAInGOGO[destInd] < numCon)
				{	
					pGOGABAOutGOGO[i][ numpGOGABAOutGOGO[i] ] = destInd;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destInd][ numpGOGABAInGOGO[destInd] ] = i;
					numpGOGABAInGOGO[destInd]++;
					
					conGOGOBoolOut[i][destInd] = true;
				}
			}
		}
	}

	float totalGOGOcons = 0;

	for (int i = 0; i < cp->numGO; i++)
	{
		totalGOGOcons += numpGOGABAInGOGO[i];
	}

	std::cout << "Total GOGO connections: " << totalGOGOcons << std::endl;
	std::cout << "Average GOGO connections:	" << totalGOGOcons / float(cp->numGO) << std::endl;
	std::cout << cp->numGO << std::endl;

	int recipCounter = 0;

	for (int i = 0; i < cp->numGO; i++)
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

	float fracRecip = recipCounter / totalGOGOcons;
	std::cout << "FracRecip: " << fracRecip << std::endl;

	rGOGOSpanInd.clear();
}

void InNetConnectivityState::connectGOGODecay(CRandomSFMT *randGen)
{
	float A 		 = 0.01;
	float pRecipGOGO = 1;
	float PconX;
	float PconY;

	std::cout << cp->spanGOGOsynX << std::endl;
	std::cout << cp->spanGOGOsynY << std::endl;
	std::cout << cp->sigmaGOGOsynML << std::endl;
	std::cout << cp->sigmaGOGOsynS << std::endl;

	std::cout << cp->pRecipGOGOsyn << std::endl;
	std::cout << cp->maxGOGOsyn << std::endl;

	for (int i = 0; i < cp->spanGOGOsynX + 1; i++)
	{
		int ind = cp->spanGOGOsynX - i;
		spanArrayGOtoGOsynX[i] = (cp->spanGOGOsynX / 2) - ind;
	}

	for (int i = 0; i < cp->spanGOGOsynY + 1; i++)
	{
		int ind = cp->spanGOGOsynY - i;
		spanArrayGOtoGOsynY[i] = (cp->spanGOGOsynY / 2) - ind;
	}
		
	for (int i = 0; i < cp->numpGOGOsyn; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (cp->spanGOGOsynX + 1)];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (cp->spanGOGOsynX+1)];		
	}

	for (int i = 0; i < cp->numpGOGOsyn; i++)
	{
		PconX = (xCoorsGOGOsyn[i] * xCoorsGOGOsyn[i]) / (2 * cp->sigmaGOGOsynML * cp->sigmaGOGOsynML);
		PconY = (yCoorsGOGOsyn[i] * yCoorsGOGOsyn[i]) / (2 * cp->sigmaGOGOsynS * cp->sigmaGOGOsynS);
		Pcon[i] = A * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < cp->numpGOGOsyn; i++)
	{
		if ((xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0)) Pcon[i] = 0;
	}
	
	std::vector<int> rGOGOSpanInd;
	rGOGOSpanInd.assign(cp->numpGOGOsyn,0);
	for (int ind = 0; ind < cp->numpGOGOsyn; ind++) rGOGOSpanInd[ind] = ind;
	
	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	for (int atmpt = 0; atmpt <200; atmpt++)
	{
		for (int i = 0; i < cp->numGO; i++)
		{	
			srcPosX = i % cp->goX;
			srcPosY = i / cp->goX;	
			
			std::random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
			
			for (int j = 0; j < cp->numpGOGOsyn; j++)
			{	
				
				preDestPosX = xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
				preDestPosY = yCoorsGOGOsyn[rGOGOSpanInd[j]];	

				tempDestPosX = srcPosX + preDestPosX;
				tempDestPosY = srcPosY + preDestPosY;

				tempDestPosX = (tempDestPosX % cp->goX + cp->goX) % cp->goX;
				tempDestPosY = (tempDestPosY % cp->goY + cp->goY) % cp->goY;
						
				destInd = tempDestPosY * cp->goX + tempDestPosX;

				if (randGen->Random()>=1 - Pcon[rGOGOSpanInd[j]] && !conGOGOBoolOut[i][destInd] &&
					   numpGOGABAOutGOGO[i] < cp->maxGOGOsyn && numpGOGABAInGOGO[destInd] < cp->maxGOGOsyn)
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destInd;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = i;
					numpGOGABAInGOGO[destInd]++;
					
					conGOGOBoolOut[i][destInd] = true;
					
					if (randGen->Random() >= 1 - pRecipGOGO && !conGOGOBoolOut[destInd][i] &&
							numpGOGABAOutGOGO[destInd] < cp->maxGOGOsyn && numpGOGABAInGOGO[i] < cp->maxGOGOsyn)
					{
						pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = i;
						numpGOGABAOutGOGO[destInd]++;

						pGOGABAInGOGO[i][numpGOGABAInGOGO[i]] = destInd;
						numpGOGABAInGOGO[i]++;
						
						conGOGOBoolOut[destInd][i] = true;
					}
				}
			}
		}
	}

	std::ofstream fileGOGOconIn;
	fileGOGOconIn.open("GOGOInputcon.txt");
	
	for (int i = 0; i < cp->numGO; i++)
	{
		for (int j = 0; j < cp->maxGOGOsyn; j++)
		{
			fileGOGOconIn << pGOGABAInGOGO[i][j] << " ";
		}

		fileGOGOconIn << std::endl;
	}
	
	std::ofstream fileGOGOconOut;
	fileGOGOconOut.open("GOGOOutputcon.txt");
	
	for (int i = 0; i < cp->numGO; i++)
	{
		for (int j = 0; j < cp->maxGOGOsyn; j++)
		{
			fileGOGOconOut << pGOGABAOutGOGO[i][j] << " ";
		}

		fileGOGOconOut << std::endl;
	}

	float totalGOGOcons = 0;
	
	for (int i = 0; i < cp->numGO; i++)
	{
		totalGOGOcons += numpGOGABAInGOGO[i];
	}

	std::cout << "Total GOGO connections: " << totalGOGOcons << std::endl;
	std::cout << "Average GOGO connections:	" << totalGOGOcons / float(cp->numGO) << std::endl;
	std::cout << cp->numGO << std::endl;
	int recipCounter = 0;
	
	for (int i = 0; i < cp->numGO; i++)
	{
		for (int j = 0; j < numpGOGABAInGOGO[i]; j++)
		{
			for (int k = 0; k < numpGOGABAOutGOGO[i]; k++)
			{
				if (pGOGABAInGOGO[i][j] == pGOGABAOutGOGO[i][k] && pGOGABAInGOGO[i][j] != UINT_MAX
						&& pGOGABAOutGOGO[i][k] != UINT_MAX) recipCounter++; 
			}
		}
	}

	float fracRecip = recipCounter / totalGOGOcons;
	std::cout << "FracRecip: " << fracRecip << std::endl;
	rGOGOSpanInd.clear();
}

void InNetConnectivityState::connectGOGOBias(CRandomSFMT *randGen)
{
	int spanGOtoGOsynX = 12;
	int spanGOtoGOsynY = 12;
	int numpGOtoGOsyn = (spanGOtoGOsynX + 1) *(spanGOtoGOsynY + 1); 

	//Make Span Array: Complete	
	for (int i = 0; i < spanGOtoGOsynX +1;i++)
	{
		spanArrayGOtoGOsynX[i] = (spanGOtoGOsynX / 2) - (spanGOtoGOsynX - i);
	}

	for (int i =0; i <spanGOtoGOsynY +1; i++)
	{
		spanArrayGOtoGOsynY[i] = (spanGOtoGOsynY / 2) - (spanGOtoGOsynY - i);
	}
		
	for (int i = 0; i < numpGOtoGOsyn; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (spanGOtoGOsynX + 1)];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (spanGOtoGOsynX + 1)];		
	}

	//Make Random Mossy Fiber Index Array: Complete	
	std::vector<int> rGOInd;
	rGOInd.assign(cp->numGO, 0);
	
	for (int i = 0; i < cp->numGO; i++) rGOInd[i] = i;	
	
	std::random_shuffle(rGOInd.begin(), rGOInd.end());

	//Make Random Span Array
	std::vector<int> rGOGOSpanInd;
	rGOGOSpanInd.assign(numpGOtoGOsyn,0);
	for (int ind = 0; ind < numpGOtoGOsyn; ind++) rGOGOSpanInd[ind] = ind;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	int numOutputs = cp->numConGOGO;
	int conType = 0;

	for (int i = 0; i < cp->numGO; i++)
	{	
		srcPosX = rGOInd[i] % cp->goX;
		srcPosY = rGOInd[i] / cp->goX;	
		
		std::random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
			
		for (int j = 0; j < numpGOtoGOsyn; j++)
		{	
			preDestPosX = xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
			preDestPosY = yCoorsGOGOsyn[rGOGOSpanInd[j]];	

			tempDestPosX = srcPosX + preDestPosX;
			tempDestPosY = srcPosY + preDestPosY;

			tempDestPosX = (tempDestPosX % cp->goX + cp->goX) % cp->goX;
			tempDestPosY = (tempDestPosY % cp->goY + cp->goY) % cp->goY;
			
			destInd = tempDestPosY * cp->goX + tempDestPosX;
				
			if (randGen->Random() >= 0.9695 && !conGOGOBoolOut[rGOInd[i]][destInd])
			{
				pGOGABAOutGOGO[rGOInd[i]][numpGOGABAOutGOGO[rGOInd[i]]] = destInd;
				numpGOGABAOutGOGO[rGOInd[i]]++;
					
				pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = rGOInd[i];
				numpGOGABAInGOGO[destInd]++;	
					
				conGOGOBoolOut[rGOInd[i]][destInd] = true;
			
				// conditional statement against making double output

				if (randGen->Random() > 0 && !conGOGOBoolOut[destInd][rGOInd[i]])
				{
					pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = rGOInd[i];
					numpGOGABAOutGOGO[destInd]++;

					pGOGABAInGOGO[rGOInd[i]][numpGOGABAInGOGO[rGOInd[i]]] = destInd;
					numpGOGABAInGOGO[rGOInd[i]]++;
					
					conGOGOBoolOut[destInd][rGOInd[i]] = true;
				}
			}	
		}
	}
	
	if (conType == 0)
	{
		for (int k = 0; k < 20; k++)
		{
			std::cout << "numpGOGABAInGOGO[" << k << "]: " << numpGOGABAInGOGO[k] << std::endl;
			for (int j = 0; j < numpGOGABAInGOGO[k]; j++) std::cout << pGOGABAInGOGO[k][j] << " ";
			std::cout << std::endl;
		}
	}
	else if (conType == 1)
	{	
		int pNonRecip 	 = 0;
		int missedOutCon = 0;
		int missedInCon  = 0;

		for (int i = 0; i < cp->numGO; i++)
		{

			if (numpGOGABAInGOGO[i]  != numpGOGABAOutGOGO[i]) pNonRecip++; 
			if (numpGOGABAInGOGO[i]  != numOutputs) missedInCon++; 
			if (numpGOGABAOutGOGO[i] != numOutputs) missedOutCon++; 
		}

		std::cout << "Potential non-reciprocal connection: " << pNonRecip << std::endl;
		std::cout << "Missing Input: " << missedInCon << std::endl;
		std::cout << "Missing Output: " << missedOutCon << std::endl;
	}
}

void InNetConnectivityState::connectGOGO(CRandomSFMT *randGen)
{
	int spanGOtoGOsynX = 12;
	int spanGOtoGOsynY = 12;
	int numpGOtoGOsyn = (spanGOtoGOsynX + 1) * (spanGOtoGOsynY + 1); 

	//Make Span Array: Complete	
	for (int i = 0; i < spanGOtoGOsynX + 1;i++)
	{
		spanArrayGOtoGOsynX[i] = (spanGOtoGOsynX / 2) - (spanGOtoGOsynX - i);
	}

	for (int i = 0; i < spanGOtoGOsynY + 1; i++)
	{
		int ind = spanGOtoGOsynY - i;
		spanArrayGOtoGOsynY[i] = (spanGOtoGOsynY / 2) - (spanGOtoGOsynY - i);
	}
		
	for (int i = 0; i < numpGOtoGOsyn; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (spanGOtoGOsynX + 1)];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (spanGOtoGOsynX + 1)];		
	}

	//Make Random Mossy Fiber Index Array: Complete	
	std::vector<int> rGOInd;
	rGOInd.assign(cp->numGO,0);
	
	for (int i = 0; i < cp->numGO; i++) rGOInd[i] = i;	
	
	std::random_shuffle(rGOInd.begin(), rGOInd.end());

	//Make Random Span Array
	std::vector<int> rGOGOSpanInd;
	rGOGOSpanInd.assign(numpGOtoGOsyn,0);
	for (int ind = 0; ind < numpGOtoGOsyn; ind++) rGOGOSpanInd[ind] = ind;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	int numOutputs = cp->numConGOGO;
	int conType = 2;

	for (int attempt = 1; attempt < numOutputs + 1; attempt++)
	{
		for (int i = 0; i < cp->numGO; i++)
		{	
			srcPosX = rGOInd[i] % cp->goX;
			srcPosY = rGOInd[i] / cp->goX;	
			
			std::random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
			
			for (int j = 0; j < numpGOtoGOsyn; j++)
			{	
				preDestPosX = xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
				preDestPosY = yCoorsGOGOsyn[rGOGOSpanInd[j]];	

				tempDestPosX = srcPosX + preDestPosX;
				tempDestPosY = srcPosY + preDestPosY;

				tempDestPosX = (tempDestPosX % cp->goX + cp->goX) % cp->goX;
				tempDestPosY = (tempDestPosY % cp->goY + cp->goY) % cp->goY;

				destInd = tempDestPosY * cp->goX + tempDestPosX;
					
				if (conType == 0)
				{ 	
					// Normal random connectivity		
					if (numpGOGABAOutGOGO[rGOInd[i]] == numOutputs) break;

					// conditional statment blocking the ability to make two outputs to the same cell
					pGOGABAOutGOGO[rGOInd[i]][numpGOGABAOutGOGO[rGOInd[i]]] = destInd;
					numpGOGABAOutGOGO[rGOInd[i]]++;
					
					pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = rGOInd[i];
					numpGOGABAInGOGO[destInd]++;
				}
				else if (conType == 1){	
					// 100% reciprocal	
					if (!conGOGOBoolOut[rGOInd[i]][destInd]&&
						(numpGOGABAOutGOGO[rGOInd[i]] < attempt ) &&
						(numpGOGABAInGOGO[rGOInd[i]] < attempt) &&
						(numpGOGABAOutGOGO[destInd] < attempt) &&
						(numpGOGABAInGOGO[destInd] < attempt) &&
						(destInd != rGOInd[i])) 
					{	
						pGOGABAOutGOGO[rGOInd[i]][numpGOGABAOutGOGO[rGOInd[i]]] = destInd;
						numpGOGABAOutGOGO[rGOInd[i]]++;
						pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = rGOInd[i];
						numpGOGABAOutGOGO[destInd]++;
					
						pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = rGOInd[i];
						numpGOGABAInGOGO[destInd]++;
						pGOGABAInGOGO[rGOInd[i]][numpGOGABAInGOGO[rGOInd[i]]] = destInd;
						numpGOGABAInGOGO[rGOInd[i]]++;
				
						conGOGOBoolOut[rGOInd[i]][destInd] = true;
					}
				}
			
				else if (conType == 2)
				{	
					// variable %reciprocal	
					if (!conGOGOBoolOut[rGOInd[i]][destInd] &&
						(numpGOGABAOutGOGO[rGOInd[i]] < attempt) &&
						(numpGOGABAInGOGO[rGOInd[i]] < attempt) &&
						(numpGOGABAOutGOGO[destInd] < attempt) &&
						(numpGOGABAInGOGO[destInd] < attempt) &&
						destInd != rGOInd[i]) 
					{	
						pGOGABAOutGOGO[rGOInd[i]][numpGOGABAOutGOGO[rGOInd[i]]] = destInd;
						numpGOGABAOutGOGO[rGOInd[i]]++;
					
						pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = rGOInd[i];
						numpGOGABAInGOGO[destInd]++;
						
						if (randGen->Random() >= 0.95)
						{
							pGOGABAInGOGO[rGOInd[i]][numpGOGABAInGOGO[rGOInd[i]]] = destInd;
							numpGOGABAInGOGO[rGOInd[i]]++;
							pGOGABAOutGOGO[destInd][numpGOGABAOutGOGO[destInd]] = rGOInd[i];
							numpGOGABAOutGOGO[destInd]++;
				
							conGOGOBoolOut[rGOInd[i]][destInd] = true;
						}
					}
				}
			}
		}
	}	
	
	std::ofstream fileGOGOconIn;
	fileGOGOconIn.open("GOGOInputcon.txt");
	
	for (int i = 0; i < cp->numGO; i++)
	{
		for (int j = 0; j < 30; j++)
		{
			fileGOGOconIn << pGOGABAInGOGO[i][j] << " ";
		}

		fileGOGOconIn << std::endl;
	}
	
	std::ofstream fileGOGOconOut;
	fileGOGOconOut.open("GOGOOutputcon.txt");
	
	for (int i = 0; i < cp->numGO; i++)
	{
		for (int j = 0; j < 30; j++)
		{
			fileGOGOconOut << pGOGABAOutGOGO[i][j] << " ";
		}
		fileGOGOconOut << std::endl;
	}

	if (conType == 0)
	{
		for (int k = 0; k < 20; k++)
		{
			std::cout << "numpGOGABAInGOGO[" << k << "]: " << numpGOGABAInGOGO[k] << std::endl;
			for (int j = 0; j < numpGOGABAInGOGO[k]; j++)
			{
				std::cout << pGOGABAInGOGO[k][j] << " ";
			}
			std::cout << std::endl;
		}
	}

	else if (conType == 1)
	{	
		int pNonRecip = 0;
		int missedOutCon = 0;
		int missedInCon = 0;

		for (int i = 0; i < cp->numGO; i++)
		{
			if (numpGOGABAInGOGO[i] != numpGOGABAOutGOGO[i]) pNonRecip++;
			if (numpGOGABAInGOGO[i] != numOutputs) missedInCon++;
			if (numpGOGABAOutGOGO[i] != numOutputs) missedOutCon++;
		}

		std::cout << "Potential non-reciprocal connection: " << pNonRecip << std::endl;
		std::cout << "Missing Input: " << missedInCon << std::endl;
		std::cout << "Missing Output: " << missedOutCon << std::endl;
	}
}

void InNetConnectivityState::connectGOGO_GJ(CRandomSFMT *randGen)
{
	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	//GOLGI TO GOLGI GAP JUNCTIONS

	int spanGOtoGOgjX = 8;
	int spanGOtoGOgjY = 8;
	int numpGOtoGOgj  = (spanGOtoGOgjX + 1) * (spanGOtoGOgjY + 1);

	for (int i = 0; i < spanGOtoGOgjX + 1; i++)
	{
		spanArrayGOtoGOgjX[i] = (spanGOtoGOgjX / 2) - (spanGOtoGOgjX - i);
	}

	for (int i = 0; i < spanGOtoGOgjY +1; i++)
	{
		spanArrayGOtoGOgjY[i] = (spanGOtoGOgjY/2) - (spanGOtoGOgjY - i);
	}
		
	for (int i = 0; i < numpGOtoGOgj; i++)
	{
		xCoorsGOGOgj[i] = spanArrayGOtoGOgjX[i % (spanGOtoGOgjX + 1)];
		yCoorsGOGOgj[i] = spanArrayGOtoGOgjY[i / (spanGOtoGOgjX + 1)];		
	}

	// "In Vivo additions"
	for (int i = 0; i < numpGOtoGOgj; i++)
	{
		gjPconX = exp(((abs(xCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );	
		gjPconY = exp(((abs(yCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );
		gjPcon[i] = ((-1745.0 + (1836.0 / (1 + (gjPconX + gjPconY)))) * 0.01);
		
		gjCCX = exp(abs(xCoorsGOGOgj[i] * 100.0) / 190.0);
		gjCCY = exp(abs(yCoorsGOGOgj[i] * 100.0) / 190.0);
		gjCC[i] = (-2.3 + (23.0 / ((gjCCX + gjCCY)/2.0))) * 0.09;
	}

	// Remove self connection 
	for (int i = 0; i < numpGOtoGOgj; i++)
	{
		if ((xCoorsGOGOgj[i] == 0) && (yCoorsGOGOgj[i] == 0))
		{
			gjPcon[i] = 0;
			gjCC[i] = 0;
		}
	}

	float tempCC;

	for (int i = 0; i < cp->numGO; i++)
	{	
		srcPosX = i % cp->goX;
		srcPosY = i / cp->goX;	
		
		for (int j = 0; j < numpGOtoGOgj; j++)
		{	
			preDestPosX = xCoorsGOGOgj[j]; 
			preDestPosY = yCoorsGOGOgj[j];	

			tempCC = gjCC[j];

			tempDestPosX = srcPosX + preDestPosX;
			tempDestPosY = srcPosY + preDestPosY;

			tempDestPosX = (tempDestPosX % cp->goX + cp->goX) % cp->goX;
			tempDestPosY = (tempDestPosY % cp->goY + cp->goY) % cp->goY;
					
			destInd = tempDestPosY * cp->goX + tempDestPosX;

			if ((randGen->Random()>= 1 - gjPcon[j]) && !gjConBool[i][destInd ] && 
					!gjConBool[destInd][i])
			{	
				pGOCoupInGOGO[destInd][numpGOCoupInGOGO[destInd]] = i;
				pGOCoupInGOGOCCoeff[destInd][numpGOCoupInGOGO[destInd]] = tempCC;
				numpGOCoupInGOGO[destInd]++;
					
				pGOCoupInGOGO[i][numpGOCoupInGOGO[i]] = destInd;
				pGOCoupInGOGOCCoeff[i][numpGOCoupInGOGO[i]] = tempCC;
				numpGOCoupInGOGO[i]++;

				gjConBool[i][destInd] = true;
			}
		}
	}
}

void InNetConnectivityState::connectPFtoBC()
{
	int spanPFtoBCX = cp->grX;
	int spanPFtoBCY = cp->grY / cp->numBC;
	int numpPFtoBC = (spanPFtoBCX + 1) * (spanPFtoBCY + 1);

	for (int i = 0; i < spanPFtoBCX + 1; i++)
	{
		spanArrayPFtoBCX[i] = (spanPFtoBCX / 2) - (spanPFtoBCX - i);
	}

	for (int i = 0; i < spanPFtoBCY + 1; i++)
	{
		int ind = spanPFtoBCY - i;
		spanArrayPFtoBCY[i] = (spanPFtoBCY / 2) - (spanPFtoBCY - i);
	}
	
	for (int i = 0; i < numpPFtoBC; i++)
	{
		xCoorsPFBC[i] = spanArrayPFtoBCX[i % (spanPFtoBCX + 1)];
		yCoorsPFBC[i] = spanArrayPFtoBCY[i / (spanPFtoBCX + 1)];
	}

	//Random Span Array
	std::vector<int> rPFBCSpanInd;
	rPFBCSpanInd.assign(numpPFtoBC, 0);
	
	for (int ind = 0; ind < numpPFtoBC; ind++) rPFBCSpanInd[ind] = ind;

	float gridXScaleSrctoDest = 1;
	float gridYScaleSrctoDest = (float)cp->numBC / cp->grY;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	for (int i =0; i < cp->numBC; i++)
	{
		srcPosX = cp->grX / 2;
		srcPosY = i;

		std::random_shuffle(rPFBCSpanInd.begin(), rPFBCSpanInd.end());
		
		for (int j = 0; j < 5000; j++)
		{
			preDestPosX = xCoorsPFBC[rPFBCSpanInd[j]];
			preDestPosY = yCoorsPFBC[rPFBCSpanInd[j]];
			
			tempDestPosX = (int)round(srcPosX / gridXScaleSrctoDest) + preDestPosX;
			tempDestPosY = (int)round(srcPosY / gridYScaleSrctoDest) + preDestPosY;
			
			tempDestPosX = (tempDestPosX%cp->grX + cp->grX) % cp->grX;
			tempDestPosY = (tempDestPosY%cp->grY + cp->grY) % cp->grY;

			destInd = tempDestPosY * cp->grX + tempDestPosX;	

			pBCfromPFtoBC[i][numpBCfromPFtoBC[i]] = destInd;
			numpBCfromPFtoBC[i]++;

			pGRfromPFtoBC[destInd][numpGRfromPFtoBC[destInd]] = i;
			numpGRfromPFtoBC[destInd]++;
		}
	}
}

void InNetConnectivityState::assignPFtoBCDelays(unsigned int msPerStep)
{
	
	for (int i = 0; i < cp->numGR; i++)
	{
		int grPosX;

		//calculate x coordinate of GR position
		grPosX=i%cp->grX;

		for (int j = 0; j < numpGRfromPFtoBC[i]; j++)
		{
			int dfromGRtoBC;
			int bcPosX;

			bcPosX = cp->grX / 2; 
			dfromGRtoBC=abs(bcPosX-grPosX);
		}
	}
}

void InNetConnectivityState::assignGRDelays(unsigned int msPerStep)
{
	for (int i = 0; i < cp->numGR; i++)
	{
		int grPosX;
		int grBCPCSCDist;

		//calculate x coordinate of GR position
		grPosX=i%cp->grX;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		grBCPCSCDist = abs((int)(cp->grX / 2 - grPosX));
		pGRDelayMaskfromGRtoBSP[i] = 0x1 << (int)((grBCPCSCDist / cp->grPFVelInGRXPerTStep +
					cp->grAFDelayInTStep) / msPerStep);

		for (int j = 0; j < numpGRfromGRtoGO[i]; j++)
		{
			int dfromGRtoGO;
			int goPosX;

			goPosX = (pGRfromGRtoGO[i][j] % cp->goX) * (((float)cp->grX) / cp->goX);

			dfromGRtoGO = abs(goPosX - grPosX);

			if (dfromGRtoGO > cp->grX / 2)
			{
				if (goPosX < grPosX) dfromGRtoGO = goPosX + cp->grX - grPosX;
				else dfromGRtoGO = grPosX + cp->grX - goPosX;
			}

			pGRDelayMaskfromGRtoGO[i][j] = 0x1<< (int)((dfromGRtoGO/cp->grPFVelInGRXPerTStep +
							cp->grAFDelayInTStep) / msPerStep);
		}
	}
}

void InNetConnectivityState::connectCommon(int **srcConArr, int *srcNumCon,
		int **destConArr, int *destNumCon, int srcMaxNumCon, int numSrcCells,
		int destMaxNumCon, int destNormNumCon, int srcGridX, int srcGridY,
		int destGridX, int destGridY, int srcSpanOnDestGridX, int srcSpanOnDestGridY,
		int normConAttempts, int maxConAttempts, bool needUnique, CRandomSFMT *randGen)
{
	bool *srcConnected;
	float gridXScaleStoD;
	float gridYScaleStoD;

	gridXScaleStoD = (float)srcGridX / (float)destGridX;
	gridYScaleStoD = (float)srcGridY / (float)destGridY;

	srcConnected = new bool[numSrcCells];

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

void InNetConnectivityState::translateCommon(int **pPreGLConArr, int *numpPreGLCon,
		int **pGLPostGLConArr, int *numpGLPostGLCon, int **pPreConArr, int *numpPreCon,
		int **pPostConArr, int *numpPostCon, int numPre)
{
	
	for (int i = 0; i < numPre; i++)
	{
		numpPreCon[i] = 0;

		for (int j = 0; j < numpPreGLCon[i]; j++)
		{
			int glInd;

			glInd = pPreGLConArr[i][j];

			for (int k = 0; k < numpGLPostGLCon[glInd]; k++)
			{
				int postInd;

				postInd = pGLPostGLConArr[glInd][k];

				pPreConArr[i][numpPreCon[i]] = postInd;
				numpPreCon[i]++;

				pPostConArr[postInd][numpPostCon[postInd]] = i;
				numpPostCon[postInd]++;
			}
		}
	}
}

