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

	std::cout << "connecting GO to GO" << endl;
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

InNetConnectivityState::InNetConnectivityState(ConnectivityParams *parameters, fstream &infile)
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
	cout << "check2" << endl;

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
	std::cout << "Writing input network connectivity state to disk..." << std:endl;
	stateRW(false, (fstream &)outfile);
	std::cout << "finished writing input network connectivity to disk." << std::endl;
}

bool InNetConnectivityState::equivalent(const InNetConnectivityState &compState)
{
	bool eq = true;
	
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

vector<int> InNetConnectivityState::getpGOfromGOtoGLCon(int goN)
{
	return getConCommon(goN, numpGOfromGOtoGL, pGOfromGOtoGL);
}

vector<int> InNetConnectivityState::getpGOfromGLtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromGLtoGO, pGOfromGLtoGO);
}

vector<int> InNetConnectivityState::getpMFfromMFtoGLCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGL, pMFfromMFtoGL);
}

vector<int> InNetConnectivityState::getpGLfromGLtoGRCon(int glN)
{
	return getConCommon(glN, numpGLfromGLtoGR, pGLfromGLtoGR);
}


vector<int> InNetConnectivityState::getpGRfromMFtoGR(int grN)
{
	return getConCommon(grN, numpGRfromMFtoGR, pGRfromMFtoGR);
}

vector<vector<int> > InNetConnectivityState::getpGRPopfromMFtoGR()
{
	vector<vector<int> > retVect;

	retVect.resize(cp->numGR);

	for(int i=0; i<cp->numGR; i++)
	{
		retVect[i]=getpGRfromMFtoGR(i);
	}

	return retVect;
}


vector<int> InNetConnectivityState::getpGRfromGOtoGRCon(int grN)
{
	return getConCommon(grN, numpGRfromGOtoGR, pGRfromGOtoGR);
}
vector<vector<int> > InNetConnectivityState::getpGRPopfromGOtoGRCon()
{
	return getPopConCommon(cp->numGR, numpGRfromGOtoGR, pGRfromGOtoGR);
}

/* old Method
vector<vector<ct_uint32_t> > InNetConnectivityState::getpGRPopfromGOtoGR()
{
	vector<vector<ct_uint32_t> > retVect;

	retVect.resize(cp->numGR);

	for(int i=0; i<cp->numGR; i++)
	{
		retVect[i]=getpGRfromGOtoGR(i);
	}

	return retVect;
}
*/


vector<int> InNetConnectivityState::getpGRfromGRtoGOCon(int grN)
{
	return getConCommon(grN, numpGRfromGRtoGO, pGRfromGRtoGO);
}
vector<vector<int> > InNetConnectivityState::getpGRPopfromGRtoGOCon()
{
	return getPopConCommon(cp->numGR, numpGRfromGRtoGO, pGRfromGRtoGO);
}


vector<int> InNetConnectivityState::getpGOfromGRtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromGRtoGO, pGOfromGRtoGO);
}
vector<vector<int> > InNetConnectivityState::getpGOPopfromGRtoGOCon()
{
	return getPopConCommon(cp->numGO, numpGOfromGRtoGO, pGOfromGRtoGO);
}


vector<int> InNetConnectivityState::getpGOfromGOtoGRCon(int goN)
{
	return getConCommon(goN, numpGOfromGOtoGR, pGOfromGOtoGR);
}
vector<vector<int> > InNetConnectivityState::getpGOPopfromGOtoGRCon()
{
	return getPopConCommon(cp->numGO, numpGOfromGOtoGR, pGOfromGOtoGR);
}


vector<int> InNetConnectivityState::getpGOOutGOGOCon(int goN)
{
	return getConCommon(goN, numpGOGABAOutGOGO, pGOGABAOutGOGO);
}
vector<vector<int> > InNetConnectivityState::getpGOPopOutGOGOCon()
{
	return getPopConCommon(cp->numGO, numpGOGABAOutGOGO, pGOGABAOutGOGO);
}


vector<int> InNetConnectivityState::getpGOInGOGOCon(int goN)
{
	return getConCommon(goN, numpGOGABAInGOGO, pGOGABAInGOGO);
}
vector<vector<int> > InNetConnectivityState::getpGOPopInGOGOCon()
{
	return getPopConCommon(cp->numGO, numpGOGABAInGOGO, pGOGABAInGOGO);
}


vector<int> InNetConnectivityState::getpGOCoupOutGOGOCon(int goN)
{
	return getConCommon(goN, numpGOCoupOutGOGO, pGOCoupOutGOGO);
}
vector<vector<int> > InNetConnectivityState::getpGOPopCoupOutGOGOCon()
{
	return getPopConCommon(cp->numGO, numpGOCoupOutGOGO, pGOCoupOutGOGO);
}


vector<int> InNetConnectivityState::getpGOCoupInGOGOCon(int goN)
{
	return getConCommon(goN, numpGOCoupInGOGO, pGOCoupInGOGO);
}
vector<vector<int> > InNetConnectivityState::getpGOPopCoupInGOGOCon()
{
	return getPopConCommon(cp->numGO, numpGOCoupInGOGO, pGOCoupInGOGO);
}


vector<int> InNetConnectivityState::getpMFfromMFtoGRCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGR, pMFfromMFtoGR);
}

vector<int> InNetConnectivityState::getpMFfromMFtoGOCon(int mfN)
{
	return getConCommon(mfN, numpMFfromMFtoGO, pMFfromMFtoGO);
}


vector<int> InNetConnectivityState::getpGOfromMFtoGOCon(int goN)
{
	return getConCommon(goN, numpGOfromMFtoGO, pGOfromMFtoGO);
}

vector<vector<int> > InNetConnectivityState::getpGOPopfromMFtoGOCon()
{
	vector<vector<int> > con;

	con.resize(cp->numGO);
	for(int i=0; i<cp->numGO; i++)
	{
		con[i]=getpGOfromMFtoGOCon(i);
	}

	return con;
}

vector<ct_uint32_t> InNetConnectivityState::getConCommon(int cellN, ct_int32_t *numpCellCon, ct_uint32_t **pCellCon)
{
	vector<ct_uint32_t> inds;
	inds.resize(numpCellCon[cellN]);
	for(int i=0; i<numpCellCon[cellN]; i++)
	{
		inds[i]=pCellCon[cellN][i];
	}

	return inds;
}

vector<vector<ct_uint32_t> > InNetConnectivityState::getPopConCommon(int numCells, ct_int32_t *numpCellCon, ct_uint32_t **pCellCon)
{
	vector<vector<ct_uint32_t> > con;
	con.resize(numCells);
	for(int i=0; i<numCells; i++)
	{
		con[i].insert(con[i].end(), &pCellCon[i][0], &pCellCon[i][numpCellCon[i]]);
	}

	return con;
}
vector<int> InNetConnectivityState::getConCommon(int cellN, int *numpCellCon, int **pCellCon)
{
	vector<int> inds;
	inds.resize(numpCellCon[cellN]);
	for(int i=0; i<numpCellCon[cellN]; i++)
	{
		inds[i]=pCellCon[cellN][i];
	}

	return inds;
}
vector<vector<int> > InNetConnectivityState::getPopConCommon(int numCells, int *numpCellCon, int **pCellCon)
{
	vector<vector<int> > con;
	con.resize(numCells);
	for(int i=0; i<numCells; i++)
	{
		con[i].insert(con[i].end(), &pCellCon[i][0], &pCellCon[i][numpCellCon[i]]);
	}

	return con;
}

vector<ct_uint32_t> InNetConnectivityState::getGOIncompIndfromGRtoGO()
{
	vector<ct_uint32_t> goInds;

	for(int i=0; i<cp->numGO; i++)
	{
		if(numpGOfromGRtoGO[i]<cp->maxnumpGOfromGRtoGO)
		{
			goInds.push_back(i);
		}
	}

	return goInds;
}

vector<ct_uint32_t> InNetConnectivityState::getGRIncompIndfromGRtoGO()
{
	vector<ct_uint32_t> grInds;

	for(int i=0; i<cp->numGR; i++)
	{
		if(numpGRfromGRtoGO[i]<cp->maxnumpGRfromGRtoGO)
		{
			grInds.push_back(i);
		}
	}

	return grInds;
}

bool InNetConnectivityState::deleteGOGOConPair(int srcGON, int destGON)
{
	bool hasCon;

	int conN;
	hasCon=false;
	for(int i=0; i<numpGOGABAOutGOGO[srcGON]; i++)
	{
		if(pGOGABAOutGOGO[srcGON][i]==destGON)
		{
			hasCon=true;
			conN=i;
			break;
		}
	}
	if(!hasCon)
	{
		return hasCon;
	}

	for(int i=conN; i<numpGOGABAOutGOGO[srcGON]-1; i++)
	{
		pGOGABAOutGOGO[srcGON][i]=pGOGABAOutGOGO[srcGON][i+1];
	}
	numpGOGABAOutGOGO[srcGON]--;

	for(int i=0; i<numpGOGABAInGOGO[destGON]; i++)
	{
		if(pGOGABAInGOGO[destGON][i]==srcGON)
		{
			conN=i;
		}
	}
	for(int i=conN; i<numpGOGABAInGOGO[destGON]-1; i++)
	{
		pGOGABAInGOGO[destGON][i]=pGOGABAInGOGO[destGON][i+1];
	}
	numpGOGABAInGOGO[destGON]--;

	return hasCon;
}

bool InNetConnectivityState::addGOGOConPair(int srcGON, int destGON)
{
	if(numpGOGABAOutGOGO[srcGON]>=cp->maxnumpGOGABAOutGOGO ||
			numpGOGABAInGOGO[destGON]>=cp->maxnumpGOGABAInGOGO)
	{
		return false;
	}

	pGOGABAOutGOGO[srcGON][numpGOGABAOutGOGO[srcGON]]=destGON;
	numpGOGABAOutGOGO[srcGON]++;

	pGOGABAInGOGO[destGON][numpGOGABAInGOGO[destGON]]=srcGON;
	numpGOGABAInGOGO[destGON]++;

	return true;
}

void InNetConnectivityState::allocateMemory()
{

	numpGLfromGLtoGO=new int[cp->numGL];
	pGLfromGLtoGO=allocate2DArray<int>(cp->numGL, cp->maxnumpGLfromGLtoGO);

	haspGLfromGOtoGL=new int[cp->numGL];
	numpGLfromGOtoGL=new int[cp->numGL];
	pGLfromGOtoGL=allocate2DArray<int>(cp->numGL, cp->maxnumpGLfromGOtoGL);

	numpGLfromGLtoGR=new int[cp->numGL];
	pGLfromGLtoGR=allocate2DArray<int>(cp->numGL, cp->maxnumpGLfromGLtoGR);
	spanArrayGRtoGLX = new int[5];
	spanArrayGRtoGLY = new int[5];
	xCoorsGRGL = new int[25];
	yCoorsGRGL = new int[25];
	
	//mf
	haspGLfromMFtoGL=new int[cp->numGL];
	pGLfromMFtoGL=new int[cp->numGL];
	numpMFfromMFtoGL=new int[cp->numMF];
	pMFfromMFtoGL=allocate2DArray<int>(cp->numMF, 40);
	spanArrayMFtoGLX = new int[cp->glX];
	spanArrayMFtoGLY = new int[cp->glY];
	xCoorsMFGL = new int[cp->numpMFtoGL];
	yCoorsMFGL = new int[cp->numpMFtoGL];

	numpMFfromMFtoGR=new int[cp->numMF];
	pMFfromMFtoGR=allocate2DArray<int>(cp->numMF, cp->maxnumpMFfromMFtoGR);
	numpMFfromMFtoGO=new int[cp->numMF];
	pMFfromMFtoGO=allocate2DArray<int>(cp->numMF, cp->maxnumpMFfromMFtoGO);

	//golgi
	numpGOfromGLtoGO=new int[cp->numGO];
	pGOfromGLtoGO=allocate2DArray<int>(cp->numGO, cp->maxnumpGOfromGLtoGO);

	numpGOfromGOtoGL=new int[cp->numGO];
	pGOfromGOtoGL=allocate2DArray<int>(cp->numGO, cp->maxnumpGOfromGOtoGL);
	spanArrayGOtoGLX=new int[cp->spanGOtoGLX+1];
	spanArrayGOtoGLY=new int [cp->spanGOtoGLY+1];
	xCoorsGOGL=new int[cp->numpGOGL];
	yCoorsGOGL=new int[cp->numpGOGL];
	PconGOGL=new float[cp->numpGOGL];

	numpGOfromMFtoGO=new int[cp->numGO];
	pGOfromMFtoGO=allocate2DArray<int>(cp->numGO, 40);

	numpGOfromGOtoGR=new int[cp->numGO];
	pGOfromGOtoGR=allocate2DArray<int>(cp->numGO, cp->maxnumpGOfromGOtoGR);

	numpGOfromGRtoGO=new int[cp->numGO];
	pGOfromGRtoGO=allocate2DArray<int>(cp->numGO, cp->maxnumpGOfromGRtoGO);
	spanArrayPFtoGOX = new int[cp->grX + 1];
	spanArrayPFtoGOY = new int[150 + 1];
	xCoorsPFGO = new int[(cp->grX + 1)*(151)];
	yCoorsPFGO = new int[(cp->grX + 1)*(151)];	
	spanArrayAAtoGOX = new int[202];
	spanArrayAAtoGOY = new int[202];
	xCoorsAAGO = new int[202*202];
	yCoorsAAGO = new int[202*202];

	numpGOCoupInGOGO=new int[cp->numGO];
	pGOCoupInGOGO=allocate2DArray<int>(cp->numGO, 81);
	pGOCoupInGOGOCCoeff=allocate2DArray<float>(cp->numGO, 81);
	numpGOCoupOutGOGO=new int[cp->numGO];
	pGOCoupOutGOGO=allocate2DArray<int>(cp->numGO, 81);
	pGOCoupOutGOGOCCoeff=allocate2DArray<float>(cp->numGO, 81);
	gjConBool=allocate2DArray<bool>(cp->numGO, cp->numGO);
	spanArrayGOtoGOgjX=new int[9];
	spanArrayGOtoGOgjY=new int[9];
	xCoorsGOGOgj=new int[9*9];
	yCoorsGOGOgj=new int[9*9];
	gjPcon=new float[9*9];
	gjCC=new float[9*9];

	//granule
	pGRDelayMaskfromGRtoBSP=new ct_uint32_t[cp->numGR];

	numpGRfromGLtoGR=new int[cp->numGR];
	pGRfromGLtoGR=allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromGLtoGR);

	numpGRfromGRtoGO=new int[cp->numGR];
	pGRfromGRtoGO=allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromGRtoGO);
	pGRDelayMaskfromGRtoGO=allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromGRtoGO);

	numpGRfromGOtoGR=new int[cp->numGR];
	pGRfromGOtoGR=allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromGOtoGR);

	numpGRfromMFtoGR=new int[cp->numGR];
	pGRfromMFtoGR=allocate2DArray<int>(cp->numGR, cp->maxnumpGRfromMFtoGR);
}

void InNetConnectivityState::stateRW(bool read, std::fstream &file)
{
	cout<<"glomerulus"<<endl;
	//glomerulus
	rawBytesRW((char *)haspGLfromMFtoGL, cp->numGL*sizeof(int), read, file);
	cout<<"glomerulus 1.1"<<endl;
	rawBytesRW((char *)pGLfromMFtoGL, cp->numGL*sizeof(int), read, file);

	cout<<"glomerulus 2"<<endl;
	rawBytesRW((char *)numpGLfromGLtoGO, cp->numGL*sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGO[0], cp->numGL*cp->maxnumpGLfromGLtoGO*sizeof(int), read, file);

	cout<<"glomerulus 3"<<endl;
	rawBytesRW((char *)numpGLfromGOtoGL, cp->numGL*sizeof(int), read, file);
	rawBytesRW((char *)haspGLfromGOtoGL, cp->numGL*sizeof(int), read, file);
	//rawBytesRW((char *)pGLfromGOtoGL, cp->numGL*sizeof(int), read, file);

	cout<<"glomerulus 4"<<endl;
	rawBytesRW((char *)numpGLfromGLtoGR, cp->numGL*sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGR[0], cp->numGL*cp->maxnumpGLfromGLtoGR*sizeof(int), read, file);

	cout<<"mf"<<endl;
	//mossy fibers
	rawBytesRW((char *)numpMFfromMFtoGL, cp->numMF*sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGL[0], cp->numMF*20*sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGR, cp->numMF*sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGR[0], cp->numMF*cp->maxnumpMFfromMFtoGR*sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGO, cp->numMF*sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGO[0], cp->numMF*cp->maxnumpMFfromMFtoGO*sizeof(int), read, file);

	cout<<"golgi"<<endl;
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

	cout<<"granule"<<endl;
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
	arrayInitialize<int>(haspGLfromMFtoGL, 0, cp->numGL);
	arrayInitialize<int>(pGLfromMFtoGL, UINT_MAX, cp->numGL);

	arrayInitialize<int>(numpGLfromGLtoGO, 0, cp->numGL);
	arrayInitialize<int>(pGLfromGLtoGO[0], UINT_MAX, cp->numGL*cp->maxnumpGLfromGLtoGO);

	arrayInitialize<int>(haspGLfromGOtoGL, 0, cp->numGL);
	arrayInitialize<int>(numpGLfromGOtoGL, 0, cp->numGL);
	arrayInitialize<int>(pGLfromGOtoGL[0], UINT_MAX, cp->numGL*cp->maxnumpGLfromGOtoGL);

	arrayInitialize<int>(numpGLfromGLtoGR, 0, cp->numGL);
	arrayInitialize<int>(pGLfromGLtoGR[0], UINT_MAX, cp->numGL*cp->maxnumpGLfromGOtoGL);
	arrayInitialize<int>(spanArrayGRtoGLX, 0, 5);
	arrayInitialize<int>(spanArrayGRtoGLY, 0, 5);
	arrayInitialize<int>(xCoorsGRGL, 0, 25);
	arrayInitialize<int>(yCoorsGRGL, 0, 25);
	
	//mf
	arrayInitialize<int>(haspGLfromMFtoGL, 0, cp->numGL);
	arrayInitialize<int>(numpMFfromMFtoGL, 0, cp->numMF);
	arrayInitialize<int>(pMFfromMFtoGL[0], UINT_MAX, cp->numMF*20);
	arrayInitialize<int>(spanArrayMFtoGLX, 0, cp->glX);
	arrayInitialize<int>(spanArrayMFtoGLY, 0, cp->glY);
	arrayInitialize<int>(xCoorsMFGL, 0, cp->numpMFtoGL);
	arrayInitialize<int>(yCoorsMFGL, 0, cp->numpMFtoGL);

	arrayInitialize<int>(numpMFfromMFtoGR, 0, cp->numMF);
	arrayInitialize<int>(pMFfromMFtoGR[0], UINT_MAX, cp->numMF*cp->maxnumpMFfromMFtoGR);
	arrayInitialize<int>(numpMFfromMFtoGO, 0, cp->numMF);
	arrayInitialize<int>(pMFfromMFtoGO[0], UINT_MAX, cp->numMF*cp->maxnumpMFfromMFtoGO);

	arrayInitialize<int>(numpGOfromGLtoGO, 0, cp->numGO);
	arrayInitialize<int>(pGOfromGLtoGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromGLtoGO);

	arrayInitialize<int>(numpGOfromGOtoGL, 0, cp->numGO);
	arrayInitialize<int>(pGOfromGOtoGL[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromGOtoGL);
	arrayInitialize<float>(PconGOGL, 0, cp->numpGOGL);

	arrayInitialize<int>(numpGOfromMFtoGO, 0, cp->numGO);
	arrayInitialize<int>(pGOfromMFtoGO[0], UINT_MAX, cp->numGO*16);

	arrayInitialize<int>(numpGOfromGOtoGR, 0, cp->numGO);
	arrayInitialize<int>(pGOfromGOtoGR[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromGOtoGR);

	arrayInitialize<int>(numpGOfromGRtoGO, 0, cp->numGO);
	arrayInitialize<int>(pGOfromGRtoGO[0], UINT_MAX, cp->numGO*cp->maxnumpGOfromGRtoGO);
	
	arrayInitialize<int>(spanArrayPFtoGOX, 0, cp->grX+1);
	arrayInitialize<int>(spanArrayPFtoGOY, 0, 121);
	arrayInitialize<int>(xCoorsPFGO, 0, (cp->grX+1)*121);
	arrayInitialize<int>(yCoorsPFGO, 0, (cp->grX+1)*121);
	
	arrayInitialize<int>(spanArrayAAtoGOX, 0, 202);
	arrayInitialize<int>(spanArrayAAtoGOY, 0, 202);
	arrayInitialize<int>(xCoorsAAGO, 0, 202*202);
	arrayInitialize<int>(yCoorsAAGO, 0, 202*202);

	arrayInitialize<int>(numpGOCoupInGOGO, 0, cp->numGO);
	arrayInitialize<int>(pGOCoupInGOGO[0], UINT_MAX, cp->numGO*81);
	arrayInitialize<float>(pGOCoupInGOGOCCoeff[0], UINT_MAX, cp->numGO*81);
	arrayInitialize<int>(numpGOCoupOutGOGO, 0, cp->numGO);
	arrayInitialize<int>(pGOCoupOutGOGO[0], UINT_MAX, cp->numGO*81);
	arrayInitialize<float>(pGOCoupOutGOGOCCoeff[0], UINT_MAX, cp->numGO*81);
	arrayInitialize<bool>(gjConBool[0], false, cp->numGO*cp->numGO);
	
	arrayInitialize<int>(spanArrayGOtoGOgjX, 0, 9);
	arrayInitialize<int>(spanArrayGOtoGOgjY, 0, 9);
	arrayInitialize<int>(xCoorsGOGOgj, 0, 9*9);
	arrayInitialize<int>(yCoorsGOGOgj, 0, 9*9);
	arrayInitialize<float>(gjPcon, 0, 9*9);
	arrayInitialize<float>(gjCC, 0, 9*9);
	
	arrayInitialize<ct_uint32_t>(pGRDelayMaskfromGRtoBSP, 0, cp->numGR);

	arrayInitialize<int>(numpGRfromGLtoGR, 0, cp->numGR);
	arrayInitialize<int>(pGRfromGLtoGR[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromGLtoGR);

	arrayInitialize<int>(numpGRfromGRtoGO, 0, cp->numGR);
	arrayInitialize<int>(pGRfromGRtoGO[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromGRtoGO);
	arrayInitialize<int>(pGRDelayMaskfromGRtoGO[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromGRtoGO);

	arrayInitialize<int>(numpGRfromGOtoGR, 0, cp->numGR);
	arrayInitialize<int>(pGRfromGOtoGR[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromGOtoGR);

	arrayInitialize<int>(numpGRfromMFtoGR, 0, cp->numGR);
	arrayInitialize<int>(pGRfromMFtoGR[0], UINT_MAX, cp->numGR*cp->maxnumpGRfromMFtoGR);
}

void InNetConnectivityState::connectGLUBC()
{

//ubcX = 64;
//ubcY = 48;
//numUBC = ubcX*ubcY;

//int spanGLtoUBCX = 2; //cp->glX; //50 
//int spanGLtoUBCY = 2; //cp->glY; //160
//int numpGLtoUBC = (spanGLtoUBCX+1)*(spanGLtoUBCY+1);

//Make Span Array: Complete	
	for(int i=0; i<cp->spanGLtoUBCX+1;i++){
		int ind = cp->spanGLtoUBCX - i;
		spanArrayGLtoUBCX[i] = (cp->spanGLtoUBCX/2) - ind;}
	for(int i=0; i<cp->spanGLtoUBCY+1;i++){
		int ind = cp->spanGLtoUBCY - i;
		spanArrayGLtoUBCY[i] = (cp->spanGLtoUBCY/2) - ind;}
		
	for(int i=0; i<cp->numpGLtoUBC; i++)
	{
		xCoorsGLUBC[i] = spanArrayGLtoUBCX[ i%(cp->spanGLtoUBCX+1) ];
		yCoorsGLUBC[i] = spanArrayGLtoUBCY[ i/(cp->spanGLtoUBCX+1) ];		
	}



// Grid Scale: Complete
	float gridXScaleSrctoDest =(float)cp->ubcX / (float)cp->glX; 
	float gridYScaleSrctoDest =(float)cp->ubcY / (float)cp->glY; 

//Make Random Mossy Fiber Index Array: Complete	
	vector<int> rUBCInd;
	rUBCInd.assign(cp->numUBC,0);
	for(int i=0; i<cp->numUBC; i++)
	{
		rUBCInd[i] = i;
	}

	random_shuffle(rUBCInd.begin(), rUBCInd.end());

//Make Random Span Array: Complete
	vector<int> rUBCSpanInd;
	rUBCSpanInd.assign(cp->numpGLtoUBC,0);
	for(int ind=0; ind<cp->numpGLtoUBC; ind++)
	{rUBCSpanInd[ind] = ind;}
	
	random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	
	

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
for(int attempts=0; attempts<4; attempts++)
{
	random_shuffle(rUBCInd.begin(), rUBCInd.end());	

	for(int i=0; i<cp->numUBC; i++)
	{	
		//Select MF Coordinates from random index array: Complete	
		srcPosX = rUBCInd[i]%cp->ubcX;
		srcPosY = rUBCInd[i]/cp->ubcX;		
		
		random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	
		for(int j=0; j<cp->numpGLtoUBC; j++)
		{	
			
			if( numpUBCfromGLtoUBC[ rUBCInd[i] ] == UBCInputs ){ break; }
			
			preDestPosX = xCoorsGLUBC[ rUBCSpanInd[j] ]; 
			preDestPosY = yCoorsGLUBC[ rUBCSpanInd[j] ];	

			tempDestPosX = (int)round(srcPosX/gridXScaleSrctoDest) + preDestPosX;
			tempDestPosY = (int)round(srcPosY/gridYScaleSrctoDest) + preDestPosY;	
			
			tempDestPosX = ((tempDestPosX%glX+glX)%glX);
			tempDestPosY = ((tempDestPosY%glY+glY)%glY);
			
			destInd = tempDestPosY*glX+tempDestPosX;
				
			pUBCfromGLtoUBC[ rUBCInd[i] ] = destInd;
			numpUBCfromGLtoUBC[ rUBCInd[i] ]++;
				
			pGLfromGLtoUBC[destInd] = rUBCInd[i];
			numpGLfromGLtoUBC[destInd]++;		
		
		}
	}
}

int totalGLtoUBC = 0;
for(int i=0; i<cp->numUBC; i++)
{
	totalGLtoUBC = totalGLtoUBC + numpUBCfromGLtoUBC[i];
}
cout << "Total GL to UBC connections:		" << totalGLtoUBC << endl; 


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

	//for(int i=0; i<10; i++)
	//{
	//	cout<<"numpGLfromGLtoGR["<<i<<"]: "<<numpGLfromGLtoGR[i]<<endl;
	//}
//	cout << endl;
//	for(int i=0; i<cp->numGR; i++)
//	{
//		if(numpGRfromGLtoGR[i] < 4)
//		{
//			cout << "	FUCK_GRtoGL!	" << endl;
//		}
//	}
	int count = 0;
	for(int i=0; i<cp->numGL; i++)
	{
		count = count + numpGLfromGLtoGR[i];
	}
	cout << "Total number of Glomeruli to Granule connections:	" << count << endl; 
	cout << "Correct number:                                 	" << cp->numGR*cp->maxnumpGRfromGLtoGR << endl;

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


//int spanGOtoGLX_local = 56;
//int spanGOtoGLY_local = 56;
//int numpGOtoGL_local = (spanGOtoGLX_local+1)*(spanGOtoGLY_local+1);
int initialGOOutput = 1;
float sigmaML = 100;//10.5;
float sigmaS = 100;//10.5;
float A = 0.01;//0.095;
float PconX;
float PconY;


// Make span Array
	for(int i=0; i<cp->spanGOGLX+1;i++){
		int ind = cp->spanGOGLX - i;
		spanArrayGOtoGLX[i] = (cp->spanGOGLX/2) - ind;}
	for(int i=0; i<cp->spanGOGLY+1;i++){
		int ind = cp->spanGOGLY - i;
		spanArrayGOtoGLY[i] = (cp->spanGOGLY/2) - ind;}
		
	for(int i=0; i<cp->numpGOGL; i++)
	{
		xCoorsGOGL[i] = spanArrayGOtoGLX[ i%(cp->spanGOGLX+1) ];
		yCoorsGOGL[i] = spanArrayGOtoGLY[ i/(cp->spanGOGLY+1) ];	
	}

//Make Random Golgi cell Index Array	
	vector<int> rGOInd;
	rGOInd.assign(cp->numGO,0);
	for(int i=0; i<cp->numGO; i++)
	{
		rGOInd[i] = i;
	}
	random_shuffle(rGOInd.begin(), rGOInd.end());
	

//Make Random Span Array
	vector<int> rGOSpanInd;
	rGOSpanInd.assign(cp->numpGOGL,0);
	for(int ind=0; ind<cp->numpGOGL; ind++)
	{rGOSpanInd[ind] = ind;}
	
	
	
// Probability of connection as a function of distance
	for(int i=0; i<cp->numpGOGL; i++)
	{
	
		PconX =  (xCoorsGOGL[i]*xCoorsGOGL[i]) / (2*(sigmaML*sigmaML));
		PconY = (yCoorsGOGL[i]*yCoorsGOGL[i]) / (2*(sigmaS*sigmaS));
		PconGOGL[i] = A * exp(-(PconX + PconY) );
	
	}
	
	// Remove self connection 
	for(int i=0; i<cp->numpGOGL;i++)
	{
		if( (xCoorsGOGL[i] == 0) && (yCoorsGOGL[i] == 0) )
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
for(int attempts=0; attempts<100; attempts++)
{
	//cout << attempts << endl;
	random_shuffle(rGOSpanInd.begin(), rGOSpanInd.end());	
	
	if(shitCounter == 0 ){ break; }
	
		
	for(int i=0; i<cp->numGO; i++) // Go through each golgi cell 
	{
		
		//Select GO Coordinates from random index array: Complete	
		srcPosX = rGOInd[i]%cp->goX;
		srcPosY = rGOInd[i]/cp->goX;	
		
		random_shuffle(rGOSpanInd.begin(), rGOSpanInd.end());	
		
		for(int j=0; j<cp->numpGOGL; j++)   
		{	
			
			preDestPosX = xCoorsGOGL[ rGOSpanInd[j] ]; // relative position of connection
			preDestPosY = yCoorsGOGL[ rGOSpanInd[j] ];	

			tempDestPosX = (int)round(srcPosX/gridXScaleSrctoDest) + preDestPosX; // relative position in terms of the other cell grid. 
			tempDestPosY = (int)round(srcPosY/gridYScaleSrctoDest) + preDestPosY;

			tempDestPosX = (tempDestPosX%cp->glX+cp->glX)%cp->glX; // Wrap around 
			tempDestPosY = (tempDestPosY%cp->glY+cp->glY)%cp->glY;
					
			destInd = tempDestPosY*cp->glX+tempDestPosX;// Change position to Index
				
			if( numpGOfromGOtoGL[ rGOInd[i] ] >= initialGOOutput + attempts ){ break; }
			//if ( randGen->Random()>=1-PconGOGL[ rGOSpanInd[j] ] && !haspGLfromGOtoGL[destInd]) 
			if ( randGen->Random()>=1-PconGOGL[ rGOSpanInd[j] ] && numpGLfromGOtoGL[destInd] < cp->maxnumpGLfromGOtoGL) 
			{	
				pGOfromGOtoGL[ rGOInd[i] ][ numpGOfromGOtoGL[rGOInd[i]] ] = destInd;
				numpGOfromGOtoGL[ rGOInd[i] ]++;

				pGLfromGOtoGL[ destInd ][ numpGLfromGOtoGL[destInd] ]  = rGOInd[i];
				//haspGLfromGOtoGL[destInd] = true;
				numpGLfromGOtoGL[destInd]++;
			}
		}
	}

shitCounter = 0;
totalGOGL = 0;
for(int i=0; i<cp->numGL; i++)
{
	//if(haspGLfromGOtoGL[i] == false)
	if(numpGLfromGOtoGL[i] < cp->maxnumpGLfromGOtoGL)
	{
		shitCounter++;
	}
	totalGOGL += numpGLfromGOtoGL[i];
	
}
}


//for(int i=0; i<10; i++){
//	for(int j=0; j<numpGLfromGOtoGL[i]; j++){
//		cout << pGLfromGOtoGL[i][j] << " ";
//	}
//	cout << endl;
//}

for(int i=0; i<20; i++){
	for(int j=0; j<numpGLfromGOtoGL[i]; j++){
		cout << pGLfromGOtoGL[i][j] << " ";
	}
	cout << endl;
}



	cout << "			Empty Glomeruli Counter:  " << shitCounter << endl;
	cout << "			total GO -> GL: " << totalGOGL << endl;
	cout << "			avg Num  GO -> GL Per GL: " << (float)totalGOGL / (float)cp->numGL << endl;

	


	/*ofstream fileGOGLconIn;
	fileGOGLconIn.open("GOGLInputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<55; j++){
			fileGOGLconIn << pGOfromGOtoGL[i][j] << " ";
		}
		fileGOGLconIn << endl;
	}
	
*/

	
}



void InNetConnectivityState::connectUBCGL()
{

//spanUBCtoGLX = 50;
//spanUBCtoGLY = 50;
//numpUBCtoGL = (spanUBCtoGLX+1)*(spanUBCtoGLY+1);
//ubcX = 64;
//ubcY = 48;
//numUBC = ubcX*ubcY;


	for(int i=0; i<cp->spanUBCtoGLX+1;i++){
		int ind = cp->spanUBCtoGLX - i;
		spanArrayUBCtoGLX[i] = (cp->spanUBCtoGLX/2) - ind;}
	for(int i=0; i<cp->spanUBCtoGLY+1;i++){
		int ind = cp->spanUBCtoGLY - i;
		spanArrayUBCtoGLY[i] = (cp->spanUBCtoGLY/2) - ind;}
		
	for(int i=0; i<cp->numpUBCtoGL; i++)
	{
		xCoorsUBCGL[i] = spanArrayUBCtoGLX[ i%(cp->spanUBCtoGLX+1) ];
		yCoorsUBCGL[i] = spanArrayUBCtoGLY[ i/(cp->spanUBCtoGLX+1) ];		
		
	}


// Grid Scale: Complete
	float gridXScaleSrctoDest =(float)cp->ubcX / (float)cp->glX; 
	float gridYScaleSrctoDest =(float)cp->ubcY / (float)cp->glY; 

//Make Random Mossy Fiber Index Array: Complete	
	vector<int> rUBCInd;
	rUBCInd.assign(cp->numUBC,0);
	for(int i=0; i<cp->numUBC; i++)
	{
		rUBCInd[i] = i;
	}
	random_shuffle(rUBCInd.begin(), rUBCInd.end());

//Make Random Span Array: Complete
	vector<int> rUBCSpanInd;
	rUBCSpanInd.assign(cp->numpUBCtoGL,0);
	for(int ind=0; ind<cp->numpUBCtoGL; ind++)
	{rUBCSpanInd[ind] = ind;}
	

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

for(int attempts=0; attempts<3; attempts++)
{
	random_shuffle(rUBCInd.begin(), rUBCInd.end());	

	for(int i=0; i<cp->numUBC; i++)
	{	
		
		//Select MF Coordinates from random index array: Complete	
		srcPosX = rUBCInd[i]%cp->ubcX;
		srcPosY = rUBCInd[i]/cp->ubcX;		
		

		random_shuffle(rUBCSpanInd.begin(), rUBCSpanInd.end());	
		
		for(int j=0; j<cp->numpUBCtoGL; j++)
		{	
			
			preDestPosX = xCoorsUBCGL[ rUBCSpanInd[j] ]; 
			preDestPosY = yCoorsUBCGL[ rUBCSpanInd[j] ];	

			tempDestPosX = (int)round(srcPosX/gridXScaleSrctoDest) + preDestPosX;
			tempDestPosY = (int)round(srcPosY/gridYScaleSrctoDest) + preDestPosY;	
			
			tempDestPosX = ((tempDestPosX%glX+glX)%glX);
			tempDestPosY = ((tempDestPosY%glY+glY)%glY);
			
			destInd = tempDestPosY*glX+tempDestPosX;
			
			
			if( destInd == pUBCfromGLtoUBC[rUBCInd[i]]  ||  numpUBCfromUBCtoGL[ rUBCInd[i] ] == UBCOutput ){ break; }
			
			if ( numpGLfromUBCtoGL[destInd] == 0 ) 
			{	
				pUBCfromUBCtoGL[ rUBCInd[i] ][ numpUBCfromUBCtoGL[rUBCInd[i]] ] = destInd;
				numpUBCfromUBCtoGL[ rUBCInd[i] ]++;	


				pGLfromUBCtoGL[destInd][ numpGLfromUBCtoGL[destInd] ] = rUBCInd[i];
				numpGLfromUBCtoGL[destInd]++;	
			
			}
		}
	}

}	


for(int i=0; i<10; i++)
{
	for(int j=0; j<numpUBCfromUBCtoGL[i]; j++)
	{
	//	cout << pUBCfromUBCtoGL[i][j] << " ";
	}
	//cout << endl;
}



}




void InNetConnectivityState::connectMFGL_withUBC(CRandomSFMT *randGen)
{


int initialMFOutput = 14;

//Make Span Array: Complete	
	for(int i=0; i<cp->spanMFtoGLX+1;i++){
		int ind = cp->spanMFtoGLX - i;
		spanArrayMFtoGLX[i] = (cp->spanMFtoGLX/2) - ind;}
	for(int i=0; i<cp->spanMFtoGLY+1;i++){
		int ind = cp->spanMFtoGLY - i;
		spanArrayMFtoGLY[i] = (cp->spanMFtoGLY/2) - ind;}
		
	for(int i=0; i<cp->numpMFtoGL; i++)
	{
		xCoorsMFGL[i] = spanArrayMFtoGLX[ i%(cp->spanMFtoGLX+1) ];
		yCoorsMFGL[i] = spanArrayMFtoGLY[ i/(cp->spanMFtoGLX+1) ];		
	}



// Grid Scale: Complete
	float gridXScaleSrctoDest =(float)cp->mfX / (float)cp->glX; 
	float gridYScaleSrctoDest =(float)cp->mfY / (float)cp->glY; 

//Make Random Mossy Fiber Index Array: Complete	
	vector<int> rMFInd;
	rMFInd.assign(cp->numMF,0);
	for(int i=0; i<cp->numMF; i++)
	{
		rMFInd[i] = i;	
	}
	random_shuffle(rMFInd.begin(), rMFInd.end());

//Make Random Span Array: Complete
	vector<int> rMFSpanInd;
	rMFSpanInd.assign(cp->numpMFtoGL,0);
	for(int ind=0; ind<cp->numpMFtoGL; ind++)
	{rMFSpanInd[ind] = ind;}
	

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
for(int attempts=0; attempts<3; attempts++)
{
//	cout << "MF attempt:	" << attempts << endl;
	random_shuffle(rMFInd.begin(), rMFInd.end());	
	//if(shitCounter == 0 ){ break; }

	for(int i=0; i<cp->numMF; i++)
	{	
		//Select MF Coordinates from random index array: Complete	
		srcPosX = rMFInd[i]%cp->mfX;
		srcPosY = rMFInd[i]/cp->mfX;		
		
		random_shuffle(rMFSpanInd.begin(), rMFSpanInd.end());	
		for(int j=0; j<cp->numpMFtoGL; j++)
		{	
			
			preDestPosX = xCoorsMFGL[ rMFSpanInd[j] ]; 
			preDestPosY = yCoorsMFGL[ rMFSpanInd[j] ];	

			tempDestPosX = (int)round(srcPosX/gridXScaleSrctoDest) + preDestPosX;
			tempDestPosY = (int)round(srcPosY/gridYScaleSrctoDest) + preDestPosY;	
			
			tempDestPosX = ((tempDestPosX%glX+glX)%glX);
			tempDestPosY = ((tempDestPosY%glY+glY)%glY);
			
			destInd = tempDestPosY*glX+tempDestPosX;
			
			if( numpMFfromMFtoGL[ rMFInd[i] ] == initialMFOutput + attempts ){ break; }
				
			if ( !haspGLfromMFtoGL[destInd] && numpGLfromUBCtoGL[destInd]==0 ) 
			{	
				
				pMFfromMFtoGL[ rMFInd[i] ][ numpMFfromMFtoGL[rMFInd[i]] ] = destInd;
				numpMFfromMFtoGL[ rMFInd[i] ]++;
				
				pGLfromMFtoGL[destInd] = rMFInd[i];
				haspGLfromMFtoGL[destInd] = true;	
			}
		}
	}
shitCounter = 0;
for(int i=0; i<cp->numGL; i++)
{
	if(haspGLfromMFtoGL[i] == false)
	{
		shitCounter++;
	}
}


}	
	cout << "			Empty Glomeruli Counter:  " << shitCounter << endl << endl;


	
	int count = 0;
	for(int i=0; i<cp->numMF; i++)
	{
		count = count + numpMFfromMFtoGL[i];
	}
	cout << "Total number of Mossy Fiber to Glomeruli connections:	" << count << endl;
	cout << "Correct number:                                    	" << cp->numGL << endl;


	/*for(int i=0; i<10; i++)
	{
		cout<<"numpMFfromMFtoGL["<<i<<"]: "<<numpMFfromMFtoGL[i]<<endl;
		for(int j=0; j<numpMFfromMFtoGL[i]; j++)
		{
			cout<<pMFfromMFtoGL[i][j]<<" ";
		}
		cout<<endl;
	}*/ 
	
	count = 0;
	for(int i=0; i<cp->numMF; i++)
	{
		for(int j=0; j<numpMFfromMFtoGL[i]; j++)
		{
			for(int k=0; k<numpMFfromMFtoGL[i]; k++)
			{
				if((pMFfromMFtoGL[i][j] == pMFfromMFtoGL[i][k]) & (j != k))
				{
					count = count + 1;
				}
			}
		}
	}

	cout << "Double Mossy Fiber to Glomeruli connecitons: 		" << count << endl;
	

}

void InNetConnectivityState::connectMFGL_noUBC(CRandomSFMT *randGen)
{

//int spanMFtoGLX = 50; //cp->glX; //50 
//int spanMFtoGLY = 200; //cp->glY; //160
//int numpMFtoGL = (spanMFtoGLX+1)*(spanMFtoGLY+1);

//int mfX = 128;
//int mfY = 96;

cout << cp->mfX << endl;
cout << cp->mfY << endl;
cout << cp->numMF << endl;

int initialMFOutput = 14;

//Make Span Array: Complete	
	for(int i=0; i<cp->spanMFtoGLX+1;i++){
		int ind = cp->spanMFtoGLX - i;
		spanArrayMFtoGLX[i] = (cp->spanMFtoGLX/2) - ind;}
	for(int i=0; i<cp->spanMFtoGLY+1;i++){
		int ind = cp->spanMFtoGLY - i;
		spanArrayMFtoGLY[i] = (cp->spanMFtoGLY/2) - ind;}
		
	for(int i=0; i<cp->numpMFtoGL; i++)
	{
		xCoorsMFGL[i] = spanArrayMFtoGLX[ i%(cp->spanMFtoGLX+1) ];
		yCoorsMFGL[i] = spanArrayMFtoGLY[ i/(cp->spanMFtoGLX+1) ];		
	}



// Grid Scale: Complete
	float gridXScaleSrctoDest =(float)cp->mfX / (float)cp->glX; 
	float gridYScaleSrctoDest =(float)cp->mfY / (float)cp->glY; 

//Make Random Mossy Fiber Index Array: Complete	
	vector<int> rMFInd;
	rMFInd.assign(cp->numMF,0);
	for(int i=0; i<cp->numMF; i++)
	{
		rMFInd[i] = i;	
	}
	random_shuffle(rMFInd.begin(), rMFInd.end());

//Make Random Span Array: Complete
	vector<int> rMFSpanInd;
	rMFSpanInd.assign(cp->numpMFtoGL,0);
	for(int ind=0; ind<cp->numpMFtoGL; ind++)
	{rMFSpanInd[ind] = ind;}
	

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

int glX = cp->glX;
int glY = cp->glY;
for(int attempts=0; attempts<4; attempts++)
{
	random_shuffle(rMFInd.begin(), rMFInd.end());	

	for(int i=0; i<cp->numMF; i++)
	{	
		//Select MF Coordinates from random index array: Complete	
		srcPosX = rMFInd[i]%cp->mfX;
		srcPosY = rMFInd[i]/cp->mfX;		
		
		random_shuffle(rMFSpanInd.begin(), rMFSpanInd.end());	
		for(int j=0; j<cp->numpMFtoGL; j++)
		{	
			
			preDestPosX = xCoorsMFGL[ rMFSpanInd[j] ]; 
			preDestPosY = yCoorsMFGL[ rMFSpanInd[j] ];	

			tempDestPosX = (int)round(srcPosX/gridXScaleSrctoDest) + preDestPosX;
			tempDestPosY = (int)round(srcPosY/gridYScaleSrctoDest) + preDestPosY;	
			
			tempDestPosX = ((tempDestPosX%glX+glX)%glX);
			tempDestPosY = ((tempDestPosY%glY+glY)%glY);
			
			destInd = tempDestPosY*glX+tempDestPosX;
			
			if( numpMFfromMFtoGL[ rMFInd[i] ] == initialMFOutput + attempts ){ break; }
				
			if ( !haspGLfromMFtoGL[destInd] ) 
			{	
				
				pMFfromMFtoGL[ rMFInd[i] ][ numpMFfromMFtoGL[rMFInd[i]] ] = destInd;
				numpMFfromMFtoGL[ rMFInd[i] ]++;
				
				pGLfromMFtoGL[destInd] = rMFInd[i];
				haspGLfromMFtoGL[destInd] = true;	
			}
		}
	}

}	


	
	int count = 0;
	for(int i=0; i<cp->numMF; i++)
	{
		count = count + numpMFfromMFtoGL[i];
	}
	cout << "Total number of Mossy Fiber to Glomeruli connections:	" << count << endl;
	cout << "Correct number:                                    	" << cp->numGL << endl;


/*	for(int i=0; i<10; i++)
	{
		cout<<"numpMFfromMFtoGL["<<i<<"]: "<<numpMFfromMFtoGL[i]<<endl;
		for(int j=0; j<numpMFfromMFtoGL[i]; j++)
		{
			cout<<pMFfromMFtoGL[i][j]<<" ";
		}
		cout<<endl;
	} 
	
	count = 0;
	for(int i=0; i<cp->numMF; i++)
	{
		for(int j=0; j<numpMFfromMFtoGL[i]; j++)
		{
			for(int k=0; k<numpMFfromMFtoGL[i]; k++)
			{
				if((pMFfromMFtoGL[i][j] == pMFfromMFtoGL[i][k]) & (j != k))
				{
					count = count + 1;
				}
			}
		}
	}

	cout << "Double Mossy Fiber to Glomeruli connecitons: 		" << count << endl;
*/	

}


void InNetConnectivityState::translateUBCGL()
{

int numUBCfromUBCtoGL = 10;
//UBC to GR 
	int grIndex;
	for(int i=0; i<cp->numUBC; i++)
	{
		for(int j=0; j<numUBCfromUBCtoGL; j++)
		{
				  	
			glIndex = pUBCfromUBCtoGL[i][j]; 
			
			for(int k=0; k<numpGLfromGLtoGR[glIndex]; k++)
			{
				grIndex = pGLfromGLtoGR[glIndex][k];

				pUBCfromUBCtoGR[i][ numpUBCfromUBCtoGR[i] ] = grIndex; 
				numpUBCfromUBCtoGR[i]++;			

				pGRfromUBCtoGR[grIndex][ numpGRfromUBCtoGR[grIndex] ] = i;
				numpGRfromUBCtoGR[grIndex]++;
			}
		}

	}
ofstream fileUBCGRconIn;
	fileUBCGRconIn.open("UBCGRInputcon.txt");
	for(int i=0; i<cp->numGR; i++){
		for(int j=0; j<cp->maxnumpGRfromGLtoGR; j++){
			fileUBCGRconIn << pGRfromUBCtoGR[i][j] << " ";
		}
		fileUBCGRconIn << endl;
	}






int grUBCInputCounter = 0;
for(int i=0; i<cp->numGR; i++)
{
	grUBCInputCounter = grUBCInputCounter + numpGRfromUBCtoGR[i];

}
cout << "	Total UBC inputs:		" << grUBCInputCounter << endl;

/*
for(int i=0; i<10; i++)
{
	for(int j=0; j<numpUBCfromUBCtoGR[i]; j++)
	{
		cout << pUBCfromUBCtoGR[i][j] << " ";
	}
	cout << endl;
}
*/
//UBC to GO

	int goIndex;
	for(int i=0; i<cp->numUBC; i++)
	{
		for(int j=0; j<numUBCfromUBCtoGL; j++)
		{
				  	
			glIndex = pUBCfromUBCtoGL[i][j]; 
			
			for(int k=0; k<numpGLfromGLtoGO[glIndex]; k++)
			{
				goIndex = pGLfromGLtoGO[glIndex][k];

				pUBCfromUBCtoGO[i][ numpUBCfromUBCtoGO[i] ] = goIndex; 
				numpUBCfromUBCtoGO[i]++;			

				pGOfromUBCtoGO[goIndex][ numpGOfromUBCtoGO[goIndex] ] = i;
				numpGOfromUBCtoGO[goIndex]++;
			}
		}

	}
ofstream fileUBCGOconIn;
	fileUBCGOconIn.open("UBCGOInputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<16; j++){
			fileUBCGOconIn << pGOfromUBCtoGO[i][j] << " ";
		}
		fileUBCGOconIn << endl;
	}

/*
for(int i=0; i<20; i++)
{
	for(int j=0; j<numpUBCfromUBCtoGO[i]; j++)
	{
		cout << pUBCfromUBCtoGO[i][j] << " ";
	}
	cout << endl;
}
*/





//UBC to UBC
	int ubcIndex;
	for(int i=0; i<cp->numUBC; i++)
	{
		for(int j=0; j<numUBCfromUBCtoGL; j++)
		{
				  	
			glIndex = pUBCfromUBCtoGL[i][j]; 
			
			//for(int k=0; k<numpGLfromGLtoUBC[glIndex]; k++)
			if(numpGLfromGLtoUBC[glIndex]==1)
			{
				ubcIndex = pGLfromGLtoUBC[glIndex];

				pUBCfromUBCOutUBC[i][ numpUBCfromUBCOutUBC[i] ] = ubcIndex; 
				numpUBCfromUBCOutUBC[i]++;			

				pUBCfromUBCInUBC[ubcIndex][ numpUBCfromUBCInUBC[ubcIndex] ] = i;
				numpUBCfromUBCInUBC[ubcIndex]++;
			}
		}

	}


/*for(int i=0; i<50; i++)
{
	for(int j=0; j<numpUBCfromUBCOutUBC[i]; j++)
	{
		cout << pUBCfromUBCOutUBC[i][j] << " ";
	}
	cout << endl;
}
*/

}




void InNetConnectivityState::translateMFGL()
{


// Mossy fiber to Granule
	for(int i=0; i<cp->numGR; i++)
	{
		for(int j=0; j<numpGRfromGLtoGR[i]; j++)
		{
		
			glIndex = pGRfromGLtoGR[i][j];
			if(haspGLfromMFtoGL[glIndex])
			{
				mfIndex = pGLfromMFtoGL[glIndex];

				pMFfromMFtoGR[mfIndex][ numpMFfromMFtoGR[mfIndex] ] = i; 
				numpMFfromMFtoGR[mfIndex]++;			

				pGRfromMFtoGR[i][ numpGRfromMFtoGR[i] ] = mfIndex;
				numpGRfromMFtoGR[i]++;
			}
		}
	}	

	/*ofstream fileMFGRconIn;
	fileMFGRconIn.open("MFGRInputcon.txt");
	for(int i=0; i<cp->numGR; i++){
		for(int j=0; j<4; j++){
			fileMFGRconIn << pGRfromMFtoGR[i][j] << " ";
		}
		fileMFGRconIn << endl;
	}
*/	


int grMFInputCounter = 0;
cout << cp->numGR << endl;
for(int i=0; i<100; i++){
	cout << numpGRfromMFtoGR[i] << " ";
}
cout << endl;
for(int i=0; i<cp->numGR; i++)
{
	grMFInputCounter = grMFInputCounter + numpGRfromMFtoGR[i];

}
cout << "	Total MF inputs:		" << grMFInputCounter << endl;

// Mossy fiber to Golgi	
	for(int i=0; i<cp->numGO; i++)
	{
		for(int j=0; j<numpGOfromGLtoGO[i]; j++)
		{
		
			glIndex = pGOfromGLtoGO[i][j];
			
			if(haspGLfromMFtoGL[glIndex])
			{
				mfIndex = pGLfromMFtoGL[glIndex];

				pMFfromMFtoGO[mfIndex][ numpMFfromMFtoGO[mfIndex] ] = i; 
				numpMFfromMFtoGO[mfIndex]++;			

				pGOfromMFtoGO[i][ numpGOfromMFtoGO[i] ] = mfIndex;
				numpGOfromMFtoGO[i]++;
			}
		}

	}









	
/*
// Mossy fiber to UBC
	for(int i=0; i<cp->numUBC; i++)
	{
		
		glIndex = pUBCfromGLtoUBC[i];
		
		if(haspGLfromMFtoGL[glIndex])
		{
			mfIndex = pGLfromMFtoGL[glIndex];
		

			pMFfromMFtoUBC[mfIndex][ numpMFfromMFtoUBC[mfIndex] ] = i; 
			numpMFfromMFtoUBC[mfIndex]++;			

			pUBCfromMFtoUBC[i] = mfIndex;
			numpUBCfromMFtoUBC[i]++;

		}
	
	}
*/
	

//	cout << endl << endl;
//	for(int i=0; i<cp->numUBC; i++)
//	{
//		cout << pUBCfromMFtoUBC[i] << "  ";
//	}


}

void InNetConnectivityState::translateGOGL(CRandomSFMT *randGen)
{
/*
	translateCommon(pGOfromGOtoGL, numpGOfromGOtoGL,
			pGLfromGLtoGR, numpGLfromGLtoGR,
			pGOfromGOtoGR, numpGOfromGOtoGR,
			pGRfromGOtoGR, numpGRfromGOtoGR,
			cp->numGO);
	*/
	for(int i=0; i<cp->numGR; i++)
	{
		for(int j=0; j<cp->maxnumpGRfromGOtoGR; j++)
		{
			for(int k=0; k<cp->maxnumpGLfromGOtoGL; k++){	
			
				if(numpGRfromGOtoGR[i] < cp->maxnumpGRfromGOtoGR){	
					glIndex = pGRfromGLtoGR[i][j];
				
					goIndex = pGLfromGOtoGL[glIndex][k];
					
					pGOfromGOtoGR[goIndex][ numpGOfromGOtoGR[goIndex] ] = i; 
					numpGOfromGOtoGR[goIndex]++;			

					pGRfromGOtoGR[i][ numpGRfromGOtoGR[i] ] = goIndex;
					numpGRfromGOtoGR[i]++;
				}
			}
		}

	}

	for(int i=0; i<cp->numGR; i++){
		totalGOGR += numpGRfromGOtoGR[i];
	}
	cout << "		total GO->GR:  " << totalGOGR << endl;
	cout << "		GO->GR Per GR:  " << (float)totalGOGR/(float)cp->numGR << endl;

	/*ofstream fileGOGRconIn;
	fileGOGRconIn.open("GOGRInputcon.txt");
	for(int i=0; i<cp->numGR; i++){
		for(int j=0; j<numpGRfromGOtoGR[i]; j++){
			fileGOGRconIn << pGRfromGOtoGR[i][j] << " ";
		}
		fileGOGRconIn << endl;
	}
*/
/*
// Golgi to UBC
	for(int i=0; i<cp->numUBC; i++)
	{
		
		glIndex = pUBCfromGLtoUBC[i];
		goIndex = pGLfromGOtoGL[glIndex];

		pGOfromGOtoUBC[goIndex][ numpGOfromGOtoUBC[goIndex] ] = i; 
		numpGOfromGOtoUBC[goIndex]++;			

		pUBCfromGOtoUBC[i][ numpUBCfromGOtoUBC[i] ] = goIndex;
		numpUBCfromGOtoUBC[i]++;

	}

*/
/*
for(int i=0; i<20; i++)
{
	cout << "numpUBCfromGOtoUBC: " << numpUBCfromGOtoUBC[i] << endl;

	for(int j=0; j<numpUBCfromGOtoUBC[i]; j++)
	{
		cout << pUBCfromGOtoUBC[i][j] << " ";
	}
	cout << endl;
}
*/

}


void InNetConnectivityState::connectGRGO(CRandomSFMT *randGen, int goRecipParam)
{

int pfConv[8] = {3750, 3000, 2250, 1500, 750, 375, 188, 93}; 

int spanPFtoGOX = cp->grX;
int spanPFtoGOY = 150;
int numpPFtoGO = (spanPFtoGOX+1)*(spanPFtoGOY+1);
int maxPFtoGOInput = pfConv[goRecipParam];

//PARALLEL FIBER TO GOLGI 
	for(int i=0; i<spanPFtoGOX+1;i++){
		int ind = spanPFtoGOX - i;
		spanArrayPFtoGOX[i] = (spanPFtoGOX/2) - ind;}
	for(int i=0; i<spanPFtoGOY+1;i++){
		int ind = spanPFtoGOY - i;
		spanArrayPFtoGOY[i] = (spanPFtoGOY/2) - ind;}
		
	for(int i=0; i<numpPFtoGO; i++)
	{
		xCoorsPFGO[i] = spanArrayPFtoGOX[ i%(spanPFtoGOX+1) ];
		yCoorsPFGO[i] = spanArrayPFtoGOY[ i/(spanPFtoGOX+1) ];	
	}

// Grid Scale: Complete
	float gridXScaleSrctoDest =(float)cp->goX / (float)cp->grX; 
	float gridYScaleSrctoDest =(float)cp->goY / (float)cp->grY; 



//Make Random Span Array: Complete
	vector<int> rPFSpanInd;
	rPFSpanInd.assign(numpPFtoGO,0);
	for(int ind=0; ind<numpPFtoGO; ind++)
	{rPFSpanInd[ind] = ind;}

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;



int grX = cp->grX;
int grY = cp->grY;
for(int atmps = 0; atmps<5; atmps++){
	for(int i=0; i<cp->numGO; i++)
	{	
		srcPosX = i%cp->goX;
		srcPosY = i/cp->goX;
		
		random_shuffle(rPFSpanInd.begin(), rPFSpanInd.end());		
		
		for(int j=0; j<maxPFtoGOInput; j++)
		{	

			preDestPosX = xCoorsPFGO[ rPFSpanInd[j] ]; 
			preDestPosY = yCoorsPFGO[ rPFSpanInd[j] ];	
			
			tempDestPosX = (int)round(srcPosX/gridXScaleSrctoDest) + preDestPosX;
			tempDestPosY = (int)round(srcPosY/gridYScaleSrctoDest) + preDestPosY;
			
			tempDestPosX = (tempDestPosX%grX+grX)%grX;
			tempDestPosY = (tempDestPosY%grY+grY)%grY;
					
			destInd = tempDestPosY*cp->grX+tempDestPosX;

			if(/*numpGRfromGRtoGO[destInd] < 22 &&*/ numpGOfromGRtoGO[i] <  maxPFtoGOInput){	
				pGOfromGRtoGO[i][ numpGOfromGRtoGO[i] ] = destInd;
				numpGOfromGRtoGO[i]++;

				pGRfromGRtoGO[destInd][ numpGRfromGRtoGO[destInd] ] = i;
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
	for(int i=0; i<spanAAtoGOX+1;i++){
		int ind = spanAAtoGOX - i;
		spanArrayAAtoGOX[i] = (spanAAtoGOX/2) - ind;}
	for(int i=0; i<spanAAtoGOY+1;i++){
		int ind = spanAAtoGOY - i;
		spanArrayAAtoGOY[i] = (spanAAtoGOY/2) - ind;}
		
	for(int i=0; i<numpAAtoGO; i++)
	{
		xCoorsAAGO[i] = spanArrayAAtoGOX[ i%(spanAAtoGOX+1) ];
		yCoorsAAGO[i] = spanArrayAAtoGOY[ i/(spanAAtoGOX+1) ];	
	}




//Make Random Span Array: Complete
	vector<int> rAASpanInd;
	rAASpanInd.assign(numpAAtoGO,0);
	for(int ind=0; ind<numpAAtoGO; ind++)
	{rAASpanInd[ind] = ind;}

for(int atmps = 0; atmps<15; atmps++){


	for(int i=0; i<cp->numGO; i++)
	{	
		srcPosX = i%cp->goX;
		srcPosY = i/cp->goX;

		random_shuffle(rAASpanInd.begin(), rAASpanInd.end());		
		
		for(int j=0; j<maxAAtoGOInput; j++)
		{	

			preDestPosX = xCoorsAAGO[ rAASpanInd[j] ]; 
			preDestPosY = yCoorsAAGO[ rAASpanInd[j] ];	
			
			tempDestPosX = (int)round(srcPosX/gridXScaleSrctoDest) + preDestPosX;
			tempDestPosY = (int)round(srcPosY/gridYScaleSrctoDest) + preDestPosY;

			tempDestPosX = (tempDestPosX%grX+grX)%grX;
			tempDestPosY = (tempDestPosY%grY+grY)%grY;
					
			destInd = tempDestPosY*cp->grX+tempDestPosX;
			
			if(/*numpGRfromGRtoGO[destInd] < 22 &&*/ numpGOfromGRtoGO[i] <  maxAAtoGOInput + maxPFtoGOInput){	
					
				pGOfromGRtoGO[i][ numpGOfromGRtoGO[i] ] = destInd;	
				numpGOfromGRtoGO[i]++;

				pGRfromGRtoGO[destInd][ numpGRfromGRtoGO[destInd] ] = i;
				numpGRfromGRtoGO[destInd]++;	
			}

		}
	}

}	
int sumGOGR_GO = 0;
for(int i=0; i<cp->numGO; i++){
	sumGOGR_GO = sumGOGR_GO + numpGOfromGRtoGO[i];
}
cout << "	GRtoGO_GO:	" << sumGOGR_GO << endl;
int sumGOGR_GR = 0;
for(int i=0; i<cp->numGR; i++){
	sumGOGR_GR = sumGOGR_GR + numpGRfromGRtoGO[i];
}
cout << "	GRtoGO_GR:	" << sumGOGR_GR << endl;
/*	ofstream fileGOGRcon;
	fileGOGRcon.open("pGOfromGRtoGO.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<numpGOfromGRtoGO[i]; j++){
			fileGOGRcon << pGOfromGRtoGO[i][j] << " ";
		}
		fileGOGRcon << endl;
	}

*/

/*	connectCommon(pGOfromGRtoGO, numpGOfromGRtoGO,
			pGRfromGRtoGO, numpGRfromGRtoGO,
			cp->maxnumpGOfromGRtoGO, cp->numGO,
			cp->maxnumpGRfromGRtoGO, cp->maxnumpGRfromGRtoGO,
			cp->goX, cp->goY, cp->grX, cp->grY,
			cp->spanGOAscDenOnGRX, cp->spanGOAscDenOnGRY,
			20000, 50000, false,
			randGen);
*/
}


void InNetConnectivityState::connectGOGODecayP(CRandomSFMT *randGen, int goRecipParam, int simNum) {

	//int numberConnections[12] = {4,8,12,16,20,24,28,32,36,40,44,48};
	int numberConnections[26] = {1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50};

	//int numberConnections[5] = {14, 16, 18, 20, 22};

	int span = 4;
	int numCon = 12;//numberConnections[goRecipParam];
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
	arrayInitialize<int>(numpGOGABAInGOGO, 0, cp->numGO);
	arrayInitialize<int>(pGOGABAInGOGO[0], UINT_MAX, cp->numGO*numCon);
	arrayInitialize<int>(numpGOGABAOutGOGO, 0, cp->numGO);
	arrayInitialize<int>(pGOGABAOutGOGO[0], UINT_MAX, cp->numGO*numCon);
	arrayInitialize<bool>(conGOGOBoolOut[0], false, cp->numGO*cp->numGO);
	arrayInitialize<int>(spanArrayGOtoGOsynX, 0, span+1);
	arrayInitialize<int>(spanArrayGOtoGOsynY, 0, span+1);
	arrayInitialize<int>(xCoorsGOGOsyn, 0, numP);
	arrayInitialize<int>(yCoorsGOGOsyn, 0, numP);
	arrayInitialize<float>(Pcon, 0, numP);


	float A = 0.35;
	float sig = 1000;//1.95;

	float recipParamArray_P[11] = {1.0, 0.925, 0.78, 0.62, 0.47, 0.32, 0.19, 0.07, 0.0, 0.0, 0.0};
	float recipParamArray_LowerP[11] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.35, 0.0}; 
	bool recipParamArray_ReduceBase[11] = {false, false, false, false, false, false, false, false, true, true, false};
	bool recipParamArray_noRecip[11] = {false, false, false, false, false, false, false, false, false, false, true};
	 

	float pRecipGOGO = 1.0;//recipParamArray_P[goRecipParam];
	bool noRecip = false;//recipParamArray_noRecip[goRecipParam];
	float pRecipLowerBase = 0.0;//recipParamArray_LowerP[goRecipParam];
	bool reduceBaseRecip = false;//recipParamArray_ReduceBase[goRecipParam];

	//int maxGOOut = 8;
	float PconX;
	float PconY;


	for (int i = 0; i < span + 1; i++) {
		int ind = span - i;
		spanArrayGOtoGOsynX[i] = (span / 2) - ind;
	}
	
	for (int i = 0; i < span + 1; i++) {
		int ind = span - i;
		spanArrayGOtoGOsynY[i] = (span / 2) - ind;
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
	// NOTE: there is a more efficient way of doing this
	for (int i = 0; i < numP; i++) {
		if ((xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0)) Pcon[i] = 0;
	}
	
	std::vector<int> rGOGOSpanInd;
	rGOGOSpanInd.assign(numP, 0);
	for (int ind = 0; ind < numP; ind++) {
		rGOGOSpanInd[ind] = ind;
	}

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	for (int atmpt = 0; atmpt < 200; atmpt++) {
		
		for (int i = 0; i < this->cp->numGO; i++) {	
			srcPosX = i % this->cp->goX;
			srcPosY = i / this->cp->goX;	
			
			random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
			
			for (int j = 0; j < numP; j++) {	
				
				preDestPosX = xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
				preDestPosY = yCoorsGOGOsyn[rGOGOSpanInd[j]];	

				tempDestPosX = srcPosX + preDestPosX;
				tempDestPosY = srcPosY + preDestPosY;

				tempDestPosX = (tempDestPosX % this->cp->goX + this->cp->goX) % this->cp->goX;
				tempDestPosY = (tempDestPosY % this->cp->goY + this->cp->goY) % this->cp->goY;
						
				destInd = tempDestPosY * this->cp->goX + tempDestPosX;
			
				// Normal One	
				if ( !noRecip && !reduceBaseRecip && randGen->Random()>= 1 - Pcon[rGOGOSpanInd[j]]
						&& !conGOGOBoolOut[i][destInd] && numpGOGABAOutGOGO[i] < numCon
						&& numpGOGABAInGOGO[destInd] < numCon) {	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destInd;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = i;
					numpGOGABAInGOGO[destInd]++;
					
					conGOGOBoolOut[i][destInd] = true;
							
					if (randGen->Random() >= 1 - pRecipGOGO && !conGOGOBoolOut[destInd][i]
							&& numpGOGABAOutGOGO[destInd] < numCon && numpGOGABAInGOGO[i] < numCon) {
						
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
						&& numpGOGABAInGOGO[destInd] < numCon) {	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destInd;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destInd][numpGOGABAInGOGO[destInd]] = i;
					numpGOGABAInGOGO[destInd]++;
					
					conGOGOBoolOut[i][destInd] = true;	
				}

				if (noRecip && !reduceBaseRecip && randGen->Random() >= 1 - Pcon[rGOGOSpanInd[j]]
						&& (!conGOGOBoolOut[i][destInd]) && conGOGOBoolOut[destInd][i] == false && numpGOGABAOutGOGO[i] < numCon && numpGOGABAInGOGO[destInd] < numCon)
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

//	int conv[7] = {4000, 3000, 2000, 1000, 500, 250, 125};
	//ofstream fileGOGOconIn;

	//fileGOGOconIn.open("GOGOInputcon_PSTH_recip100_conv"+to_string(numberConnections[goRecipParam])+"_spanFlat4_"+to_string(simNum)+".txt");
/*	fileGOGOconIn.open("GOGOInputcon_Raster_recip100_conv"+to_string(numberConnections[goRecipParam])+"_spanFlat10_"+to_string(simNum)+".txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<numCon; j++){
			fileGOGOconIn << pGOGABAInGOGO[i][j] << " ";
		}
		fileGOGOconIn << endl;
	}
*/	
	/*ofstream fileGOGOconOut;
	fileGOGOconOut.open("GOGOOutputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<numCon; j++){
			fileGOGOconOut << pGOGABAOutGOGO[i][j] << " ";
		}
		fileGOGOconOut << endl;
	}*/

float totalGOGOcons = 0;
for(int i=0; i<cp->numGO; i++)
{
	totalGOGOcons = totalGOGOcons + numpGOGABAInGOGO[i];
}

cout << "Total GOGO connections:		" << totalGOGOcons << endl;
cout << "Average GOGO connections:	" << totalGOGOcons/float(cp->numGO) << endl;\
cout << cp->numGO << endl;
int recipCounter = 0;
for(int i=0; i<cp->numGO; i++)
{
	for(int j=0; j<numpGOGABAInGOGO[i]; j++)
	{
		for(int k=0; k<numpGOGABAOutGOGO[i]; k++)
		{
			if(pGOGABAInGOGO[i][j] == pGOGABAOutGOGO[i][k] && pGOGABAInGOGO[i][j] != UINT_MAX && pGOGABAOutGOGO[i][k] != UINT_MAX)
			{
				recipCounter = recipCounter + 1;	
			}
		}

	}
}
float fracRecip = recipCounter / totalGOGOcons;
cout << "FracRecip:			" << fracRecip << endl;








rGOGOSpanInd.clear();


}





void InNetConnectivityState::connectGOGODecay(CRandomSFMT *randGen)
{
//int spanGOtoGOsynX = 10;
//int spanGOtoGOsynY = 10;
//int numpGOtoGOsyn = (cp->spanGOGOsynX+1) * (cp->spanGOGOsynY+1);
//float sigmaML = 100;//1.89;
//float sigmaS = 100;//1.89;
float A = 0.01;
float pRecipGOGO = 1;
//int maxGOOut = 8;
float PconX;
float PconY;



	cout << cp->spanGOGOsynX << endl;
	cout << cp->spanGOGOsynY << endl;
	cout << cp->sigmaGOGOsynML << endl;
	cout << cp->sigmaGOGOsynS << endl;
	
	cout << cp->pRecipGOGOsyn << endl;
	cout << cp->maxGOGOsyn << endl;


	for(int i=0; i<cp->spanGOGOsynX+1;i++){
		int ind = cp->spanGOGOsynX - i;
		spanArrayGOtoGOsynX[i] = (cp->spanGOGOsynX/2) - ind;}
	for(int i=0; i<cp->spanGOGOsynY+1;i++){
		int ind = cp->spanGOGOsynY - i;
		spanArrayGOtoGOsynY[i] = (cp->spanGOGOsynY/2) - ind;}
		
	for(int i=0; i<cp->numpGOGOsyn; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[ i%(cp->spanGOGOsynX+1) ];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[ i/(cp->spanGOGOsynX+1) ];		
	}
	

	for(int i=0; i<cp->numpGOGOsyn; i++)
	{
	
		PconX =  (xCoorsGOGOsyn[i]*xCoorsGOGOsyn[i]) / (2*(cp->sigmaGOGOsynML*cp->sigmaGOGOsynML));
		PconY = (yCoorsGOGOsyn[i]*yCoorsGOGOsyn[i]) / (2*(cp->sigmaGOGOsynS*cp->sigmaGOGOsynS));
		Pcon[i] = A * exp(-(PconX + PconY) );
	
	}
	
	// Remove self connection 
	for(int i=0; i<cp->numpGOGOsyn;i++)
	{
		if( (xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0) )
		{
			Pcon[i] = 0;
		}
	}
	
	vector<int> rGOGOSpanInd;
	rGOGOSpanInd.assign(cp->numpGOGOsyn,0);
	for(int ind=0; ind<cp->numpGOGOsyn; ind++)
	{rGOGOSpanInd[ind] = ind;}
	

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;
for(int atmpt = 0; atmpt<200; atmpt++){
	
	for(int i=0; i<cp->numGO; i++)
	{	
		srcPosX = i%cp->goX;
		srcPosY = i/cp->goX;	
		
		random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
		
		for(int j=0; j<cp->numpGOGOsyn; j++)
		{	
			
			preDestPosX = xCoorsGOGOsyn[ rGOGOSpanInd[j] ]; 
			preDestPosY = yCoorsGOGOsyn[ rGOGOSpanInd[j] ];	

			tempDestPosX = srcPosX + preDestPosX;
			tempDestPosY = srcPosY + preDestPosY;

			tempDestPosX = (tempDestPosX%cp->goX+cp->goX)%cp->goX;
			tempDestPosY = (tempDestPosY%cp->goY+cp->goY)%cp->goY;
					
			destInd = tempDestPosY*cp->goX+tempDestPosX;

			if(randGen->Random()>=1-Pcon[rGOGOSpanInd[j] ] && (conGOGOBoolOut[i][destInd] == false) && numpGOGABAOutGOGO[i] < cp->maxGOGOsyn && numpGOGABAInGOGO[destInd] < cp->maxGOGOsyn)
			{	
				pGOGABAOutGOGO[i][ numpGOGABAOutGOGO[i] ] = destInd;
				numpGOGABAOutGOGO[i]++;
				
				pGOGABAInGOGO[destInd][ numpGOGABAInGOGO[destInd] ] = i;
				numpGOGABAInGOGO[destInd]++;
				
				conGOGOBoolOut[i][destInd] = true;
				
				if(randGen->Random() >= 1-pRecipGOGO && conGOGOBoolOut[destInd][i] == false && numpGOGABAOutGOGO[destInd] < cp->maxGOGOsyn && numpGOGABAInGOGO[i] < cp->maxGOGOsyn){
					
					pGOGABAOutGOGO[destInd][ numpGOGABAOutGOGO[destInd] ] = i;
					numpGOGABAOutGOGO[destInd]++;

					pGOGABAInGOGO[i][ numpGOGABAInGOGO[i] ] = destInd;
					numpGOGABAInGOGO[i]++;
					
					conGOGOBoolOut[ destInd ][i] = true;

				}
			
			}
		}
	}

}
ofstream fileGOGOconIn;
	fileGOGOconIn.open("GOGOInputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<cp->maxGOGOsyn; j++){
			fileGOGOconIn << pGOGABAInGOGO[i][j] << " ";
		}
		fileGOGOconIn << endl;
	}
	
	ofstream fileGOGOconOut;
	fileGOGOconOut.open("GOGOOutputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<cp->maxGOGOsyn; j++){
			fileGOGOconOut << pGOGABAOutGOGO[i][j] << " ";
		}
		fileGOGOconOut << endl;
	}

float totalGOGOcons = 0;
for(int i=0; i<cp->numGO; i++)
{
	totalGOGOcons = totalGOGOcons + numpGOGABAInGOGO[i];
}

cout << "Total GOGO connections:		" << totalGOGOcons << endl;
cout << "Average GOGO connections:	" << totalGOGOcons/float(cp->numGO) << endl;\
cout << cp->numGO << endl;
int recipCounter = 0;
for(int i=0; i<cp->numGO; i++)
{
	for(int j=0; j<numpGOGABAInGOGO[i]; j++)
	{
		for(int k=0; k<numpGOGABAOutGOGO[i]; k++)
		{
			if(pGOGABAInGOGO[i][j] == pGOGABAOutGOGO[i][k] && pGOGABAInGOGO[i][j] != UINT_MAX && pGOGABAOutGOGO[i][k] != UINT_MAX)
			{
				recipCounter = recipCounter + 1;	
			}
		}

	}
}
float fracRecip = recipCounter / totalGOGOcons;
cout << "FracRecip:			" << fracRecip << endl;








rGOGOSpanInd.clear();


}






void InNetConnectivityState::connectGOGOBias(CRandomSFMT *randGen)
{


int spanGOtoGOsynX = 12;//6
int spanGOtoGOsynY = 12;//18
int numpGOtoGOsyn = (spanGOtoGOsynX+1)*(spanGOtoGOsynY+1); 
//float conProbability = 0.03;//cp->conProbabilityGOGO;


//Make Span Array: Complete	
	for(int i=0; i<spanGOtoGOsynX+1;i++){
		int ind = spanGOtoGOsynX - i;
		spanArrayGOtoGOsynX[i] = (spanGOtoGOsynX/2) - ind;}
	for(int i=0; i<spanGOtoGOsynY+1;i++){
		int ind = spanGOtoGOsynY - i;
		spanArrayGOtoGOsynY[i] = (spanGOtoGOsynY/2) - ind;}
		
	for(int i=0; i<numpGOtoGOsyn; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[ i%(spanGOtoGOsynX+1) ];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[ i/(spanGOtoGOsynX+1) ];		
	}


//Make Random Mossy Fiber Index Array: Complete	
	vector<int> rGOInd;
	rGOInd.assign(cp->numGO,0);
	for(int i=0; i<cp->numGO; i++)
	{
		rGOInd[i] = i;	
	}
	random_shuffle(rGOInd.begin(), rGOInd.end());

//Make Random Span Array
	vector<int> rGOGOSpanInd;
	rGOGOSpanInd.assign(numpGOtoGOsyn,0);
	for(int ind=0; ind<numpGOtoGOsyn; ind++)
	{rGOGOSpanInd[ind] = ind;}


	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

int numOutputs = cp->numConGOGO;
int conType = 0;


	for(int i=0; i<cp->numGO; i++)
	{	
		srcPosX = rGOInd[i]%cp->goX;
		srcPosY = rGOInd[i]/cp->goX;	
		
		random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
			
		
		for(int j=0; j<numpGOtoGOsyn; j++)
		{	
			
	
			preDestPosX = xCoorsGOGOsyn[ rGOGOSpanInd[j] ]; 
			preDestPosY = yCoorsGOGOsyn[ rGOGOSpanInd[j] ];	

			tempDestPosX = srcPosX + preDestPosX;
			tempDestPosY = srcPosY + preDestPosY;

			tempDestPosX = (tempDestPosX%cp->goX+cp->goX)%cp->goX;
			tempDestPosY = (tempDestPosY%cp->goY+cp->goY)%cp->goY;
			

			destInd = tempDestPosY*cp->goX+tempDestPosX;
					
				
				
				
			if(randGen->Random() >= 1-0.0305 && conGOGOBoolOut[rGOInd[i]][destInd] == false){

				pGOGABAOutGOGO[ rGOInd[i] ][ numpGOGABAOutGOGO[rGOInd[i]] ] = destInd;
				numpGOGABAOutGOGO[ rGOInd[i] ]++;
					
				pGOGABAInGOGO[destInd][ numpGOGABAInGOGO[destInd] ] = rGOInd[i];
				numpGOGABAInGOGO[destInd]++;	
					
				conGOGOBoolOut[ rGOInd[i] ][destInd] = true;
			
				// conditional statement against making double output

				if(randGen->Random() >= 1-1 && conGOGOBoolOut[destInd][rGOInd[i]] == false){
					
					pGOGABAOutGOGO[destInd][ numpGOGABAOutGOGO[destInd] ] = rGOInd[i];
					numpGOGABAOutGOGO[destInd]++;

					pGOGABAInGOGO[ rGOInd[i] ][ numpGOGABAInGOGO[rGOInd[i]] ] = destInd;
					numpGOGABAInGOGO[rGOInd[i]]++;
					
					conGOGOBoolOut[ destInd ][ rGOInd[i] ] = true;

				}
			
			
			}	
		
		}
	}
		
	
	

	
/*
	ofstream fileGOGOconIn;
	fileGOGOconIn.open("GOGOInputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<30; j++){
			fileGOGOconIn << pGOGABAInGOGO[i][j] << " ";
		}
		fileGOGOconIn << endl;
	}
	
	ofstream fileGOGOconOut;
	fileGOGOconOut.open("GOGOOutputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<30; j++){
			fileGOGOconOut << pGOGABAOutGOGO[i][j] << " ";
		}
		fileGOGOconOut << endl;
	}
	
*/
if(conType == 0){
	
	for(int k=0; k<20; k++)
	{
		cout<<"numpGOGABAInGOGO["<<k<<"]: "<<numpGOGABAInGOGO[k]<<endl;
		for(int j=0; j<numpGOGABAInGOGO[k]; j++)
		{
			cout<<pGOGABAInGOGO[k][j]<<" ";
		}
		cout<<endl;
	}

	}



if(conType == 1){	
	
	int pNonRecip = 0;
	int missedOutCon = 0;
	int missedInCon = 0;

	for(int i=0; i<cp->numGO; i++){

		if(numpGOGABAInGOGO[i] != numpGOGABAOutGOGO[i]){ pNonRecip++; }
		if(numpGOGABAInGOGO[i] != numOutputs) { missedInCon++; }
		if(numpGOGABAOutGOGO[i] != numOutputs) { missedOutCon++; }
	}

	cout << "		" << "Potential non-reciprocal connection:		" << pNonRecip << endl;
	cout << "		" << "Missing Input:					" << missedInCon << endl;
	cout << "		" << "Missing Output:					" << missedOutCon << endl;
	}


}

void InNetConnectivityState::connectGOGO(CRandomSFMT *randGen)
{


int spanGOtoGOsynX = 12;//6
int spanGOtoGOsynY = 12;//18
int numpGOtoGOsyn = (spanGOtoGOsynX+1)*(spanGOtoGOsynY+1); 
//float conProbability = 0.03;//cp->conProbabilityGOGO;


//Make Span Array: Complete	
	for(int i=0; i<spanGOtoGOsynX+1;i++){
		int ind = spanGOtoGOsynX - i;
		spanArrayGOtoGOsynX[i] = (spanGOtoGOsynX/2) - ind;}
	for(int i=0; i<spanGOtoGOsynY+1;i++){
		int ind = spanGOtoGOsynY - i;
		spanArrayGOtoGOsynY[i] = (spanGOtoGOsynY/2) - ind;}
		
	for(int i=0; i<numpGOtoGOsyn; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[ i%(spanGOtoGOsynX+1) ];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[ i/(spanGOtoGOsynX+1) ];		
	}


//Make Random Mossy Fiber Index Array: Complete	
	vector<int> rGOInd;
	rGOInd.assign(cp->numGO,0);
	for(int i=0; i<cp->numGO; i++)
	{
		rGOInd[i] = i;	
	}
	random_shuffle(rGOInd.begin(), rGOInd.end());

//Make Random Span Array
	vector<int> rGOGOSpanInd;
	rGOGOSpanInd.assign(numpGOtoGOsyn,0);
	for(int ind=0; ind<numpGOtoGOsyn; ind++)
	{rGOGOSpanInd[ind] = ind;}


	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

int numOutputs = cp->numConGOGO;
int conType = 2;

for(int attempt = 1; attempt<numOutputs+1; attempt++)
{

	for(int i=0; i<cp->numGO; i++)
	{	
		srcPosX = rGOInd[i]%cp->goX;
		srcPosY = rGOInd[i]/cp->goX;	
		
		random_shuffle(rGOGOSpanInd.begin(), rGOGOSpanInd.end());		
			
		
		for(int j=0; j<numpGOtoGOsyn; j++)
		{	
			
	
			preDestPosX = xCoorsGOGOsyn[ rGOGOSpanInd[j] ]; 
			preDestPosY = yCoorsGOGOsyn[ rGOGOSpanInd[j] ];	

			tempDestPosX = srcPosX + preDestPosX;
			tempDestPosY = srcPosY + preDestPosY;

			tempDestPosX = (tempDestPosX%cp->goX+cp->goX)%cp->goX;
			tempDestPosY = (tempDestPosY%cp->goY+cp->goY)%cp->goY;
			

			destInd = tempDestPosY*cp->goX+tempDestPosX;
					
				
				
			if(conType == 0){ 	// Normal random connectivity		
				
				if(numpGOGABAOutGOGO[ rGOInd[i] ] == numOutputs){ break; }

				// conditional statment blocking the ability to make two outputs to the same cell
				pGOGABAOutGOGO[ rGOInd[i] ][ numpGOGABAOutGOGO[rGOInd[i]] ] = destInd;
				numpGOGABAOutGOGO[ rGOInd[i] ]++;
				
				pGOGABAInGOGO[destInd][ numpGOGABAInGOGO[destInd] ] = rGOInd[i];
				numpGOGABAInGOGO[destInd]++;
			
			}
			if(conType == 1){	// 100% reciprocal	
				
				
				if( (conGOGOBoolOut[rGOInd[i]][destInd] == false)&&
				    (numpGOGABAOutGOGO[rGOInd[i]] < attempt )&&
				    (numpGOGABAInGOGO[rGOInd[i]] < attempt)&&
				    (numpGOGABAOutGOGO[destInd] < attempt )&&
				    (numpGOGABAInGOGO[destInd] < attempt)&&
				    (destInd != rGOInd[i])	
				  ) 
				{	
			
				
					pGOGABAOutGOGO[ rGOInd[i] ][ numpGOGABAOutGOGO[rGOInd[i]] ] = destInd;
					numpGOGABAOutGOGO[ rGOInd[i] ]++;
					pGOGABAOutGOGO[destInd][ numpGOGABAOutGOGO[destInd] ] = rGOInd[i];
					numpGOGABAOutGOGO[destInd]++;
				
					pGOGABAInGOGO[destInd][ numpGOGABAInGOGO[destInd] ] = rGOInd[i];
					numpGOGABAInGOGO[destInd]++;
					pGOGABAInGOGO[ rGOInd[i] ][ numpGOGABAInGOGO[rGOInd[i]] ] = destInd;
					numpGOGABAInGOGO[rGOInd[i]]++;
			
					conGOGOBoolOut[ rGOInd[i] ][destInd] = true;
			
				}
			
			
			}
		
			if(conType == 2){	// variable %reciprocal	
				
				
				if( (conGOGOBoolOut[rGOInd[i]][destInd] == false)&&
				    (numpGOGABAOutGOGO[rGOInd[i]] < attempt )&&
				    (numpGOGABAInGOGO[rGOInd[i]] < attempt)&&
				    (numpGOGABAOutGOGO[destInd] < attempt )&&
				    (numpGOGABAInGOGO[destInd] < attempt)&&
				    (destInd != rGOInd[i])	
				  ) 
				{	
			
				
					pGOGABAOutGOGO[ rGOInd[i] ][ numpGOGABAOutGOGO[rGOInd[i]] ] = destInd;
					numpGOGABAOutGOGO[ rGOInd[i] ]++;
				
					pGOGABAInGOGO[destInd][ numpGOGABAInGOGO[destInd] ] = rGOInd[i];
					numpGOGABAInGOGO[destInd]++;
					
					if(randGen->Random() >= 1-0.05){
						pGOGABAInGOGO[ rGOInd[i] ][ numpGOGABAInGOGO[rGOInd[i]] ] = destInd;
						numpGOGABAInGOGO[rGOInd[i]]++;
						pGOGABAOutGOGO[destInd][ numpGOGABAOutGOGO[destInd] ] = rGOInd[i];
						numpGOGABAOutGOGO[destInd]++;
			
						conGOGOBoolOut[ rGOInd[i] ][destInd] = true;
					}
				}
			
			
			}
		
		
		
		}
	}
		
}	
	

	
	
	ofstream fileGOGOconIn;
	fileGOGOconIn.open("GOGOInputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<30; j++){
			fileGOGOconIn << pGOGABAInGOGO[i][j] << " ";
		}
		fileGOGOconIn << endl;
	}
	
	ofstream fileGOGOconOut;
	fileGOGOconOut.open("GOGOOutputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<30; j++){
			fileGOGOconOut << pGOGABAOutGOGO[i][j] << " ";
		}
		fileGOGOconOut << endl;
	}
	

if(conType == 0){
	
	for(int k=0; k<20; k++)
	{
		cout<<"numpGOGABAInGOGO["<<k<<"]: "<<numpGOGABAInGOGO[k]<<endl;
		for(int j=0; j<numpGOGABAInGOGO[k]; j++)
		{
			cout<<pGOGABAInGOGO[k][j]<<" ";
		}
		cout<<endl;
	}

	}



if(conType == 1){	
	
	int pNonRecip = 0;
	int missedOutCon = 0;
	int missedInCon = 0;

	for(int i=0; i<cp->numGO; i++){

		if(numpGOGABAInGOGO[i] != numpGOGABAOutGOGO[i]){ pNonRecip++; }
		if(numpGOGABAInGOGO[i] != numOutputs) { missedInCon++; }
		if(numpGOGABAOutGOGO[i] != numOutputs) { missedOutCon++; }
	}

	cout << "		" << "Potential non-reciprocal connection:		" << pNonRecip << endl;
	cout << "		" << "Missing Input:					" << missedInCon << endl;
	cout << "		" << "Missing Output:					" << missedOutCon << endl;
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
int numpGOtoGOgj = (spanGOtoGOgjX+1) * (spanGOtoGOgjY+1);


	for(int i=0; i<spanGOtoGOgjX+1;i++){
		int ind = spanGOtoGOgjX - i;
		spanArrayGOtoGOgjX[i] = (spanGOtoGOgjX/2) - ind;}
	for(int i=0; i<spanGOtoGOgjY+1;i++){
		int ind = spanGOtoGOgjY - i;
		spanArrayGOtoGOgjY[i] = (spanGOtoGOgjY/2) - ind;}
		
	for(int i=0; i<numpGOtoGOgj; i++)
	{
		xCoorsGOGOgj[i] = spanArrayGOtoGOgjX[ i%(spanGOtoGOgjX+1) ];
		yCoorsGOGOgj[i] = spanArrayGOtoGOgjY[ i/(spanGOtoGOgjX+1) ];		
		//cout << yCoorsGOGOgj[i] << " ";		
	}
/*	
//Real Measurements
for(int i=0; i<numpGOtoGOgj; i++)
{

	gjPconX = exp( ((abs(xCoorsGOGOgj[i])*48) - 267) / 39.0 );	
	gjPconY = exp( ((abs(yCoorsGOGOgj[i])*48) - 267) / 39.0 );
	gjPcon[i] = (( -1745 + (1836 / (1 + (gjPconX + gjPconY)) ) ) * 0.01);
	
	gjCCX = exp( abs(xCoorsGOGOgj[i]*56) / 70.4 );
	gjCCY = exp( abs(yCoorsGOGOgj[i]*56) / 70.4 );
	gjCC[i] = (-2.3+(29.7 / ( (gjCCX + gjCCY)/2.0) )) * 0.09;
}*/

// "In Vivo additions"
for(int i=0; i<numpGOtoGOgj; i++)
{

	//gjPcon(i) = exp( ((abs(i)*38) - 267) / 39.0 );

	gjPconX = exp( ((abs(xCoorsGOGOgj[i])*36.0) - 267.0) / 39.0 );	
	gjPconY = exp( ((abs(yCoorsGOGOgj[i])*36.0) - 267.0) / 39.0 );
	gjPcon[i] = (( -1745.0 + (1836.0 / (1 + (gjPconX + gjPconY)) ) ) * 0.01);
	
	gjCCX = exp( abs(xCoorsGOGOgj[i]*100.0) / 190.0 );
	gjCCY = exp( abs(yCoorsGOGOgj[i]*100.0) / 190.0 );
	gjCC[i] = (-2.3+(23.0 / ( (gjCCX + gjCCY)/2.0) )) * 0.09;
}

// Remove self connection 
for(int i=0; i<numpGOtoGOgj;i++)
{
	if( (xCoorsGOGOgj[i] == 0) && (yCoorsGOGOgj[i] == 0) )
	{
		gjPcon[i] = 0;
		gjCC[i] = 0;
	}

}

float tempCC;

	for(int i=0; i<cp->numGO; i++)
	{	
		srcPosX = i%cp->goX;
		srcPosY = i/cp->goX;	
		
		for(int j=0; j<numpGOtoGOgj; j++)
		{	
			
			preDestPosX = xCoorsGOGOgj[j]; 
			preDestPosY = yCoorsGOGOgj[j];	

			tempCC = gjCC[j];

			tempDestPosX = srcPosX + preDestPosX;
			tempDestPosY = srcPosY + preDestPosY;

			tempDestPosX = (tempDestPosX%cp->goX+cp->goX)%cp->goX;
			tempDestPosY = (tempDestPosY%cp->goY+cp->goY)%cp->goY;
					
			destInd = tempDestPosY*cp->goX+tempDestPosX;


			//TODO: add CC to connectivtiy array

			if( (randGen->Random()>=1-gjPcon[j]) && (gjConBool[i][destInd ] == false) && (gjConBool[destInd][i] == false))
			{	
				

				pGOCoupInGOGO[destInd][ numpGOCoupInGOGO[destInd] ] = i;
				pGOCoupInGOGOCCoeff[destInd][ numpGOCoupInGOGO[destInd] ] = tempCC;
				numpGOCoupInGOGO[destInd]++;
					
				pGOCoupInGOGO[i][ numpGOCoupInGOGO[i] ] = destInd;
				pGOCoupInGOGOCCoeff[i][ numpGOCoupInGOGO[i] ] = tempCC;
				numpGOCoupInGOGO[i]++;

				gjConBool[i][destInd] = true;
				
			
			}
		}
	}
/*
	ofstream fileGOGOGJcon;
	fileGOGOGJcon.open("GOGOGJInputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<numpGOtoGOgj; j++){
			fileGOGOGJcon << pGOCoupInGOGO[i][j] << " ";
		}
		fileGOGOGJcon << endl;
	}
	
	ofstream fileGOGOCC;
	fileGOGOCC.open("GOGOCCInputcon.txt");
	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<numpGOtoGOgj; j++){
			fileGOGOCC << pGOCoupInGOGOCCoeff[i][j] << " ";
		}
		fileGOGOCC << endl;
	}
*/

}



















void InNetConnectivityState::connectPFtoBC()
{

int spanPFtoBCX = cp->grX;
int spanPFtoBCY = cp->grY/cp->numBC;
int numpPFtoBC = (spanPFtoBCX+1)*(spanPFtoBCY+1);


	for(int i=0; i<spanPFtoBCX + 1; i++){
		int ind = spanPFtoBCX - i;
		spanArrayPFtoBCX[i] = (spanPFtoBCX/2) - ind;}
	for(int i=0; i<spanPFtoBCY + 1; i++){
		int ind = spanPFtoBCY - i;
		spanArrayPFtoBCY[i] = (spanPFtoBCY/2) - ind;}
	
	for(int i=0; i<numpPFtoBC; i++){
		xCoorsPFBC[i] = spanArrayPFtoBCX[ i%(spanPFtoBCX+1) ];
		yCoorsPFBC[i] = spanArrayPFtoBCY[ i/(spanPFtoBCX+1) ];
	}


//Random Span Array
	vector<int> rPFBCSpanInd;
	rPFBCSpanInd.assign(numpPFtoBC,0);
	for(int ind=0; ind<numpPFtoBC; ind++){ rPFBCSpanInd[ind]=ind; }

	float gridXScaleSrctoDest = 1;
	float gridYScaleSrctoDest = (float)cp->numBC / cp->grY;

	int srcPosX;
	int srcPosY;
	int preDestPosX;
	int preDestPosY;
	int tempDestPosX;
	int tempDestPosY;
	int destInd;

	for(int i=0; i<cp->numBC; i++)
	{
		srcPosX = cp->grX/2;
		srcPosY = i;
		random_shuffle(rPFBCSpanInd.begin(), rPFBCSpanInd.end());
		
		for(int j=0; j<5000; j++)
		{
			preDestPosX = xCoorsPFBC[ rPFBCSpanInd[j] ];
			preDestPosY = yCoorsPFBC[ rPFBCSpanInd[j] ];
			
			tempDestPosX = (int)round(srcPosX/gridXScaleSrctoDest) + preDestPosX;
			tempDestPosY = (int)round(srcPosY/gridYScaleSrctoDest) + preDestPosY;
			
			tempDestPosX = (tempDestPosX%cp->grX+cp->grX)%cp->grX;
			tempDestPosY = (tempDestPosY%cp->grY+cp->grY)%cp->grY;
			

			destInd = tempDestPosY*cp->grX+tempDestPosX;	

			pBCfromPFtoBC[i][numpBCfromPFtoBC[i]] = destInd;
			numpBCfromPFtoBC[i]++;

			pGRfromPFtoBC[destInd][numpGRfromPFtoBC[destInd]] = i;
			numpGRfromPFtoBC[destInd]++;
		
		}

	}



for(int i=0; i<2000; i++)
{
	for(int j=0; j<numpGRfromPFtoBC[i]; j++)
	{
		//cout << pGRfromPFtoBC[i][j] << " ";
	}
//	cout << endl;
}




}





void InNetConnectivityState::assignPFtoBCDelays(unsigned int msPerStep)
{
	
	for(int i=0; i<cp->numGR; i++)
	{
		int grPosX;
//		int grBCPCSCDist;

		//calculate x coordinate of GR position
		grPosX=i%cp->grX;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
//		grBCPCSCDist=abs((int)(cp->grX/2-grPosX));
//		pGRDelayMaskfromGRtoBSP[i]=0x1<<
//				(int)((grBCPCSCDist/cp->grPFVelInGRXPerTStep+
//						cp->grAFDelayInTStep)/msPerStep);

		for(int j=0; j<numpGRfromPFtoBC[i]; j++)
		{
			int dfromGRtoBC;
			int bcPosX;

			//bcPosX=(pGRfromPFtoBC[i][j]%cp->goX)*(((float)cp->grX)/cp->goX);
			bcPosX = cp->grX/2; 


			dfromGRtoBC=abs(bcPosX-grPosX);

			/*if(dfromGRtoBC > cp->grX/2)
			{
				if(bcPosX<grPosX)
				{
					dfromGRtoBC=bcPosX+cp->grX-grPosX;
				}
				else
				{
					dfromGRtoBC=grPosX+cp->grX-bcPosX;
				}
			}*/

			//pGRDelayMaskfromGRtoBC[i][j]=0x1<<
			//		(int)((dfromGRtoBC/cp->grPFVelInGRXPerTStep+
			//				cp->grAFDelayInTStep)/msPerStep);
		}
	}

	/*for(int i=0; i<500; i++)
	{
		cout<<"numpGRfromPFtoBC["<<i<<"]: "<<numpGRfromPFtoBC[i]<<endl;
		for(int j=0; j<numpGRfromPFtoBC[i]; j++)
		{
			cout<<pGRDelayMaskfromGRtoBC[i][j]<<" ";
		}
		cout<<endl;
	}*/


}


void InNetConnectivityState::assignGRDelays(unsigned int msPerStep)
{
	
	for(int i=0; i<cp->numGR; i++)
	{
		int grPosX;
		int grBCPCSCDist;

		//calculate x coordinate of GR position
		grPosX=i%cp->grX;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		grBCPCSCDist=abs((int)(cp->grX/2-grPosX));
		pGRDelayMaskfromGRtoBSP[i]=0x1<<
				(int)((grBCPCSCDist/cp->grPFVelInGRXPerTStep+
						cp->grAFDelayInTStep)/msPerStep);

		for(int j=0; j<numpGRfromGRtoGO[i]; j++)
		{
			int dfromGRtoGO;
			int goPosX;

			goPosX=(pGRfromGRtoGO[i][j]%cp->goX)*(((float)cp->grX)/cp->goX);

			dfromGRtoGO=abs(goPosX-grPosX);

			if(dfromGRtoGO > cp->grX/2)
			{
				if(goPosX<grPosX)
				{
					dfromGRtoGO=goPosX+cp->grX-grPosX;
				}
				else
				{
					dfromGRtoGO=grPosX+cp->grX-goPosX;
				}
			}

			pGRDelayMaskfromGRtoGO[i][j]=0x1<<
					(int)((dfromGRtoGO/cp->grPFVelInGRXPerTStep+
							cp->grAFDelayInTStep)/msPerStep);
		}
	}
/*
	for(int i=0; i<10; i++)
	{
		cout<<"numpGRfromGRtoGO["<<i<<"]: "<<numpGRfromGRtoGO[i]<<endl;
		for(int j=0; j<numpGRfromGRtoGO[i]; j++)
		{
			cout<<pGRDelayMaskfromGRtoGO[i][j]<<" ";
		}
		cout<<endl;
	}
*/

}

/*
void InNetConnectivityState::connectCommon(ct_uint32_t **srcConArr, ct_int32_t *srcNumCon,
		ct_uint32_t **destConArr, ct_int32_t *destNumCon,
		ct_uint32_t srcMaxNumCon, ct_uint32_t numSrcCells,
		ct_uint32_t destMaxNumCon, ct_uint32_t destNormNumCon,
		ct_uint32_t srcGridX, ct_uint32_t srcGridY, ct_uint32_t destGridX, ct_uint32_t destGridY,
		ct_uint32_t srcSpanOnDestGridX, ct_uint32_t srcSpanOnDestGridY,
		ct_uint32_t normConAttempts, ct_uint32_t maxConAttempts, bool needUnique,
		CRandomSFMT *randGen)
{
	bool *srcConnected;
	float gridXScaleStoD;
	float gridYScaleStoD;

	gridXScaleStoD=((float)srcGridX)/((float)destGridX);
	gridYScaleStoD=((float)srcGridY)/((float)destGridY);

	srcConnected=new bool[numSrcCells];

	cout<<"srcMaxNumCon "<<srcMaxNumCon<<" numSrcCells "<<numSrcCells<<endl;
	cout<<"destMaxNumCon "<<destMaxNumCon<<" destNormNumCon "<<destNormNumCon<<endl;
	cout<<"srcGridX "<<srcGridX<<" srcGridY "<<srcGridY<<" destGridX "<<destGridX<<" destGridY "<<destGridY<<endl;
	cout<<"srcSpanOnDestGridX "<<srcSpanOnDestGridX<<" srcSpanOnDestGridY "<<srcSpanOnDestGridY<<endl;
	cout<<"gridXScaleStoD "<<gridXScaleStoD<<" gridYScaleStoD "<<gridYScaleStoD<<endl;

	for(int i=0; i<srcMaxNumCon; i++)
	{
		int srcNumConnected;

		memset(srcConnected, false, numSrcCells*sizeof(bool));
		srcNumConnected=0;

//		cout<<"i "<<i<<endl;

		while(srcNumConnected<numSrcCells)
		{
			int srcInd;
			int srcPosX;
			int srcPosY;
			int attempts;
			int tempDestNumConLim;
			bool complete;

			srcInd=randGen->IRandom(0, numSrcCells-1);

			if(srcConnected[srcInd])
			{
				continue;
			}
//			cout<<"i "<<i<<" srcInd "<<srcInd<<" srcNumConnected "<<srcNumConnected<<endl;
			srcConnected[srcInd]=true;
			srcNumConnected++;

			srcPosX=srcInd%srcGridX;
			srcPosY=(int)(srcInd/srcGridX);

			tempDestNumConLim=destNormNumCon;

			for(attempts=0; attempts<maxConAttempts; attempts++)
			{
				int tempDestPosX;
				int tempDestPosY;
				int derivedDestInd;

				if(attempts==normConAttempts)
				{
					tempDestNumConLim=destMaxNumCon;
				}

				tempDestPosX=(int)round(srcPosX/gridXScaleStoD);
				tempDestPosY=(int)round(srcPosY/gridXScaleStoD);
//				cout<<"before rand: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				tempDestPosX+=round((randGen->Random()-0.5)*srcSpanOnDestGridX);//randGen->IRandom(-srcSpanOnDestGridX/2, srcSpanOnDestGridX/2);
				tempDestPosY+=round((randGen->Random()-0.5)*srcSpanOnDestGridY);//.randGen->IRandom(-srcSpanOnDestGridY/2, srcSpanOnDestGridY/2);
//				cout<<"after  rand: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				tempDestPosX=((tempDestPosX%destGridX+destGridX)%destGridX);
				tempDestPosY=((tempDestPosY%destGridY+destGridY)%destGridY);
//				cout<<"after mod: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				derivedDestInd=tempDestPosY*destGridX+tempDestPosX;
//				cout<<"derivedDestInd "<<derivedDestInd<<endl;

				if(needUnique)
				{
					bool unique=true;

					for(int j=0; j<i; j++)
					{
						if(derivedDestInd==srcConArr[srcInd][j])
						{
							unique=false;
							break;
						}
					}
					if(!unique)
					{
						continue;
					}
				}

				if(destNumCon[derivedDestInd]<tempDestNumConLim)
				{
					destConArr[derivedDestInd][destNumCon[derivedDestInd]]=srcInd;
					destNumCon[derivedDestInd]++;
					srcConArr[srcInd][i]=derivedDestInd;
					srcNumCon[srcInd]++;

					break;
				}
			}
			if(attempts==maxConAttempts)
			{
//				cout<<"incomplete connection for cell #"<<srcInd<<endl;
			}
		}
	}

	delete[] srcConnected;
}

void InNetConnectivityState::translateCommon(ct_uint32_t **pPreGLConArr, ct_int32_t *numpPreGLCon,
		ct_uint32_t **pGLPostGLConArr, ct_int32_t *numpGLPostGLCon,
		ct_uint32_t **pPreConArr, ct_int32_t *numpPreCon,
		ct_uint32_t **pPostConArr, ct_int32_t *numpPostCon,
		ct_uint32_t numPre)
{
//	cout<<"numPre "<<endl;
	for(int i=0; i<numPre; i++)
	{
		numpPreCon[i]=0;

		for(int j=0; j<numpPreGLCon[i]; j++)
		{
			ct_uint32_t glInd;

			glInd=pPreGLConArr[i][j];

			for(int k=0; k<numpGLPostGLCon[glInd]; k++)
			{
				ct_uint32_t postInd;

				postInd=pGLPostGLConArr[glInd][k];
//				cout<<"i "<<i<<" j "<<j<<" k "<<k<<" numpPreCon "<<numpPreCon[i]<<" glInd "<<glInd<<" postInd "<<postInd;
//				cout.flush();

				pPreConArr[i][numpPreCon[i]]=postInd;
				numpPreCon[i]++;

				pPostConArr[postInd][numpPostCon[postInd]]=i;
				numpPostCon[postInd]++;

//				cout<<" "<<numpGLPostGLCon[glInd]<<" "<<numpPreGLCon[i]<<" "<<numPre<<" "<<" done "<<endl;
			}
//			cout<<"k done"<<endl;
		}
	}
//	cout<<"i done"<<endl;
}
*/








void InNetConnectivityState::connectCommon(int **srcConArr, int *srcNumCon,
		int **destConArr, int *destNumCon,
		int srcMaxNumCon, int numSrcCells,
		int destMaxNumCon, int destNormNumCon,
		int srcGridX, int srcGridY, int destGridX, int destGridY,
		int srcSpanOnDestGridX, int srcSpanOnDestGridY,
		int normConAttempts, int maxConAttempts, bool needUnique,
		CRandomSFMT *randGen)
{
	bool *srcConnected;
	float gridXScaleStoD;
	float gridYScaleStoD;

	gridXScaleStoD=((float)srcGridX)/((float)destGridX);
	gridYScaleStoD=((float)srcGridY)/((float)destGridY);

	srcConnected=new bool[numSrcCells];

	cout<<"srcMaxNumCon "<<srcMaxNumCon<<" numSrcCells "<<numSrcCells<<endl;
	cout<<"destMaxNumCon "<<destMaxNumCon<<" destNormNumCon "<<destNormNumCon<<endl;
	cout<<"srcGridX "<<srcGridX<<" srcGridY "<<srcGridY<<" destGridX "<<destGridX<<" destGridY "<<destGridY<<endl;
	cout<<"srcSpanOnDestGridX "<<srcSpanOnDestGridX<<" srcSpanOnDestGridY "<<srcSpanOnDestGridY<<endl;
	cout<<"gridXScaleStoD "<<gridXScaleStoD<<" gridYScaleStoD "<<gridYScaleStoD<<endl;

	for(int i=0; i<srcMaxNumCon; i++)
	{
		int srcNumConnected;

		memset(srcConnected, false, numSrcCells*sizeof(bool));
		srcNumConnected=0;

		cout<<"i "<<i<<endl;

		while(srcNumConnected<numSrcCells)
		{
			int srcInd;
			int srcPosX;
			int srcPosY;
			int attempts;
			int tempDestNumConLim;
			bool complete;

			srcInd=randGen->IRandom(0, numSrcCells-1);

			if(srcConnected[srcInd])
			{
				continue;
			}
//			cout<<"i "<<i<<" srcInd "<<srcInd<<" srcNumConnected "<<srcNumConnected<<endl;
			srcConnected[srcInd]=true;
			srcNumConnected++;

			srcPosX=srcInd%srcGridX;
			srcPosY=(int)(srcInd/srcGridX);

			tempDestNumConLim=destNormNumCon;

			for(attempts=0; attempts<maxConAttempts; attempts++)
			{
				int tempDestPosX;
				int tempDestPosY;
				int derivedDestInd;

				if(attempts==normConAttempts)
				{
					tempDestNumConLim=destMaxNumCon;
				}

				tempDestPosX=(int)round(srcPosX/gridXScaleStoD);
				tempDestPosY=(int)round(srcPosY/gridXScaleStoD);
//				cout<<"before rand: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				tempDestPosX+=round((randGen->Random()-0.5)*srcSpanOnDestGridX);//randGen->IRandom(-srcSpanOnDestGridX/2, srcSpanOnDestGridX/2);
				tempDestPosY+=round((randGen->Random()-0.5)*srcSpanOnDestGridY);//.randGen->IRandom(-srcSpanOnDestGridY/2, srcSpanOnDestGridY/2);
//				cout<<"after  rand: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				tempDestPosX=((tempDestPosX%destGridX+destGridX)%destGridX);
				tempDestPosY=((tempDestPosY%destGridY+destGridY)%destGridY);
//				cout<<"after mod: tempDestPosX "<<tempDestPosX<<" tempDestPosY "<<tempDestPosY<<endl;

				derivedDestInd=tempDestPosY*destGridX+tempDestPosX;
//				cout<<"derivedDestInd "<<derivedDestInd<<endl;

				if(needUnique)
				{
					bool unique=true;

					for(int j=0; j<i; j++)
					{
						if(derivedDestInd==srcConArr[srcInd][j])
						{
							unique=false;
							break;
						}
					}
					if(!unique)
					{
						continue;
					}
				}

				if(destNumCon[derivedDestInd]<tempDestNumConLim)
				{
					destConArr[derivedDestInd][destNumCon[derivedDestInd]]=srcInd;
					destNumCon[derivedDestInd]++;
					srcConArr[srcInd][i]=derivedDestInd;
					srcNumCon[srcInd]++;

					break;
				}
			}
			if(attempts==maxConAttempts)
			{
//				cout<<"incomplete connection for cell #"<<srcInd<<endl;
			}
		}
	}

	delete[] srcConnected;
}


void InNetConnectivityState::translateCommon(int **pPreGLConArr, int *numpPreGLCon,
		int **pGLPostGLConArr, int *numpGLPostGLCon,
		int **pPreConArr, int *numpPreCon,
		int **pPostConArr, int *numpPostCon,
		int numPre)
{
//	cout<<"numPre "<<endl;
	for(int i=0; i<numPre; i++)
	{
		numpPreCon[i]=0;

		for(int j=0; j<numpPreGLCon[i]; j++)
		{
			int glInd;

			glInd=pPreGLConArr[i][j];

			for(int k=0; k<numpGLPostGLCon[glInd]; k++)
			{
				int postInd;

				postInd=pGLPostGLConArr[glInd][k];
//				cout<<"i "<<i<<" j "<<j<<" k "<<k<<" numpPreCon "<<numpPreCon[i]<<" glInd "<<glInd<<" postInd "<<postInd;
//				cout.flush();

				pPreConArr[i][numpPreCon[i]]=postInd;
				numpPreCon[i]++;

				pPostConArr[postInd][numpPostCon[postInd]]=i;
				numpPostCon[postInd]++;

//				cout<<" "<<numpGLPostGLCon[glInd]<<" "<<numpPreGLCon[i]<<" "<<numPre<<" "<<" done "<<endl;
			}
//			cout<<"k done"<<endl;
		}
	}
//	cout<<"i done"<<endl;
}


