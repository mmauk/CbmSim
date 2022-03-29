/*
 * mzoneconnectivitystate.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: consciousness
 */

#include "state/mzoneconnectivitystate.h"
#include <vector>
#include <algorithm>

MZoneConnectivityState::MZoneConnectivityState(ConnectivityParams *parameters, int randSeed)
{
	cp = parameters;

	allocateMemory();
	initalizeVars();

	connectBCtoPC();
	connectPCtoBC();
	connectSCtoPC();
	connectPCtoNC(randSeed);
	connectNCtoIO();
	connectMFtoNC();
	connectIOtoPC();
	connectIOtoIO();
}

MZoneConnectivityState::MZoneConnectivityState(ConnectivityParams *parameters, std::fstream &infile)
{
	cp = parameters;

	allocateMemory();

	stateRW(true, infile);
}

MZoneConnectivityState::MZoneConnectivityState(const MZoneConnectivityState &state)
{
	cp = state.cp;

	allocateMemory();

	arrayCopy<ct_uint32_t>(pBCfromBCtoPC[0], state.pBCfromBCtoPC[0],
			cp->numBC * cp->numpBCfromBCtoPC);
	arrayCopy<ct_uint32_t>(pBCfromPCtoBC[0], state.pBCfromPCtoBC[0],
			cp->numBC * cp->numpBCfromPCtoBC);
	arrayCopy<ct_uint32_t>(pSCfromSCtoPC[0], state.pSCfromSCtoPC[0],
			cp->numSC * cp->numpSCfromSCtoPC);
	arrayCopy<ct_uint32_t>(pPCfromBCtoPC[0], state.pPCfromBCtoPC[0],
			cp->numPC * cp->numpPCfromBCtoPC);
	arrayCopy<ct_uint32_t>(pPCfromPCtoBC[0], state.pPCfromPCtoBC[0],
			cp->numPC * cp->numpPCfromPCtoBC);
	arrayCopy<ct_uint32_t>(pPCfromSCtoPC[0], state.pPCfromSCtoPC[0],
			cp->numPC * cp->numpPCfromSCtoPC);
	arrayCopy<ct_uint32_t>(pPCfromPCtoNC[0], state.pPCfromPCtoNC[0],
			cp->numPC * cp->numpPCfromPCtoNC);
	arrayCopy<ct_uint32_t>(pPCfromIOtoPC, state.pPCfromIOtoPC, cp->numPC);

	arrayCopy<ct_uint32_t>(pNCfromPCtoNC[0], state.pNCfromPCtoNC[0],
			cp->numNC * cp->numpNCfromPCtoNC);
	arrayCopy<ct_uint32_t>(pNCfromNCtoIO[0], state.pNCfromNCtoIO[0],
			cp->numNC * cp->numpNCfromNCtoIO);
	arrayCopy<ct_uint32_t>(pNCfromMFtoNC[0], state.pNCfromMFtoNC[0],
			cp->numNC * cp->numpNCfromMFtoNC);

	arrayCopy<ct_uint32_t>(pIOfromIOtoPC[0], state.pIOfromIOtoPC[0],
			cp->numIO * cp->numpIOfromIOtoPC);
	arrayCopy<ct_uint32_t>(pIOfromNCtoIO[0], state.pIOfromNCtoIO[0],
			cp->numIO * cp->numpIOfromNCtoIO);
	arrayCopy<ct_uint32_t>(pIOInIOIO[0], state.pIOInIOIO[0],
			cp->numIO * cp->numpIOInIOIO);
	arrayCopy<ct_uint32_t>(pIOOutIOIO[0], state.pIOOutIOIO[0],
			cp->numIO * cp->numpIOOutIOIO);
}

MZoneConnectivityState::~MZoneConnectivityState()
{
	delete2DArray<ct_uint32_t>(pBCfromBCtoPC);
	delete2DArray<ct_uint32_t>(pBCfromPCtoBC);

	delete2DArray<ct_uint32_t>(pSCfromSCtoPC);

	delete2DArray<ct_uint32_t>(pPCfromBCtoPC);
	delete2DArray<ct_uint32_t>(pPCfromPCtoBC);
	delete2DArray<ct_uint32_t>(pPCfromSCtoPC);
	delete2DArray<ct_uint32_t>(pPCfromPCtoNC);
	delete[] pPCfromIOtoPC;

	delete2DArray<ct_uint32_t>(pNCfromPCtoNC);
	delete2DArray<ct_uint32_t>(pNCfromNCtoIO);
	delete2DArray<ct_uint32_t>(pNCfromMFtoNC);

	delete2DArray<ct_uint32_t>(pIOfromIOtoPC);
	delete2DArray<ct_uint32_t>(pIOfromNCtoIO);
	delete2DArray<ct_uint32_t>(pIOInIOIO);
	delete2DArray<ct_uint32_t>(pIOOutIOIO);
}

void MZoneConnectivityState::writeState(std::fstream &outfile)
{
	stateRW(false, (std::fstream &)outfile);
}

void MZoneConnectivityState::allocateMemory()
{
		
	pBCfromBCtoPC = allocate2DArray<ct_uint32_t>(cp->numBC, cp->numpBCfromBCtoPC);
	pBCfromPCtoBC = allocate2DArray<ct_uint32_t>(cp->numBC, cp->numpBCfromPCtoBC);

	pSCfromSCtoPC = allocate2DArray<ct_uint32_t>(cp->numSC, cp->numpSCfromSCtoPC);

	pPCfromBCtoPC = allocate2DArray<ct_uint32_t>(cp->numPC, cp->numpPCfromBCtoPC);
	pPCfromPCtoBC = allocate2DArray<ct_uint32_t>(cp->numPC, cp->numpPCfromPCtoBC);
	pPCfromSCtoPC = allocate2DArray<ct_uint32_t>(cp->numPC, cp->numpPCfromSCtoPC);
	pPCfromPCtoNC = allocate2DArray<ct_uint32_t>(cp->numPC, cp->numpPCfromPCtoNC);
	pPCfromIOtoPC = new ct_uint32_t[cp->numPC];

	pNCfromPCtoNC = allocate2DArray<ct_uint32_t>(cp->numNC, cp->numpNCfromPCtoNC);
	pNCfromNCtoIO = allocate2DArray<ct_uint32_t>(cp->numNC, cp->numpNCfromNCtoIO);
	pNCfromMFtoNC = allocate2DArray<ct_uint32_t>(cp->numNC, cp->numpNCfromMFtoNC);

	pIOfromIOtoPC = allocate2DArray<ct_uint32_t>(cp->numIO, cp->numpIOfromIOtoPC);
	pIOfromNCtoIO = allocate2DArray<ct_uint32_t>(cp->numIO, cp->numpIOfromNCtoIO);
	pIOInIOIO 	  = allocate2DArray<ct_uint32_t>(cp->numIO, cp->numpIOInIOIO);
	pIOOutIOIO	  = allocate2DArray<ct_uint32_t>(cp->numIO, cp->numpIOOutIOIO);
}

void MZoneConnectivityState::stateRW(bool read, std::fstream &file)
{
	rawBytesRW((char *)pBCfromBCtoPC[0], cp->numBC * cp->numpBCfromBCtoPC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pBCfromPCtoBC[0], cp->numBC * cp->numpBCfromPCtoBC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pSCfromSCtoPC[0], cp->numSC * cp->numpSCfromSCtoPC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pPCfromBCtoPC[0], cp->numPC * cp->numpPCfromBCtoPC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromPCtoBC[0], cp->numPC * cp->numpPCfromPCtoBC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromSCtoPC[0], cp->numPC * cp->numpPCfromSCtoPC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromPCtoNC[0], cp->numPC * cp->numpPCfromPCtoNC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromIOtoPC, cp->numPC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pNCfromPCtoNC[0], cp->numNC * cp->numpNCfromPCtoNC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pNCfromNCtoIO[0], cp->numNC * cp->numpNCfromNCtoIO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pNCfromMFtoNC[0], cp->numNC * cp->numpNCfromMFtoNC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pIOfromIOtoPC[0], cp->numIO * cp->numpIOfromIOtoPC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOfromNCtoIO[0], cp->numIO * cp->numpIOfromNCtoIO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOInIOIO[0], cp->numIO * cp->numpIOInIOIO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOOutIOIO[0], cp->numIO * cp->numpIOOutIOIO * sizeof(ct_uint32_t), read, file);
}

bool MZoneConnectivityState::operator==(const MZoneConnectivityState &compState)
{
	bool eq = true;

	for (int i = 0; i < cp->numBC; i++)
	{
		for (int j = 0; j < cp->numpBCfromPCtoBC; j++)
		{
			eq = eq && (pBCfromPCtoBC[i][j] == compState.pBCfromPCtoBC[i][j]);
		}
	}

	for (int i = 0; i < cp->numIO; i++)
	{
		for (int j = 0; j < cp->numpIOInIOIO; j++)
		{
			eq = eq && (pIOInIOIO[i][j] == compState.pIOInIOIO[i][j]);
		}
	}
	return eq;
}

bool MZoneConnectivityState::operator!=(const MZoneConnectivityState &compState)
{
	return !(*this == compState);
}

void MZoneConnectivityState::initalizeVars()
{
	arrayInitialize<ct_uint32_t>(pBCfromBCtoPC[0], UINT_MAX, cp->numBC*cp->numpBCfromBCtoPC);
	arrayInitialize<ct_uint32_t>(pBCfromPCtoBC[0], UINT_MAX, cp->numBC*cp->numpBCfromPCtoBC);

	arrayInitialize<ct_uint32_t>(pSCfromSCtoPC[0], UINT_MAX, cp->numSC*cp->numpSCfromSCtoPC);

	arrayInitialize<ct_uint32_t>(pPCfromBCtoPC[0], UINT_MAX, cp->numPC*cp->numpPCfromBCtoPC);
	arrayInitialize<ct_uint32_t>(pPCfromPCtoBC[0], UINT_MAX, cp->numPC*cp->numpPCfromPCtoBC);
	arrayInitialize<ct_uint32_t>(pPCfromSCtoPC[0], UINT_MAX, cp->numPC*cp->numpPCfromSCtoPC);
	arrayInitialize<ct_uint32_t>(pPCfromPCtoNC[0], UINT_MAX, cp->numPC*cp->numpPCfromPCtoNC);
	arrayInitialize<ct_uint32_t>(pPCfromIOtoPC, UINT_MAX, cp->numPC);

	arrayInitialize<ct_uint32_t>(pNCfromPCtoNC[0], UINT_MAX, cp->numNC*cp->numpNCfromPCtoNC);
	arrayInitialize<ct_uint32_t>(pNCfromNCtoIO[0], UINT_MAX, cp->numNC*cp->numpNCfromNCtoIO);
	arrayInitialize<ct_uint32_t>(pNCfromMFtoNC[0], UINT_MAX, cp->numNC*cp->numpNCfromMFtoNC);

	arrayInitialize<ct_uint32_t>(pIOfromIOtoPC[0], UINT_MAX, cp->numIO*cp->numpIOfromIOtoPC);
	arrayInitialize<ct_uint32_t>(pIOfromNCtoIO[0], UINT_MAX, cp->numIO*cp->numpIOfromNCtoIO);
	arrayInitialize<ct_uint32_t>(pIOInIOIO[0], UINT_MAX, cp->numIO*cp->numpIOInIOIO);
	arrayInitialize<ct_uint32_t>(pIOOutIOIO[0], UINT_MAX, cp->numIO*cp->numpIOOutIOIO);
}

void MZoneConnectivityState::connectBCtoPC()
{
	int bcToPCRatio = cp->numBC / cp->numPC;

	for (int i = 0; i < cp->numPC; i++)
	{
		pBCfromBCtoPC[i * bcToPCRatio][0] = ((i + 1) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio][1] = ((i - 1) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio][2] = ((i + 2) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio][3] = ((i - 2) % cp->numPC + cp->numPC) % cp->numPC;

		pBCfromBCtoPC[i * bcToPCRatio+1][0] = ((i + 1) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+1][1] = ((i - 1) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+1][2] = ((i + 3) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+1][3] = ((i - 3) % cp->numPC + cp->numPC) % cp->numPC;

		pBCfromBCtoPC[i * bcToPCRatio+2][0] = ((i + 3) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+2][1] = ((i - 3) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+2][2] = ((i + 6) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+2][3] = ((i - 6) % cp->numPC + cp->numPC) % cp->numPC;

		pBCfromBCtoPC[i * bcToPCRatio+3][0] = ((i + 4) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+3][1] = ((i - 4) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+3][2] = ((i + 9) % cp->numPC + cp->numPC) % cp->numPC;
		pBCfromBCtoPC[i * bcToPCRatio+3][3] = ((i - 9) % cp->numPC + cp->numPC) % cp->numPC;
	}
}

void MZoneConnectivityState::connectPCtoBC()
{
	int bcToPCRatio = cp->numBC / cp->numPC;

	for (int i = 0; i < cp->numPC; i++)
	{
		for (int j = 0; j < cp->numpPCfromPCtoBC; j++)
		{
			int bcInd = i * bcToPCRatio - 6 + j;
			pPCfromPCtoBC[i][j] = (bcInd % cp->numBC + cp->numBC) % cp->numBC;
		}
	}
}

void MZoneConnectivityState::connectSCtoPC()
{
	for (int i = 0; i < cp->numSC; i++)
	{
		for (int j = 0; j < cp->numpSCfromSCtoPC; j++)
		{
			int pcInd = i / cp->numpPCfromSCtoPC;
			pSCfromSCtoPC[i][j] = pcInd;
			pPCfromSCtoPC[pcInd][i % cp->numpPCfromSCtoPC] = i;
		}
	}
}

void MZoneConnectivityState::connectPCtoNC(int randSeed)
{
	// NOTE: this was heap-allocated before. why???
	// FIXME: This is not working properly: '{1}' does not
	// expand to all 1's
	int pcNumConnected[cp->numPC];

	for (int i = 0; i < cp->numPC; i++)
	{
		pcNumConnected[i] = 1;
	}	

	CRandomSFMT0 randGen(randSeed);

	int countPCNC;
	int pcInd;	
	
	for (int i = 0; i < cp->numNC; i++)
	{
		for (int j = 0; j < cp->numPC / cp->numNC; j++)
		{
			pcInd = i * (cp->numPC / cp->numNC) + j;
			pNCfromPCtoNC[i][j] = pcInd;
			pPCfromPCtoNC[pcInd][0] = i;
		}
	}

	for (int i = 0; i < cp->numNC - 1; i++)
	{
		for (int j = cp->numpNCfromPCtoNC / 3; j<cp->numpNCfromPCtoNC; j++)
		{
			countPCNC = 0;
			
			while(true)
			{
				bool connect = true;

				pcInd = randGen.IRandomX(0, cp->numPC - 1);

				if (pcNumConnected[pcInd] >= cp->numpPCfromPCtoNC) continue;
				for (int k = 0; k < pcNumConnected[pcInd]; k++)
				{
					if (pPCfromPCtoNC[pcInd][k] == i)
					{
						connect = false;
						break;
					}
				}
				if (connect || countPCNC > 100) break;
				countPCNC++;
			}

			pNCfromPCtoNC[i][j] = pcInd;
			pPCfromPCtoNC[pcInd][pcNumConnected[pcInd]] = i;

			pcNumConnected[pcInd]++;
		}
	}

	// static cast?	
	unsigned int numSyn = cp->numPC / cp->numNC;

	for (int h = 1; h < cp->numpPCfromPCtoNC; h++)
	{
		for(int i = 0; i < cp->numPC; i++)
		{
			if (pcNumConnected[i] < cp->numpPCfromPCtoNC)
			{
				pNCfromPCtoNC[cp->numNC - 1][numSyn] = i;
				pPCfromPCtoNC[i][pcNumConnected[i]] = cp->numNC - 1;

				pcNumConnected[i]++;
				numSyn++;
			}
		}
	}
}

void MZoneConnectivityState::connectNCtoIO()
{
	for (int i = 0; i < cp->numIO; i++)
	{
		for (int j = 0; j < cp->numpIOfromNCtoIO; j++)
		{
			pIOfromNCtoIO[i][j] = j;
			pNCfromNCtoIO[j][i] = i;
		}
	}
}

void MZoneConnectivityState::connectMFtoNC()
{
	for (int i = 0; i<cp->numMF; i++)
	{
		for (int j = 0; j < cp->numpMFfromMFtoNC; j++)
		{
			pNCfromMFtoNC[i / cp->numpNCfromMFtoNC][i % cp->numpNCfromMFtoNC] = i;
		}
	}
}

void MZoneConnectivityState::connectIOtoPC()
{
	for (int i = 0; i < cp->numIO; i++)
	{
		for (int j = 0; j < cp->numpIOfromIOtoPC; j++)
		{
			int pcInd = i * cp->numpIOfromIOtoPC + j;

			pIOfromIOtoPC[i][j]  = pcInd;
			pPCfromIOtoPC[pcInd] = i;
		}
	}
}

void MZoneConnectivityState::connectIOtoIO()
{
	for (int i = 0; i < cp->numIO; i++)
	{
		int inInd = 0;
		for (int j = 0; j < cp->numpIOInIOIO; j++)
		{
			if (inInd == i) inInd++;
			pIOInIOIO[i][j] = inInd;
			inInd++;
		}
	}
}

