/*
 * mzoneconnectivitystate.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: consciousness
 */

#include "state/mzoneconnectivitystate.h"

MZoneConnectivityState::MZoneConnectivityState(ConnectivityParams &cp, int randSeed)
{
	initializeVars(cp);

	connectBCtoPC(cp);
	connectPCtoBC(cp);
	connectSCtoPC(cp);
	connectPCtoNC(cp, randSeed);
	connectNCtoIO(cp);
	connectMFtoNC(cp);
	connectIOtoPC(cp);
	connectIOtoIO(cp);
}

MZoneConnectivityState::MZoneConnectivityState(ConnectivityParams &cp, std::fstream &infile)
{
	stateRW(cp, true, infile);
}

//MZoneConnectivityState::MZoneConnectivityState(const MZoneConnectivityState &state)
//{
//	arrayCopy<ct_uint32_t>(pBCfromBCtoPC[0], state.pBCfromBCtoPC[0],
//			cp->numBC * cp->numpBCfromBCtoPC);
//	arrayCopy<ct_uint32_t>(pBCfromPCtoBC[0], state.pBCfromPCtoBC[0],
//			cp->numBC * cp->numpBCfromPCtoBC);
//	arrayCopy<ct_uint32_t>(pSCfromSCtoPC[0], state.pSCfromSCtoPC[0],
//			cp->numSC * cp->numpSCfromSCtoPC);
//	arrayCopy<ct_uint32_t>(pPCfromBCtoPC[0], state.pPCfromBCtoPC[0],
//			cp->numPC * cp->numpPCfromBCtoPC);
//	arrayCopy<ct_uint32_t>(pPCfromPCtoBC[0], state.pPCfromPCtoBC[0],
//			cp->numPC * cp->numpPCfromPCtoBC);
//	arrayCopy<ct_uint32_t>(pPCfromSCtoPC[0], state.pPCfromSCtoPC[0],
//			cp->numPC * cp->numpPCfromSCtoPC);
//	arrayCopy<ct_uint32_t>(pPCfromPCtoNC[0], state.pPCfromPCtoNC[0],
//			cp->numPC * cp->numpPCfromPCtoNC);
//	arrayCopy<ct_uint32_t>(pPCfromIOtoPC, state.pPCfromIOtoPC, cp->numPC);
//
//	arrayCopy<ct_uint32_t>(pNCfromPCtoNC[0], state.pNCfromPCtoNC[0],
//			cp->numNC * cp->numpNCfromPCtoNC);
//	arrayCopy<ct_uint32_t>(pNCfromNCtoIO[0], state.pNCfromNCtoIO[0],
//			cp->numNC * cp->numpNCfromNCtoIO);
//	arrayCopy<ct_uint32_t>(pNCfromMFtoNC[0], state.pNCfromMFtoNC[0],
//			cp->numNC * cp->numpNCfromMFtoNC);
//
//	arrayCopy<ct_uint32_t>(pIOfromIOtoPC[0], state.pIOfromIOtoPC[0],
//			cp->numIO * cp->numpIOfromIOtoPC);
//	arrayCopy<ct_uint32_t>(pIOfromNCtoIO[0], state.pIOfromNCtoIO[0],
//			cp->numIO * cp->numpIOfromNCtoIO);
//	arrayCopy<ct_uint32_t>(pIOInIOIO[0], state.pIOInIOIO[0],
//			cp->numIO * cp->numpIOInIOIO);
//	arrayCopy<ct_uint32_t>(pIOOutIOIO[0], state.pIOOutIOIO[0],
//			cp->numIO * cp->numpIOOutIOIO);
//}

MZoneConnectivityState::~MZoneConnectivityState() {}

void MZoneConnectivityState::writeState(ConnectivityParams &cp, std::fstream &outfile)
{
	stateRW(cp, false, outfile);
}

void MZoneConnectivityState::stateRW(ConnectivityParams &cp, bool read, std::fstream &file)
{
	rawBytesRW((char *)pBCfromBCtoPC[0], cp.NUM_BC * cp.NUM_P_BC_FROM_BC_TO_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pBCfromPCtoBC[0], cp.NUM_BC * cp.NUM_P_BC_FROM_PC_TO_BC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pSCfromSCtoPC[0], cp.NUM_SC * cp.NUM_P_SC_FROM_SC_TO_PC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pPCfromBCtoPC[0], cp.NUM_PC * cp.NUM_P_PC_FROM_BC_TO_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromPCtoBC[0], cp.NUM_PC * cp.NUM_P_PC_FROM_PC_TO_BC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromSCtoPC[0], cp.NUM_PC * cp.NUM_P_PC_FROM_SC_TO_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromPCtoNC[0], cp.NUM_PC * cp.NUM_P_PC_FROM_PC_TO_NC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromIOtoPC, cp.NUM_PC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pNCfromPCtoNC[0], cp.NUM_NC * cp.NUM_P_NC_FROM_PC_TO_NC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pNCfromNCtoIO[0], cp.NUM_NC * cp.NUM_P_NC_FROM_NC_TO_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pNCfromMFtoNC[0], cp.NUM_NC * cp.NUM_P_NC_FROM_MF_TO_NC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pIOfromIOtoPC[0], cp.NUM_IO * cp.NUM_P_IO_FROM_IO_TO_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOfromNCtoIO[0], cp.NUM_IO * cp.NUM_P_IO_FROM_NC_TO_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOInIOIO[0], cp.NUM_IO * cp.NUM_P_IO_IN_IO_TO_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOOutIOIO[0], cp.NUM_IO * cp.NUMP_IO_OUT_IO_TO_IO * sizeof(ct_uint32_t), read, file);
}

bool MZoneConnectivityState::state_equal(const ConnectivityParams &cp,
	const MZoneConnectivityState &compState)
{
	bool eq = true;

	for (int i = 0; i < cp.NUM_BC; i++)
	{
		for (int j = 0; j < cp.NUM_P_BC_FROM_PC_TO_BC; j++)
		{
			eq = eq && (pBCfromPCtoBC[i][j] == compState.pBCfromPCtoBC[i][j]);
		}
	}

	for (int i = 0; i < cp.NUM_IO; i++)
	{
		for (int j = 0; j < cp.NUM_P_IO_IN_IO_TO_IO; j++)
		{
			eq = eq && (pIOInIOIO[i][j] == compState.pIOInIOIO[i][j]);
		}
	}
	return eq;
}

bool MZoneConnectivityState::state_unequal(const ConnectivityParams &cp,
	const MZoneConnectivityState &compState)
{
	return !state_equal(cp, compState);
}

void MZoneConnectivityState::initializeVars(ConnectivityParams &cp)
{
	std::fill(pBCfromBCtoPC[0], pBCfromBCtoPC[0] + cp.NUM_BC * cp.NUM_P_BC_FROM_BC_TO_PC, UINT_MAX);
	std::fill(pBCfromPCtoBC[0], pBCfromPCtoBC[0] + cp.NUM_BC * cp.NUM_P_BC_FROM_PC_TO_BC, UINT_MAX);

	std::fill(pSCfromSCtoPC[0], pSCfromSCtoPC[0] + cp.NUM_SC * cp.NUM_P_SC_FROM_SC_TO_PC, UINT_MAX);

	std::fill(pPCfromBCtoPC[0], pPCfromBCtoPC[0] + cp.NUM_PC * cp.NUM_P_PC_FROM_BC_TO_PC, UINT_MAX);
	std::fill(pPCfromPCtoBC[0], pPCfromPCtoBC[0] + cp.NUM_PC * cp.NUM_P_PC_FROM_PC_TO_BC, UINT_MAX);
	std::fill(pPCfromSCtoPC[0], pPCfromSCtoPC[0] + cp.NUM_PC * cp.NUM_P_PC_FROM_SC_TO_PC, UINT_MAX);
	std::fill(pPCfromPCtoNC[0], pPCfromPCtoNC[0] + cp.NUM_PC * cp.NUM_P_PC_FROM_PC_TO_NC, UINT_MAX);
	std::fill(pPCfromIOtoPC, pPCfromIOtoPC + cp.NUM_PC, UINT_MAX);

	std::fill(pNCfromPCtoNC[0], pNCfromPCtoNC[0] + cp.num_NC * cp.NUM_P_NC_FROM_PC_TO_NC, UINT_MAX);
	std::fill(pNCfromNCtoIO[0], pNCfromNCtoIO[0] + cp.num_NC * cp.NUM_P_NC_FROM_NC_TO_IO, UINT_MAX);
	std::fill(pNCfromMFtoNC[0], pNCfromMFtoNC[0] + cp.num_NC * cp.NUM_P_NC_FROM_MF_TO_NC, UINT_MAX);

	std::fill(pIOfromIOtoPC[0], pIOfromIOtoPC[0] + cp.NUM_IO * cp.NUM_P_IO_FROM_IO_TO_PC, UINT_MAX);
	std::fill(pIOfromNCtoIO[0], pIOfromNCtoIO[0] + cp.NUM_IO * cp.NUM_P_IO_FROM_NC_TO_IO, UINT_MAX);
	std::fill(pIOInIOIO[0], pIOInIOIO[0] + cp.NUM_IO * cp.NUM_P_IO_IN_IO_TO_IO, UINT_MAX);
	std::fill(pIOOutIOIO[0], pIOOutIOIO[0] + cp.NUM_IO * cp.NUMP_IO_OUT_IO_TO_IO, UINT_MAX);
}

void MZoneConnectivityState::connectBCtoPC(ConnectivityParams &cp)
{
	int bcToPCRatio = cp.NUM_BC / cp.NUM_PC;

	for (int i = 0; i < cp.NUM_PC; i++)
	{
		pBCfromBCtoPC[i * bcToPCRatio][0] = ((i + 1) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio][1] = ((i - 1) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio][2] = ((i + 2) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio][3] = ((i - 2) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;

		pBCfromBCtoPC[i * bcToPCRatio+1][0] = ((i + 1) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+1][1] = ((i - 1) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+1][2] = ((i + 3) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+1][3] = ((i - 3) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;

		pBCfromBCtoPC[i * bcToPCRatio+2][0] = ((i + 3) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+2][1] = ((i - 3) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+2][2] = ((i + 6) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+2][3] = ((i - 6) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;

		pBCfromBCtoPC[i * bcToPCRatio+3][0] = ((i + 4) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+3][1] = ((i - 4) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+3][2] = ((i + 9) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+3][3] = ((i - 9) % cp.NUM_PC + cp.NUM_PC) % cp.NUM_PC;
	}
}

void MZoneConnectivityState::connectPCtoBC(ConnectivityParams &cp)
{
	int bcToPCRatio = cp.NUM_BC / cp.NUM_PC;

	for (int i = 0; i < cp.NUM_PC; i++)
	{
		for (int j = 0; j < cp.NUM_P_PC_FROM_PC_TO_BC; j++)
		{
			int bcInd = i * bcToPCRatio - 6 + j;
			pPCfromPCtoBC[i][j] = (bcInd % cp.NUM_BC + cp.NUM_BC) % cp.NUM_BC;
		}
	}
}

void MZoneConnectivityState::connectSCtoPC(ConnectivityParams &cp)
{
	for (int i = 0; i < cp.NUM_SC; i++)
	{
		for (int j = 0; j < cp.NUM_P_SC_FROM_SC_TO_PC; j++)
		{
			int pcInd = i / cp.NUM_P_PC_FROM_SC_TO_PC;
			pSCfromSCtoPC[i][j] = pcInd;
			pPCfromSCtoPC[pcInd][i % cp.NUM_P_PC_FROM_SC_TO_PC] = i;
		}
	}
}

void MZoneConnectivityState::connectPCtoNC(ConnectivityParams &cp, int randSeed)
{
	int pcNumConnected[cp.NUM_PC]();
	std::fill(pcNumConnected, pcNumConnected + cp.NUM_PC, 1);

	CRandomSFMT0 randGen(randSeed);
	
	for (int i = 0; i < cp.NUM_NC; i++)
	{
		for (int j = 0; j < cp.NUM_PC / cp.NUM_NC; j++)
		{
			int pcInd = i * (cp.NUM_PC / cp.NUM_NC) + j;
			pNCfromPCtoNC[i][j] = pcInd;
			pPCfromPCtoNC[pcInd][0] = i;
		}
	}

	for (int i = 0; i < cp.NUM_NC - 1; i++)
	{
		for (int j = cp.NUM_P_NC_FROM_PC_TO_NC / 3; j < cp.NUM_P_NC_FROM_PC_TO_NC; j++)
		{
			int countPCNC = 0;
			
			while(true)
			{
				bool connect = true;

				int pcInd = randGen.IRandomX(0, cp.NUM_PC - 1);

				if (pcNumConnected[pcInd] >= cp.NUM_P_PC_FROM_PC_TO_NC) continue;
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
	unsigned int numSyn = cp.NUM_PC / cp.NUM_NC;

	for (int h = 1; h < cp.NUM_P_PC_FROM_PC_TO_NC; h++)
	{
		for(int i = 0; i < cp.NUM_PC; i++)
		{
			if (pcNumConnected[i] < cp.NUM_P_PC_FROM_PC_TO_NC)
			{
				pNCfromPCtoNC[cp.NUM_NC - 1][numSyn] = i;
				pPCfromPCtoNC[i][pcNumConnected[i]] = cp.NUM_NC - 1;

				pcNumConnected[i]++;
				numSyn++;
			}
		}
	}
}

void MZoneConnectivityState::connectNCtoIO(ConnectivityParams &cp)
{
	for (int i = 0; i < cp.NUM_IO; i++)
	{
		for (int j = 0; j < cp.NUM_P_IO_FROM_NC_TO_IO; j++)
		{
			pIOfromNCtoIO[i][j] = j;
			pNCfromNCtoIO[j][i] = i;
		}
	}
}

void MZoneConnectivityState::connectMFtoNC(ConnectivityParams &cp)
{
	for (int i = 0; i < cp.NUM_MF; i++)
	{
		for (int j = 0; j < cp.NUM_P_MF_FROM_MF_TO_NC; j++)
		{
			pNCfromMFtoNC[i / cp.NUM_P_NC_FROM_MF_TO_NC][i % cp.NUM_P_NC_FROM_MF_TO_NC] = i;
		}
	}
}

void MZoneConnectivityState::connectIOtoPC(ConnectivityParams &cp)
{
	for (int i = 0; i < cp.NUM_IO; i++)
	{
		for (int j = 0; j < cp.NUM_P_IO_FROM_IO_TO_PC; j++)
		{
			int pcInd = i * cp.NUM_P_IO_FROM_IO_TO_PC + j;

			pIOfromIOtoPC[i][j]  = pcInd;
			pPCfromIOtoPC[pcInd] = i;
		}
	}
}

void MZoneConnectivityState::connectIOtoIO(ConnectivityParams &cp)
{
	for (int i = 0; i < cp.NUM_IO; i++)
	{
		int inInd = 0;
		for (int j = 0; j < cp.NUM_P_IO_IN_IO_TO_IO; j++)
		{
			if (inInd == i) inInd++;
			pIOInIOIO[i][j] = inInd;
			inInd++;
		}
	}
}

