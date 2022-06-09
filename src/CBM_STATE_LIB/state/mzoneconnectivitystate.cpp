/*
 * mzoneconnectivitystate.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: consciousness
 */

#include "state/mzoneconnectivitystate.h"

MZoneConnectivityState::MZoneConnectivityState(int randSeed)
{
	initializeVars();

	connectBCtoPC();
	connectPCtoBC();
	connectSCtoPC();
	connectPCtoNC(randSeed);
	connectNCtoIO();
	connectMFtoNC();
	connectIOtoPC();
	connectIOtoIO();
}

MZoneConnectivityState::MZoneConnectivityState(std::fstream &infile)
{
	stateRW(true, infile);
}

//MZoneConnectivityState::MZoneConnectivityState(const MZoneConnectivityState &state)
//{
//	arrayCopy<ct_uint32_t>(pBCfromBCtoPC[0], state.pBCfromBCtoPC[0],
//			numBC * numpBCfromBCtoPC);
//	arrayCopy<ct_uint32_t>(pBCfromPCtoBC[0], state.pBCfromPCtoBC[0],
//			numBC * numpBCfromPCtoBC);
//	arrayCopy<ct_uint32_t>(pSCfromSCtoPC[0], state.pSCfromSCtoPC[0],
//			numSC * numpSCfromSCtoPC);
//	arrayCopy<ct_uint32_t>(pPCfromBCtoPC[0], state.pPCfromBCtoPC[0],
//			numPC * numpPCfromBCtoPC);
//	arrayCopy<ct_uint32_t>(pPCfromPCtoBC[0], state.pPCfromPCtoBC[0],
//			numPC * numpPCfromPCtoBC);
//	arrayCopy<ct_uint32_t>(pPCfromSCtoPC[0], state.pPCfromSCtoPC[0],
//			numPC * numpPCfromSCtoPC);
//	arrayCopy<ct_uint32_t>(pPCfromPCtoNC[0], state.pPCfromPCtoNC[0],
//			numPC * numpPCfromPCtoNC);
//	arrayCopy<ct_uint32_t>(pPCfromIOtoPC, state.pPCfromIOtoPC, numPC);
//
//	arrayCopy<ct_uint32_t>(pNCfromPCtoNC[0], state.pNCfromPCtoNC[0],
//			numNC * numpNCfromPCtoNC);
//	arrayCopy<ct_uint32_t>(pNCfromNCtoIO[0], state.pNCfromNCtoIO[0],
//			numNC * numpNCfromNCtoIO);
//	arrayCopy<ct_uint32_t>(pNCfromMFtoNC[0], state.pNCfromMFtoNC[0],
//			numNC * numpNCfromMFtoNC);
//
//	arrayCopy<ct_uint32_t>(pIOfromIOtoPC[0], state.pIOfromIOtoPC[0],
//			numIO * numpIOfromIOtoPC);
//	arrayCopy<ct_uint32_t>(pIOfromNCtoIO[0], state.pIOfromNCtoIO[0],
//			numIO * numpIOfromNCtoIO);
//	arrayCopy<ct_uint32_t>(pIOInIOIO[0], state.pIOInIOIO[0],
//			numIO * numpIOInIOIO);
//	arrayCopy<ct_uint32_t>(pIOOutIOIO[0], state.pIOOutIOIO[0],
//			numIO * numpIOOutIOIO);
//}

MZoneConnectivityState::~MZoneConnectivityState() {}

void MZoneConnectivityState::writeState(std::fstream &outfile)
{
	stateRW(false, outfile);
}

void MZoneConnectivityState::stateRW(bool read, std::fstream &file)
{
	rawBytesRW((char *)pBCfromBCtoPC[0], NUM_BC * NUM_P_BC_FROM_BC_TO_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pBCfromPCtoBC[0], NUM_BC * NUM_P_BC_FROM_PC_TO_BC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pSCfromSCtoPC[0], NUM_SC * NUM_P_SC_FROM_SC_TO_PC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pPCfromBCtoPC[0], NUM_PC * NUM_P_PC_FROM_BC_TO_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromPCtoBC[0], NUM_PC * NUM_P_PC_FROM_PC_TO_BC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromSCtoPC[0], NUM_PC * NUM_P_PC_FROM_SC_TO_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromPCtoNC[0], NUM_PC * NUM_P_PC_FROM_PC_TO_NC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromIOtoPC, NUM_PC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pNCfromPCtoNC[0], NUM_NC * NUM_P_NC_FROM_PC_TO_NC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pNCfromNCtoIO[0], NUM_NC * NUM_P_NC_FROM_NC_TO_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pNCfromMFtoNC[0], NUM_NC * NUM_P_NC_FROM_MF_TO_NC * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)pIOfromIOtoPC[0], NUM_IO * NUM_P_IO_FROM_IO_TO_PC * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOfromNCtoIO[0], NUM_IO * NUM_P_IO_FROM_NC_TO_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOInIOIO[0], NUM_IO * NUM_P_IO_IN_IO_TO_IO * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOOutIOIO[0], NUM_IO * NUMP_IO_OUT_IO_TO_IO * sizeof(ct_uint32_t), read, file);
}

bool MZoneConnectivityState::state_equal(const MZoneConnectivityState &compState)
{
	bool eq = true;

	for (int i = 0; i < NUM_BC; i++)
	{
		for (int j = 0; j < NUM_P_BC_FROM_PC_TO_BC; j++)
		{
			eq = eq && (pBCfromPCtoBC[i][j] == compState.pBCfromPCtoBC[i][j]);
		}
	}

	for (int i = 0; i < NUM_IO; i++)
	{
		for (int j = 0; j < NUM_P_IO_IN_IO_TO_IO; j++)
		{
			eq = eq && (pIOInIOIO[i][j] == compState.pIOInIOIO[i][j]);
		}
	}
	return eq;
}

bool MZoneConnectivityState::state_unequal(const MZoneConnectivityState &compState)
{
	return !state_equal(compState);
}

void MZoneConnectivityState::initializeVars()
{
	std::fill(pBCfromBCtoPC[0], pBCfromBCtoPC[0] + NUM_BC * NUM_P_BC_FROM_BC_TO_PC, UINT_MAX);
	std::fill(pBCfromPCtoBC[0], pBCfromPCtoBC[0] + NUM_BC * NUM_P_BC_FROM_PC_TO_BC, UINT_MAX);

	std::fill(pSCfromSCtoPC[0], pSCfromSCtoPC[0] + NUM_SC * NUM_P_SC_FROM_SC_TO_PC, UINT_MAX);

	std::fill(pPCfromBCtoPC[0], pPCfromBCtoPC[0] + NUM_PC * NUM_P_PC_FROM_BC_TO_PC, UINT_MAX);
	std::fill(pPCfromPCtoBC[0], pPCfromPCtoBC[0] + NUM_PC * NUM_P_PC_FROM_PC_TO_BC, UINT_MAX);
	std::fill(pPCfromSCtoPC[0], pPCfromSCtoPC[0] + NUM_PC * NUM_P_PC_FROM_SC_TO_PC, UINT_MAX);
	std::fill(pPCfromPCtoNC[0], pPCfromPCtoNC[0] + NUM_PC * NUM_P_PC_FROM_PC_TO_NC, UINT_MAX);
	std::fill(pPCfromIOtoPC, pPCfromIOtoPC + NUM_PC, UINT_MAX);

	std::fill(pNCfromPCtoNC[0], pNCfromPCtoNC[0] + num_NC * NUM_P_NC_FROM_PC_TO_NC, UINT_MAX);
	std::fill(pNCfromNCtoIO[0], pNCfromNCtoIO[0] + num_NC * NUM_P_NC_FROM_NC_TO_IO, UINT_MAX);
	std::fill(pNCfromMFtoNC[0], pNCfromMFtoNC[0] + num_NC * NUM_P_NC_FROM_MF_TO_NC, UINT_MAX);

	std::fill(pIOfromIOtoPC[0], pIOfromIOtoPC[0] + NUM_IO * NUM_P_IO_FROM_IO_TO_PC, UINT_MAX);
	std::fill(pIOfromNCtoIO[0], pIOfromNCtoIO[0] + NUM_IO * NUM_P_IO_FROM_NC_TO_IO, UINT_MAX);
	std::fill(pIOInIOIO[0], pIOInIOIO[0] + NUM_IO * NUM_P_IO_IN_IO_TO_IO, UINT_MAX);
	std::fill(pIOOutIOIO[0], pIOOutIOIO[0] + NUM_IO * NUMP_IO_OUT_IO_TO_IO, UINT_MAX);
}

void MZoneConnectivityState::connectBCtoPC()
{
	int bcToPCRatio = NUM_BC / NUM_PC;

	for (int i = 0; i < NUM_PC; i++)
	{
		pBCfromBCtoPC[i * bcToPCRatio][0] = ((i + 1) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio][1] = ((i - 1) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio][2] = ((i + 2) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio][3] = ((i - 2) % NUM_PC + NUM_PC) % NUM_PC;

		pBCfromBCtoPC[i * bcToPCRatio+1][0] = ((i + 1) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+1][1] = ((i - 1) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+1][2] = ((i + 3) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+1][3] = ((i - 3) % NUM_PC + NUM_PC) % NUM_PC;

		pBCfromBCtoPC[i * bcToPCRatio+2][0] = ((i + 3) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+2][1] = ((i - 3) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+2][2] = ((i + 6) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+2][3] = ((i - 6) % NUM_PC + NUM_PC) % NUM_PC;

		pBCfromBCtoPC[i * bcToPCRatio+3][0] = ((i + 4) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+3][1] = ((i - 4) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+3][2] = ((i + 9) % NUM_PC + NUM_PC) % NUM_PC;
		pBCfromBCtoPC[i * bcToPCRatio+3][3] = ((i - 9) % NUM_PC + NUM_PC) % NUM_PC;
	}
}

void MZoneConnectivityState::connectPCtoBC()
{
	int bcToPCRatio = NUM_BC / NUM_PC;

	for (int i = 0; i < NUM_PC; i++)
	{
		for (int j = 0; j < NUM_P_PC_FROM_PC_TO_BC; j++)
		{
			int bcInd = i * bcToPCRatio - 6 + j;
			pPCfromPCtoBC[i][j] = (bcInd % NUM_BC + NUM_BC) % NUM_BC;
		}
	}
}

void MZoneConnectivityState::connectSCtoPC()
{
	for (int i = 0; i < NUM_SC; i++)
	{
		for (int j = 0; j < NUM_P_SC_FROM_SC_TO_PC; j++)
		{
			int pcInd = i / NUM_P_PC_FROM_SC_TO_PC;
			pSCfromSCtoPC[i][j] = pcInd;
			pPCfromSCtoPC[pcInd][i % NUM_P_PC_FROM_SC_TO_PC] = i;
		}
	}
}

void MZoneConnectivityState::connectPCtoNC(int randSeed)
{
	int pcNumConnected[NUM_PC] = {0};
	std::fill(pcNumConnected, pcNumConnected + NUM_PC, 1);

	CRandomSFMT0 randGen(randSeed);
	
	for (int i = 0; i < NUM_NC; i++)
	{
		for (int j = 0; j < NUM_PC / NUM_NC; j++)
		{
			int pcInd = i * (NUM_PC / NUM_NC) + j;
			pNCfromPCtoNC[i][j] = pcInd;
			pPCfromPCtoNC[pcInd][0] = i;
		}
	}

	for (int i = 0; i < NUM_NC - 1; i++)
	{
		for (int j = NUM_P_NC_FROM_PC_TO_NC / 3; j < NUM_P_NC_FROM_PC_TO_NC; j++)
		{
			int countPCNC = 0;
			
			while(true)
			{
				bool connect = true;

				int pcInd = randGen.IRandomX(0, NUM_PC - 1);

				if (pcNumConnected[pcInd] >= NUM_P_PC_FROM_PC_TO_NC) continue;
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
	unsigned int numSyn = NUM_PC / NUM_NC;

	for (int h = 1; h < NUM_P_PC_FROM_PC_TO_NC; h++)
	{
		for(int i = 0; i < NUM_PC; i++)
		{
			if (pcNumConnected[i] < NUM_P_PC_FROM_PC_TO_NC)
			{
				pNCfromPCtoNC[NUM_NC - 1][numSyn] = i;
				pPCfromPCtoNC[i][pcNumConnected[i]] = NUM_NC - 1;

				pcNumConnected[i]++;
				numSyn++;
			}
		}
	}
}

void MZoneConnectivityState::connectNCtoIO()
{
	for (int i = 0; i < NUM_IO; i++)
	{
		for (int j = 0; j < NUM_P_IO_FROM_NC_TO_IO; j++)
		{
			pIOfromNCtoIO[i][j] = j;
			pNCfromNCtoIO[j][i] = i;
		}
	}
}

void MZoneConnectivityState::connectMFtoNC()
{
	for (int i = 0; i < NUM_MF; i++)
	{
		for (int j = 0; j < NUM_P_MF_FROM_MF_TO_NC; j++)
		{
			pNCfromMFtoNC[i / NUM_P_NC_FROM_MF_TO_NC][i % NUM_P_NC_FROM_MF_TO_NC] = i;
		}
	}
}

void MZoneConnectivityState::connectIOtoPC()
{
	for (int i = 0; i < NUM_IO; i++)
	{
		for (int j = 0; j < NUM_P_IO_FROM_IO_TO_PC; j++)
		{
			int pcInd = i * NUM_P_IO_FROM_IO_TO_PC + j;

			pIOfromIOtoPC[i][j]  = pcInd;
			pPCfromIOtoPC[pcInd] = i;
		}
	}
}

void MZoneConnectivityState::connectIOtoIO()
{
	for (int i = 0; i < NUM_IO; i++)
	{
		int inInd = 0;
		for (int j = 0; j < NUM_P_IO_IN_IO_TO_IO; j++)
		{
			if (inInd == i) inInd++;
			pIOInIOIO[i][j] = inInd;
			inInd++;
		}
	}
}

