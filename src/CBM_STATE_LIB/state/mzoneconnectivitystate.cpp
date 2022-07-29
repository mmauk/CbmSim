/*
 * mzoneconnectivitystate.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: consciousness
 */

#include "state/mzoneconnectivitystate.h"

MZoneConnectivityState::MZoneConnectivityState(ConnectivityParams *cp, int randSeed)
{
	std::cout << "[INFO]: Allocating and initializing mzone connectivity arrays..." << std::endl;
	allocateMemory(cp);
	initializeVals(cp);
	std::cout << "[INFO]: Initializing mzone connections..." << std::endl;
	std::cout << "[INFO]: Connecting BC to PC" << std::endl;
	connectBCtoPC(cp);
	std::cout << "[INFO]: Connecting PC to BC" << std::endl;
	connectPCtoBC(cp);
	std::cout << "[INFO]: Connecting SC and PC" << std::endl;
	connectSCtoPC(cp);
	std::cout << "[INFO]: Connecting PC and NC" << std::endl;
	connectPCtoNC(cp, randSeed);
	std::cout << "[INFO]: Connecting NC and IO" << std::endl;
	connectNCtoIO(cp);
	std::cout << "[INFO]: Connecting MF and NC" << std::endl;
	connectMFtoNC(cp);
	std::cout << "[INFO]: Connecting IO and PC" << std::endl;
	connectIOtoPC(cp);
	std::cout << "[INFO]: Connecting IO and IO" << std::endl;
	connectIOtoIO(cp);
	std::cout << "[INFO]: Finished making mzone connections." << std::endl;
}

MZoneConnectivityState::MZoneConnectivityState(ConnectivityParams *cp, std::fstream &infile)
{
	allocateMemory(cp);
	stateRW(cp, true, infile);
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

MZoneConnectivityState::~MZoneConnectivityState() {deallocMemory();}

void MZoneConnectivityState::readState(ConnectivityParams *cp, std::fstream &infile)
{
	stateRW(cp, true, infile);
}

void MZoneConnectivityState::writeState(ConnectivityParams *cp, std::fstream &outfile)
{
	stateRW(cp, false, outfile);
}

void MZoneConnectivityState::allocateMemory(ConnectivityParams *cp)
{
	//basket cells
	pBCfromBCtoPC  = allocate2DArray<ct_uint32_t>(cp->int_params["num_bc"], cp->int_params["num_p_bc_from_bc_to_pc"]);
	pBCfromPCtoBC  = allocate2DArray<ct_uint32_t>(cp->int_params["num_bc"], cp->int_params["num_p_bc_from_pc_to_bc"]);

	//stellate cells
	pSCfromSCtoPC = allocate2DArray<ct_uint32_t>(cp->int_params["num_sc"], cp->int_params["num_p_sc_from_sc_to_pc"]);

	//purkinje cells
	pPCfromBCtoPC = allocate2DArray<ct_uint32_t>(cp->int_params["num_pc"], cp->int_params["num_p_pc_from_bc_to_pc"]);
	pPCfromPCtoBC = allocate2DArray<ct_uint32_t>(cp->int_params["num_pc"], cp->int_params["num_p_pc_from_pc_to_bc"]);
	pPCfromSCtoPC = allocate2DArray<ct_uint32_t>(cp->int_params["num_pc"], cp->int_params["num_p_pc_from_sc_to_pc"]);
	pPCfromPCtoNC = allocate2DArray<ct_uint32_t>(cp->int_params["num_pc"], cp->int_params["num_p_pc_from_pc_to_nc"]);
	pPCfromIOtoPC = new ct_uint32_t[cp->int_params["num_pc"]]();

	//nucleus cells
	pNCfromPCtoNC = allocate2DArray<ct_uint32_t>(cp->int_params["num_nc"], cp->int_params["num_p_nc_from_pc_to_nc"]);
	pNCfromNCtoIO = allocate2DArray<ct_uint32_t>(cp->int_params["num_nc"], cp->int_params["num_p_nc_from_nc_to_io"]);
	pNCfromMFtoNC = allocate2DArray<ct_uint32_t>(cp->int_params["num_nc"], cp->int_params["num_p_nc_from_mf_to_nc"]);

	//inferior olivary cells
	pIOfromIOtoPC = allocate2DArray<ct_uint32_t>(cp->int_params["num_io"], cp->int_params["num_p_io_from_io_to_pc"]);
	pIOfromNCtoIO = allocate2DArray<ct_uint32_t>(cp->int_params["num_io"], cp->int_params["num_p_io_from_nc_to_io"]);
	pIOInIOIO = allocate2DArray<ct_uint32_t>(cp->int_params["num_io"], cp->int_params["num_p_io_in_io_to_io"]);
	pIOOutIOIO = allocate2DArray<ct_uint32_t>(cp->int_params["num_io"], cp->int_params["num_p_io_out_io_to_io"]);
}

void MZoneConnectivityState::initializeVals(ConnectivityParams *cp)
{
	// basket cells
	std::fill(pBCfromBCtoPC[0], pBCfromBCtoPC[0]
			+ cp->int_params["num_bc"] * cp->int_params["num_p_bc_from_bc_to_pc"], 0);
	std::fill(pBCfromPCtoBC[0], pBCfromPCtoBC[0]
			+ cp->int_params["num_bc"] * cp->int_params["num_p_bc_from_pc_to_bc"], 0);

	// stellate cells
	std::fill(pSCfromSCtoPC[0], pSCfromSCtoPC[0]
			+ cp->int_params["num_sc"] * cp->int_params["num_p_sc_from_sc_to_pc"], 0);

	// purkinje cells
	std::fill(pPCfromBCtoPC[0], pPCfromBCtoPC[0]
			+ cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_bc_to_pc"], 0);
	std::fill(pPCfromPCtoBC[0], pPCfromPCtoBC[0]
			+ cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_pc_to_bc"], 0);
	std::fill(pPCfromSCtoPC[0], pPCfromSCtoPC[0]
			+ cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_sc_to_pc"], 0);
	std::fill(pPCfromPCtoNC[0], pPCfromPCtoNC[0]
			+ cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_pc_to_nc"], 0);

	// nucleus cells
	std::fill(pNCfromPCtoNC[0], pNCfromPCtoNC[0]
			+ cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_pc_to_nc"], 0);
	std::fill(pNCfromNCtoIO[0], pNCfromNCtoIO[0]
			+ cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_nc_to_io"], 0);
	std::fill(pNCfromMFtoNC[0], pNCfromMFtoNC[0]
			+ cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"], 0);

	// inferior olivary cells
	std::fill(pIOfromIOtoPC[0], pIOfromIOtoPC[0]
			+ cp->int_params["num_io"] * cp->int_params["num_p_io_from_io_to_pc"], 0);
	std::fill(pIOfromNCtoIO[0], pIOfromNCtoIO[0]
			+ cp->int_params["num_io"] * cp->int_params["num_p_io_from_nc_to_io"], 0);
	std::fill(pIOInIOIO[0], pIOInIOIO[0]
			+ cp->int_params["num_io"] * cp->int_params["num_p_io_in_io_to_io"], 0);
	std::fill(pIOOutIOIO[0], pIOOutIOIO[0]
			+ cp->int_params["num_io"] * cp->int_params["num_p_io_out_io_to_io"], 0);
}

void MZoneConnectivityState::deallocMemory()
{
	// basket cells
	delete2DArray<ct_uint32_t>(pBCfromBCtoPC);
	delete2DArray<ct_uint32_t>(pBCfromPCtoBC);

	// stellate cells
	delete2DArray<ct_uint32_t>(pSCfromSCtoPC);

	// purkinje cells
	delete2DArray<ct_uint32_t>(pPCfromBCtoPC);
	delete2DArray<ct_uint32_t>(pPCfromPCtoBC);
	delete2DArray<ct_uint32_t>(pPCfromSCtoPC);
	delete2DArray<ct_uint32_t>(pPCfromPCtoNC);
	delete[] pPCfromIOtoPC;

	// nucleus cells
	delete2DArray<ct_uint32_t>(pNCfromPCtoNC);
	delete2DArray<ct_uint32_t>(pNCfromNCtoIO);
	delete2DArray<ct_uint32_t>(pNCfromMFtoNC);

	// inferior olivary cells
	delete2DArray<ct_uint32_t>(pIOfromIOtoPC);
	delete2DArray<ct_uint32_t>(pIOfromNCtoIO);
	delete2DArray<ct_uint32_t>(pIOInIOIO);
	delete2DArray<ct_uint32_t>(pIOOutIOIO);
}

void MZoneConnectivityState::stateRW(ConnectivityParams *cp, bool read, std::fstream &file)
{
	// basket cells
	rawBytesRW((char *)pBCfromBCtoPC[0], cp->int_params["num_bc"] * cp->int_params["num_p_bc_from_bc_to_pc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pBCfromPCtoBC[0], cp->int_params["num_bc"] * cp->int_params["num_p_bc_from_pc_to_bc"] * sizeof(ct_uint32_t), read, file);

	// stellate cells
	rawBytesRW((char *)pSCfromSCtoPC[0], cp->int_params["num_sc"] * cp->int_params["num_p_sc_from_sc_to_pc"] * sizeof(ct_uint32_t), read, file);

	// purkinje cells
	rawBytesRW((char *)pPCfromBCtoPC[0], cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_bc_to_pc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromPCtoBC[0], cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_pc_to_bc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromSCtoPC[0], cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_sc_to_pc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromPCtoNC[0], cp->int_params["num_pc"] * cp->int_params["num_p_pc_from_pc_to_nc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pPCfromIOtoPC, cp->int_params["num_pc"] * sizeof(ct_uint32_t), read, file);

	// nucleus cells
	rawBytesRW((char *)pNCfromPCtoNC[0], cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_pc_to_nc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pNCfromNCtoIO[0], cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_nc_to_io"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pNCfromMFtoNC[0], cp->int_params["num_nc"] * cp->int_params["num_p_nc_from_mf_to_nc"] * sizeof(ct_uint32_t), read, file);

	// inferior olivary cells
	rawBytesRW((char *)pIOfromIOtoPC[0], cp->int_params["num_io"] * cp->int_params["num_p_io_from_io_to_pc"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOfromNCtoIO[0], cp->int_params["num_io"] * cp->int_params["num_p_io_from_nc_to_io"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOInIOIO[0], cp->int_params["num_io"] * cp->int_params["num_p_io_in_io_to_io"] * sizeof(ct_uint32_t), read, file);
	rawBytesRW((char *)pIOOutIOIO[0], cp->int_params["num_io"] * cp->int_params["num_p_io_out_io_to_io"] * sizeof(ct_uint32_t), read, file);
}

void MZoneConnectivityState::connectBCtoPC(ConnectivityParams *cp)
{
	int bcToPCRatio = cp->int_params["num_bc"] / cp->int_params["num_pc"];

	for (int i = 0; i < cp->int_params["num_pc"]; i++)
	{
		pBCfromBCtoPC[i * bcToPCRatio][0] = ((i + 1) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio][1] = ((i - 1) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio][2] = ((i + 2) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio][3] = ((i - 2) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];

		pBCfromBCtoPC[i * bcToPCRatio+1][0] = ((i + 1) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+1][1] = ((i - 1) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+1][2] = ((i + 3) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+1][3] = ((i - 3) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];

		pBCfromBCtoPC[i * bcToPCRatio+2][0] = ((i + 3) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+2][1] = ((i - 3) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+2][2] = ((i + 6) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+2][3] = ((i - 6) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];

		pBCfromBCtoPC[i * bcToPCRatio+3][0] = ((i + 4) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+3][1] = ((i - 4) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+3][2] = ((i + 9) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
		pBCfromBCtoPC[i * bcToPCRatio+3][3] = ((i - 9) % cp->int_params["num_pc"] + cp->int_params["num_pc"]) % cp->int_params["num_pc"];
	}
}

void MZoneConnectivityState::connectPCtoBC(ConnectivityParams *cp)
{
	int bcToPCRatio = cp->int_params["num_bc"] / cp->int_params["num_pc"];

	for (int i = 0; i < cp->int_params["num_pc"]; i++)
	{
		for (int j = 0; j < cp->int_params["num_p_pc_from_pc_to_bc"]; j++)
		{
			int bcInd = i * bcToPCRatio - 6 + j;
			pPCfromPCtoBC[i][j] = (bcInd % cp->int_params["num_bc"] + cp->int_params["num_bc"]) % cp->int_params["num_bc"];
		}
	}
}

void MZoneConnectivityState::connectSCtoPC(ConnectivityParams *cp)
{
	for (int i = 0; i < cp->int_params["num_sc"]; i++)
	{
		for (int j = 0; j < cp->int_params["num_p_sc_from_sc_to_pc"]; j++)
		{
			int pcInd = i / cp->int_params["num_p_pc_from_sc_to_pc"];
			pSCfromSCtoPC[i][j] = pcInd;
			pPCfromSCtoPC[pcInd][i % cp->int_params["num_p_pc_from_sc_to_pc"]] = i;
		}
	}
}

void MZoneConnectivityState::connectPCtoNC(ConnectivityParams *cp, int randSeed)
{
	int pcNumConnected[cp->int_params["num_pc"]] = {0};
	std::fill(pcNumConnected, pcNumConnected + cp->int_params["num_pc"], 1);

	CRandomSFMT0 randGen(randSeed);
	
	for (int i = 0; i < cp->int_params["num_nc"]; i++)
	{
		for (int j = 0; j < cp->int_params["num_pc"] / cp->int_params["num_nc"]; j++)
		{
			int pcInd = i * (cp->int_params["num_pc"] / cp->int_params["num_nc"]) + j;
			pNCfromPCtoNC[i][j] = pcInd;
			pPCfromPCtoNC[pcInd][0] = i;
		}
	}

	for (int i = 0; i < cp->int_params["num_nc"] - 1; i++)
	{
		for (int j = cp->int_params["num_p_nc_from_pc_to_nc"] / 3; j < cp->int_params["num_p_nc_from_pc_to_nc"]; j++)
		{
			int countPCNC = 0;
			int pcInd;	
			while(true)
			{
				bool connect = true;

				pcInd = randGen.IRandomX(0, cp->int_params["num_pc"] - 1);

				if (pcNumConnected[pcInd] >= cp->int_params["num_p_pc_from_pc_to_nc"]) continue;
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
	unsigned int numSyn = cp->int_params["num_pc"] / cp->int_params["num_nc"];

	for (int h = 1; h < cp->int_params["num_p_pc_from_pc_to_nc"]; h++)
	{
		for(int i = 0; i < cp->int_params["num_pc"]; i++)
		{
			if (pcNumConnected[i] < cp->int_params["num_p_pc_from_pc_to_nc"])
			{
				pNCfromPCtoNC[cp->int_params["num_nc"] - 1][numSyn] = i;
				pPCfromPCtoNC[i][pcNumConnected[i]] = cp->int_params["num_nc"] - 1;

				pcNumConnected[i]++;
				numSyn++;
			}
		}
	}
}

void MZoneConnectivityState::connectNCtoIO(ConnectivityParams *cp)
{
	for (int i = 0; i < cp->int_params["num_io"]; i++)
	{
		for (int j = 0; j < cp->int_params["num_p_io_from_nc_to_io"]; j++)
		{
			pIOfromNCtoIO[i][j] = j;
			pNCfromNCtoIO[j][i] = i;
		}
	}
}

void MZoneConnectivityState::connectMFtoNC(ConnectivityParams *cp)
{
	for (int i = 0; i < cp->int_params["num_mf"]; i++)
	{
		for (int j = 0; j < cp->int_params["num_p_mf_from_mf_to_nc"]; j++)
		{
			pNCfromMFtoNC[i / cp->int_params["num_p_nc_from_mf_to_nc"]][i % cp->int_params["num_p_nc_from_mf_to_nc"]] = i;
		}
	}
}

void MZoneConnectivityState::connectIOtoPC(ConnectivityParams *cp)
{
	for (int i = 0; i < cp->int_params["num_io"]; i++)
	{
		for (int j = 0; j < cp->int_params["num_p_io_from_io_to_pc"]; j++)
		{
			int pcInd = i * cp->int_params["num_p_io_from_io_to_pc"] + j;

			pIOfromIOtoPC[i][j]  = pcInd;
			pPCfromIOtoPC[pcInd] = i;
		}
	}
}

void MZoneConnectivityState::connectIOtoIO(ConnectivityParams *cp)
{
	for (int i = 0; i < cp->int_params["num_io"]; i++)
	{
		int inInd = 0;
		for (int j = 0; j < cp->int_params["num_p_io_in_io_to_io"]; j++)
		{
			if (inInd == i) inInd++;
			pIOInIOIO[i][j] = inInd;
			inInd++;
		}
	}
}

