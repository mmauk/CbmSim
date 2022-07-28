/*
 * innetconnectivitystate.cpp
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */
#include "state/innetconnectivitystate.h"

InNetConnectivityState::InNetConnectivityState(
		ConnectivityParams *cp,
		unsigned int msPerStep,
		int randSeed
		)
{
	CRandomSFMT0 randGen(randSeed);

	std::cout << "[INFO]: allocating and initializing connectivity arrays..." << std::endl;
	allocateMemory(cp);
	initializeVals(cp);

	std::cout << "[INFO]: Initializing innet connections..." << std::endl;

	std::cout << "[INFO]: Connecting MF and GL" << std::endl;
	connectMFGL_noUBC(cp);

	std::cout << "[INFO]: Connecting GR and GL" << std::endl;
	connectGLGR(cp, randGen);

	std::cout << "[INFO]: Connecting GR to GO" << std::endl;
	connectGRGO(cp);

	std::cout << "[INFO]: Connecting GO and GL" << std::endl;
	connectGOGL(cp, randGen);

	std::cout << "[INFO]: Connecting GO to GO" << std::endl;
	connectGOGODecayP(cp, randGen);

	std::cout << "[INFO]: Connecting GO to GO gap junctions" << std::endl;
	connectGOGO_GJ(cp, randGen);

	std::cout << "[INFO]: Translating MF GL" << std::endl;
	translateMFGL(cp);

	std::cout << "[INFO]: Translating GO and GL" << std::endl;
	translateGOGL(cp);

	std::cout << "[INFO]: Assigning GR delays" << std::endl;
	assignGRDelays(cp, msPerStep);

	std::cout << "[INFO]: Finished making innet connections." << std::endl;
}

InNetConnectivityState::InNetConnectivityState(ConnectivityParams *cp, std::fstream &infile)
{
	allocateMemory(cp);
	stateRW(cp, true, infile);
}

//InNetConnectivityState::InNetConnectivityState(const InNetConnectivityState &state)
//{
//	allocateMemory();
//
//	arrayCopy<int>(haspGLfromMFtoGL, state.haspGLfromMFtoGL, num_gl);
//	arrayCopy<int>(pGLfromMFtoGL, state.pGLfromMFtoGL, num_gl);
//
//	arrayCopy<int>(numpGLfromGLtoGO, state.numpGLfromGLtoGO, num_gl);
//	arrayCopy<int>(pGLfromGLtoGO[0], state.pGLfromGLtoGO[0],
//			num_gl*max_num_p_gl_from_gl_to_go);
//
//	arrayCopy<int>(numpGLfromGOtoGL, state.numpGLfromGOtoGL, num_gl);
//
//	arrayCopy<int>(numpGLfromGLtoGR, state.numpGLfromGLtoGR, num_gl);
//	arrayCopy<int>(pGLfromGLtoGR[0], state.pGLfromGLtoGR[0],
//			num_gl*maxnumpGLfromGOtoGL);
//
//	arrayCopy<int>(numpMFfromMFtoGL, state.numpMFfromMFtoGL, num_mf);
//	arrayCopy<int>(pMFfromMFtoGL[0], state.pMFfromMFtoGL[0],
//			num_mf*20);
//
//	arrayCopy<int>(numpMFfromMFtoGR, state.numpMFfromMFtoGR, num_mf);
//	arrayCopy<int>(pMFfromMFtoGR[0], state.pMFfromMFtoGR[0],
//			num_mf*maxnumpMFfromMFtoGR);
//
//	arrayCopy<int>(numpMFfromMFtoGO, state.numpMFfromMFtoGO, num_mf);
//	arrayCopy<int>(pMFfromMFtoGO[0], state.pMFfromMFtoGO[0],
//			num_mf*maxnumpMFfromMFtoGO);
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

void InNetConnectivityState::readState(ConnectivityParams *cp, std::fstream &infile)
{
	stateRW(cp, true, infile);
}

void InNetConnectivityState::writeState(ConnectivityParams *cp, std::fstream &outfile)
{
	stateRW(cp, false, outfile);
}

void InNetConnectivityState::allocateMemory(ConnectivityParams *cp)
{
	haspGLfromMFtoGL = new bool[cp->int_params["num_gl"]];

	numpGLfromGLtoGO = new int[cp->int_params["num_gl"]];
	pGLfromGLtoGO    = allocate2DArray<int>
	(
		cp->int_params["num_gl"],
		cp->int_params["max_num_p_gl_from_gl_to_go"]
	);

	numpGLfromGOtoGL = new int[cp->int_params["num_gl"]];
	pGLfromGOtoGL    = allocate2DArray<int>
	(
		cp->int_params["num_gl"],
		cp->int_params["max_num_p_gl_from_go_to_gl"]
	);

	numpGLfromGLtoGR = new int[cp->int_params["num_gl"]];
	pGLfromGLtoGR    = allocate2DArray<int>
	(
		cp->int_params["num_gl"],
		cp->int_params["max_num_p_gl_from_gl_to_gr"]
	);

	pGLfromMFtoGL    = new int[cp->int_params["num_gl"]];

	numpMFfromMFtoGL = new int[cp->int_params["num_mf"]];
	pMFfromMFtoGL    = allocate2DArray<int>
	(
		cp->int_params["num_mf"],
		cp->int_params["max_num_p_mf_from_mf_to_gl"]
	);

	numpMFfromMFtoGR = new int[cp->int_params["num_mf"]];
	pMFfromMFtoGR    = allocate2DArray<int>
	(
		cp->int_params["num_mf"],
		cp->int_params["max_num_p_mf_from_mf_to_gr"]
	);

	numpMFfromMFtoGO = new int[cp->int_params["num_mf"]];
	pMFfromMFtoGO    = allocate2DArray<int>
	(
		cp->int_params["num_mf"],
		cp->int_params["max_num_p_mf_from_mf_to_go"]
	);

	//golgi
	numpGOfromGLtoGO = new int[cp->int_params["num_go"]];
	pGOfromGLtoGO    = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["max_num_p_go_from_gl_to_go"]
	);

	numpGOfromGOtoGL = new int[cp->int_params["num_go"]];
	pGOfromGOtoGL    = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["max_num_p_go_from_go_to_gl"]
	);

	numpGOfromMFtoGO = new int[cp->int_params["num_go"]];
	pGOfromMFtoGO    = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["max_num_p_go_from_mf_to_go"]
	);

	numpGOfromGOtoGR = new int[cp->int_params["num_go"]];
	pGOfromGOtoGR    = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["max_num_p_go_from_go_to_gr"]
	);

	numpGOfromGRtoGO = new int[cp->int_params["num_go"]];
	pGOfromGRtoGO    = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["max_num_p_go_from_gr_to_go"]
	);

	// coincidentally, numcongotogo == maxnumpgogabaingogo
	numpGOGABAInGOGO  = new int[cp->int_params["num_go"]];
	pGOGABAInGOGO     = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["num_con_go_to_go"]
	);

	numpGOGABAOutGOGO = new int[cp->int_params["num_go"]];
	pGOGABAOutGOGO    = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["num_con_go_to_go"]
	);

	// go <-> go gap junctions
	numpGOCoupInGOGO = new int[cp->int_params["num_go"]];
	pGOCoupInGOGO    = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["num_p_go_to_go_gj"]
	);

	numpGOCoupOutGOGO = new int[cp->int_params["num_go"]];
	pGOCoupOutGOGO    = allocate2DArray<int>
	(
		cp->int_params["num_go"],
		cp->int_params["num_p_go_to_go_gj"]
	);

	pGOCoupOutGOGOCCoeff = allocate2DArray<float>
	(
		cp->int_params["num_go"],
		cp->int_params["num_p_go_to_go_gj"]
	);
	pGOCoupInGOGOCCoeff  = allocate2DArray<float>
	(
		cp->int_params["num_go"],
		cp->int_params["num_p_go_to_go_gj"]
	);

	//granule
	pGRDelayMaskfromGRtoBSP = new ct_uint32_t[cp->int_params["num_gr"]];

	numpGRfromGLtoGR = new int[cp->int_params["num_gr"]];
	pGRfromGLtoGR    = allocate2DArray<int>
	(
		cp->int_params["num_gr"],
		cp->int_params["max_num_p_gr_from_gl_to_gr"]
	);

	numpGRfromGRtoGO = new int[cp->int_params["num_gr"]];
	pGRfromGRtoGO    = allocate2DArray<int>
	(
		cp->int_params["num_gr"],
		cp->int_params["max_num_p_gr_from_gr_to_go"]
	);

	pGRDelayMaskfromGRtoGO  = allocate2DArray<int>
	(
		cp->int_params["num_gr"],
		cp->int_params["max_num_p_gr_from_gr_to_go"]
	);

	numpGRfromGOtoGR = new int[cp->int_params["num_gr"]];
	pGRfromGOtoGR    = allocate2DArray<int>
	(
		cp->int_params["num_gr"],
		cp->int_params["max_num_p_gr_from_go_to_gr"]
	);

	numpGRfromMFtoGR = new int[cp->int_params["num_gr"]];
	pGRfromMFtoGR    = allocate2DArray<int>
	(
		cp->int_params["num_gr"],
		cp->int_params["max_num_p_gr_from_mf_to_gr"]
	);
}

void InNetConnectivityState::initializeVals(ConnectivityParams *cp)
{
	std::fill(haspGLfromMFtoGL, haspGLfromMFtoGL + cp->int_params["num_gl"], false);

	std::fill(numpGLfromGLtoGO, numpGLfromGLtoGO + cp->int_params["num_gl"], 0);
	std::fill(pGLfromGLtoGO[0], pGLfromGLtoGO[0]
		+ cp->int_params["num_gl"] * cp->int_params["max_num_p_gl_from_gl_to_go"], 0);

	std::fill(numpGLfromGOtoGL, numpGLfromGOtoGL + cp->int_params["num_gl"], 0);
	std::fill(pGLfromGOtoGL[0], pGLfromGOtoGL[0]
		+ cp->int_params["num_gl"] * cp->int_params["max_num_p_gl_from_go_to_gl"], 0);

	std::fill(numpGLfromGLtoGR, numpGLfromGLtoGR + cp->int_params["num_gl"], 0);
	std::fill(pGLfromGLtoGR[0], pGLfromGLtoGR[0]
		+ cp->int_params["num_gl"] * cp->int_params["max_num_p_gl_from_gl_to_gr"], 0);

	std::fill(pGLfromMFtoGL, pGLfromMFtoGL + cp->int_params["num_gl"], 0);

	std::fill(numpMFfromMFtoGL, numpMFfromMFtoGL + cp->int_params["num_mf"], 0);
	std::fill(pMFfromMFtoGL[0], pMFfromMFtoGL[0]
		+ cp->int_params["num_mf"] * cp->int_params["max_num_p_mf_from_mf_to_gl"], 0);

	std::fill(numpMFfromMFtoGR, numpMFfromMFtoGR + cp->int_params["num_mf"], 0);
	std::fill(pMFfromMFtoGR[0], pMFfromMFtoGR[0]
		+ cp->int_params["num_mf"] * cp->int_params["max_num_p_mf_from_mf_to_gr"], 0);

	std::fill(numpMFfromMFtoGO, numpMFfromMFtoGO + cp->int_params["num_mf"], 0);
	std::fill(pMFfromMFtoGO[0], pMFfromMFtoGO[0]
		+ cp->int_params["num_mf"] * cp->int_params["max_num_p_mf_from_mf_to_go"], 0);

	std::fill(numpGOfromGLtoGO, numpGOfromGLtoGO + cp->int_params["num_go"], 0);
	std::fill(pGOfromGLtoGO[0], pGOfromGLtoGO[0]
		+ cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_gl_to_go"], 0);

	std::fill(numpGOfromGOtoGL, numpGOfromGOtoGL + cp->int_params["num_go"], 0);
	std::fill(pGOfromGOtoGL[0], pGOfromGOtoGL[0]
		+ cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_go_to_gl"], 0);

	std::fill(numpGOfromMFtoGO, numpGOfromMFtoGO + cp->int_params["num_go"], 0);
	std::fill(pGOfromMFtoGO[0], pGOfromMFtoGO[0]
		+ cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_mf_to_go"], 0);

	std::fill(numpGOfromGOtoGR, numpGOfromGOtoGR + cp->int_params["num_go"], 0);
	std::fill(pGOfromGOtoGR[0], pGOfromGOtoGR[0]
		+ cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_go_to_gr"], 0);

	std::fill(numpGOfromGRtoGO, numpGOfromGRtoGO + cp->int_params["num_go"], 0);
	std::fill(pGOfromGRtoGO[0], pGOfromGRtoGO[0]
		+ cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_gr_to_go"], 0);

	std::fill(numpGOGABAInGOGO, numpGOGABAInGOGO + cp->int_params["num_go"], 0);
	std::fill(pGOGABAInGOGO[0], pGOGABAInGOGO[0]
		+ cp->int_params["num_go"] * cp->int_params["num_con_go_to_go"], INT_MAX);

	std::fill(numpGOGABAOutGOGO, numpGOGABAOutGOGO + cp->int_params["num_go"], 0);
	std::fill(pGOGABAOutGOGO[0], pGOGABAOutGOGO[0]
		+ cp->int_params["num_go"] * cp->int_params["num_con_go_to_go"], INT_MAX);

	std::fill(numpGOCoupInGOGO, numpGOCoupInGOGO + cp->int_params["num_go"], 0);
	std::fill(pGOCoupInGOGO[0], pGOCoupInGOGO[0]
		+ cp->int_params["num_go"] * cp->int_params["num_p_go_to_go_gj"], 0);

	std::fill(numpGOCoupOutGOGO, numpGOCoupOutGOGO + cp->int_params["num_go"], 0);
	std::fill(pGOCoupOutGOGO[0], pGOCoupOutGOGO[0]
		+ cp->int_params["num_go"] * cp->int_params["num_p_go_to_go_gj"], 0);

	std::fill(pGOCoupOutGOGOCCoeff[0], pGOCoupOutGOGOCCoeff[0]
		+ cp->int_params["num_go"] * cp->int_params["num_p_go_to_go_gj"], 0.0);

	std::fill(pGOCoupInGOGOCCoeff[0], pGOCoupInGOGOCCoeff[0]
		+ cp->int_params["num_go"] * cp->int_params["num_p_go_to_go_gj"], 0.0);

	std::fill(pGRDelayMaskfromGRtoBSP, pGRDelayMaskfromGRtoBSP + cp->int_params["num_gr"], 0);

	std::fill(numpGRfromGLtoGR, numpGRfromGLtoGR + cp->int_params["num_gr"], 0);
	std::fill(pGRfromGLtoGR[0], pGRfromGLtoGR[0]
		+ cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_gl_to_gr"], 0);

	std::fill(numpGRfromGRtoGO, numpGRfromGRtoGO + cp->int_params["num_gr"], 0);
	std::fill(pGRfromGRtoGO[0], pGRfromGRtoGO[0]
		+ cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_gr_to_go"], 0);

	std::fill(pGRDelayMaskfromGRtoGO[0], pGRDelayMaskfromGRtoGO[0]
		+ cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_gr_to_go"], 0);

	std::fill(numpGRfromGOtoGR, numpGRfromGOtoGR + cp->int_params["num_gr"], 0);
	std::fill(pGRfromGOtoGR[0], pGRfromGOtoGR[0]
		+ cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_go_to_gr"], 0);

	std::fill(numpGRfromMFtoGR, numpGRfromMFtoGR + cp->int_params["num_gr"], 0);
	std::fill(pGRfromMFtoGR[0], pGRfromMFtoGR[0]
		+ cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_mf_to_gr"], 0);
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

void InNetConnectivityState::stateRW(ConnectivityParams *cp, bool read, std::fstream &file)
{
	//glomerulus
	rawBytesRW((char *)haspGLfromMFtoGL, cp->int_params["num_gl"] * sizeof(bool), read, file);
	rawBytesRW((char *)numpGLfromGLtoGO, cp->int_params["num_gl"] * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGO[0],
		cp->int_params["num_gl"] * cp->int_params["max_numpGLfromGLtoGO"] * sizeof(int), read, file);
	rawBytesRW((char *)numpGLfromGOtoGL, cp->int_params["num_gl"] * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGOtoGL[0],
		cp->int_params["num_gl"] * cp->int_params["max_num_p_gl_from_go_to_gl"] * sizeof(int), read, file);
	rawBytesRW((char *)numpGLfromGLtoGR, cp->int_params["num_gl"] * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGR[0],
		cp->int_params["num_gl"] * cp->int_params["max_num_p_gl_from_gl_to_gr"] * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromMFtoGL, cp->int_params["num_gl"] * sizeof(int), read, file);

	//mossy fibers
	rawBytesRW((char *)numpMFfromMFtoGL, cp->int_params["num_mf"] * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGL[0],
		cp->int_params["num_mf"] * cp->int_params["max_num_p_mf_from_mf_to_gl"] * sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGR, cp->int_params["num_mf"] * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGR[0],
		cp->int_params["num_mf"] * cp->int_params["max_num_p_mf_from_mf_to_gr"] * sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGO, cp->int_params["num_mf"] * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGO[0],
		cp->int_params["num_mf"] * cp->int_params["max_num_p_mf_from_mf_to_go"] * sizeof(int), read, file);

	//golgi
	rawBytesRW((char *)numpGOfromGLtoGO, cp->int_params["num_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGLtoGO[0],
		cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_gl_to_go"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGL, cp->int_params["num_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGL[0],
		cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_go_to_gl"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromMFtoGO, cp->int_params["num_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromMFtoGO[0],
		cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_mf_to_go"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGR, cp->int_params["num_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGR[0],
		cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_go_to_gr"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGRtoGO, cp->int_params["num_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGRtoGO[0],
		cp->int_params["num_go"] * cp->int_params["max_num_p_go_from_gr_to_go"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAInGOGO, cp->int_params["num_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAInGOGO[0],
		cp->int_params["num_go"] * cp->int_params["num_con_go_to_go"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAOutGOGO, cp->int_params["num_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAOutGOGO[0],
		cp->int_params["num_go"] * cp->int_params["num_con_go_to_go"] * sizeof(int), read, file);
	
	rawBytesRW((char *)numpGOCoupInGOGO, cp->int_params["num_go"]*sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupInGOGO[0],
		cp->int_params["num_go"] * cp->int_params["num_p_go_to_go_gj"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGOCoupOutGOGO, cp->int_params["num_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupOutGOGO[0],
		cp->int_params["num_go"] * cp->int_params["num_p_go_to_go_gj"] * sizeof(int), read, file);

	rawBytesRW((char *)pGOCoupOutGOGOCCoeff[0],
		cp->int_params["num_go"] * cp->int_params["num_p_go_to_go_gj"] * sizeof(float), read, file);
	rawBytesRW((char *)pGOCoupInGOGOCCoeff[0],
		cp->int_params["num_go"] * cp->int_params["num_p_go_to_go_gj"] * sizeof(float), read, file);

	//granule
	rawBytesRW((char *)pGRDelayMaskfromGRtoBSP, cp->int_params["num_gr"] * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromGLtoGR, cp->int_params["num_gr"] * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGLtoGR[0],
		cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_gl_to_gr"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGRtoGO, cp->int_params["num_gr"] * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGRtoGO[0],
		cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_gr_to_go"] * sizeof(int), read, file);
	rawBytesRW((char *)pGRDelayMaskfromGRtoGO[0],
		cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_gr_to_go"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGOtoGR, cp->int_params["num_gr"] * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGOtoGR[0],
		cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_go_to_gr"] * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromMFtoGR, cp->int_params["num_gr"] * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromMFtoGR[0],
		cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_mf_to_gr"] * sizeof(int), read, file);
}


void InNetConnectivityState::connectMFGL_noUBC(ConnectivityParams *cp)
{
	// shorter convenience var names
	int glX = cp->int_params["gl_x"];
	int glY = cp->int_params["gl_y"];
	int mfX = cp->int_params["mf_x"];
	int mfY = cp->int_params["mf_y"];

	// define span and coord arrays locally
	int spanArrayMFtoGLX[cp->int_params["span_mf_to_gl_x"] + 1] = {0};
	int spanArrayMFtoGLY[cp->int_params["span_mf_to_gl_y"] + 1] = {0};
	int xCoorsMFGL[cp->int_params["num_p_mf_to_gl"]] = {0};
	int yCoorsMFGL[cp->int_params["num_p_mf_to_gl"]] = {0};

	// fill span arrays and coord arrays
	for (int i = 0; i < cp->int_params["span_mf_to_gl_x"] + 1; i++)
	{
		spanArrayMFtoGLX[i] = i - (cp->int_params["span_mf_to_gl_x"] / 2);
	}

	for (int i = 0; i < cp->int_params["span_mf_to_gl_y"] + 1; i++)
	{
		spanArrayMFtoGLY[i] = i - (cp->int_params["span_mf_to_gl_y"] / 2);
	}
		
	for (int i = 0; i < cp->int_params["num_p_mf_to_gl"]; i++)
	{
		xCoorsMFGL[i] = spanArrayMFtoGLX[i % (cp->int_params["span_mf_to_gl_x"] + 1)];
		yCoorsMFGL[i] = spanArrayMFtoGLY[i / (cp->int_params["span_mf_to_gl_y"] + 1)];
	}

	// scale factors from one cell coord to another
	float gridXScaleSrctoDest = (float)mfX / (float)glX; 
	float gridYScaleSrctoDest = (float)mfY/ (float)glY; 

	// random mf index array, supposedly to even out distribution of connections
	int rMFInd[cp->int_params["num_mf"]] = {0};	
	for (int i = 0; i < cp->int_params["num_mf"]; i++) rMFInd[i] = i;	
	std::random_shuffle(rMFInd, rMFInd + cp->int_params["num_mf"]);

	// fill random span array with linear indices
	int rMFSpanInd[cp->int_params["num_p_mf_to_gl"]] = {0};
	for (int i = 0; i < cp->int_params["num_p_mf_to_gl"]; i++) rMFSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	// attempt to make connections
	for (int attempts = 0; attempts < cp->int_params["max_mf_to_gl_attempts"]; attempts++)
	{
		std::random_shuffle(rMFInd, rMFInd + cp->int_params["num_mf"]);	
		// for each attempt, loop through all presynaptic cells
		for (int i = 0; i < cp->int_params["num_mf"]; i++)
		{
			//Select MF Coordinates from random index array: Complete	
			srcPosX = rMFInd[i] % mfX;
			srcPosY = rMFInd[i] / mfX;
			
			std::random_shuffle(rMFSpanInd, rMFSpanInd + cp->int_params["num_p_mf_to_gl"]);
			// for each presynaptic cell, attempt to make up to initial output + max attempts
			// connections.	
			for (int j = 0; j < cp->int_params["num_p_mf_to_gl"]; j++)
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
				if (numpMFfromMFtoGL[rMFInd[i]] == (cp->int_params["initial_mf_output"] + attempts)) break;
				
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
	for (int i = 0; i < cp->int_params["num_mf"]; i++) count += numpMFfromMFtoGL[i];
	
	std::cout << "Total number of Mossy Fiber to Glomeruli connections: " << count << std::endl;
	std::cout << "Correct number: " << cp->int_params["num_gl"] << std::endl;
}

void InNetConnectivityState::connectGLGR(ConnectivityParams *cp, CRandomSFMT &randGen)
{
	int grX = cp->int_params["gr_x"];
	// int grY = GR_Y; /* unused */
	int glX = cp->int_params["gl_x"];
	int glY = cp->int_params["gl_y"];

	float gridXScaleStoD = (float)grX / (float)glX;
	// float gridYScaleStoD = (float)grY / (float)glY; /* unused :/ */

	bool srcConnected[cp->int_params["num_gr"]] = {false};

	for (int i = 0; i < cp->int_params["max_num_p_gr_from_gl_to_gr"]; i++)
	{
		int srcNumConnected = 0;
		std::fill(srcConnected, srcConnected + cp->int_params["num_gr"], false);

		while (srcNumConnected < cp->int_params["num_gr"])
		{
			int srcIndex = randGen.IRandom(0, cp->int_params["num_gr"] - 1);
			if (!srcConnected[srcIndex])
			{
				int srcPosX = srcIndex % grX;
				int srcPosY = (int)(srcIndex / grX);

				int tempDestNumConLim = cp->int_params["low_num_p_gl_from_gl_to_gr"];

				for (int attempts = 0; attempts < cp->int_params["max_gl_to_gr_attempts"]; attempts++)
				{
					if (attempts == cp->int_params["low_gl_to_gr_attempts"])
					{
						tempDestNumConLim = cp->int_params["max_num_p_gl_from_gl_to_gr"];
					}

					int destPosX = (int)round(srcPosX / gridXScaleStoD);
					int destPosY = (int)round(srcPosY / gridXScaleStoD);

					// again, should add 1 to spans
					destPosX += round((randGen.Random() - 0.5) * cp->int_params["span_gl_to_gr_x"]);
					destPosY += round((randGen.Random() - 0.5) * cp->int_params["span_gl_to_gr_y"]);

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
	for (int i = 0; i < cp->int_params["num_gl"]; i++)
	{
		count += numpGLfromGLtoGR[i];
	}

	std::cout << "Total number of Glomeruli to Granule connections: " << count << std::endl; 
	std::cout << "Correct number: " << cp->int_params["num_gr"] * cp->int_params["max_num_p_gr_from_gl_to_gr"] << std::endl;
	// for now, no empty counter
}

void InNetConnectivityState::connectGRGO(ConnectivityParams *cp)
{
	int grX = cp->int_params["gr_x"];
	int grY = cp->int_params["gr_y"];
	int numGR = cp->int_params["num_gr"];

	int goX = cp->int_params["go_x"];
	int goY = cp->int_params["go_y"];
	int numGO = cp->int_params["num_go"];

	int spanPFtoGOX = cp->int_params["span_pf_to_gox"];
	int spanPFtoGOY = cp->int_params["span_pf_to_goy"];
	int numPPFtoGO = cp->int_params["num_p_pf_to_go"];

	int maxPFtoGOAttempts = cp->int_params["max_pf_to_go_attempts"];
	int maxPFtoGOInput = cp->int_params["max_pf_to_go_input"];

	int spanArrayPFtoGOX[spanPFtoGOX + 1] = {0};
	int spanArrayPFtoGOY[spanPFtoGOY + 1] = {0};
	int xCoorsPFGO[numPPFtoGO] = {0};
	int yCoorsPFGO[numPPFtoGO] = {0};

	//PARALLEL FIBER TO GOLGI 
	for (int i = 0; i < spanPFtoGOX + 1; i++)
	{
		spanArrayPFtoGOX[i] = i - (spanPFtoGOX / 2);
	}

	for (int i = 0; i < spanPFtoGOY + 1; i++)
	{
		spanArrayPFtoGOY[i] = i - (spanPFtoGOY / 2);
	}
		
	for (int i = 0; i < numPPFtoGO; i++)
	{
		xCoorsPFGO[i] = spanArrayPFtoGOX[i % (spanPFtoGOX + 1)];
		yCoorsPFGO[i] = spanArrayPFtoGOY[i / (spanPFtoGOX + 1)];
	}

	float gridXScaleSrctoDest = (float)goX / (float)grX; 
	float gridYScaleSrctoDest = (float)goY / (float)grY; 

	int rPFSpanInd[numPPFtoGO] = {0};
	for (int i = 0; i < numPPFtoGO; i++) rPFSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int attempts = 0; attempts < maxPFtoGOAttempts; attempts++)
	{
		for (int i = 0; i < numGO; i++)
		{
			srcPosX = i % goX;
			srcPosY = i / goX;

			std::random_shuffle(rPFSpanInd, rPFSpanInd + numPPFtoGO);
			for (int j = 0; j < maxPFtoGOInput; j++)
			{
				destPosX = xCoorsPFGO[rPFSpanInd[j]]; 
				destPosY = yCoorsPFGO[rPFSpanInd[j]];
				
				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);
				
				destPosX = (destPosX % grX + grX) % grX;
				destPosY = (destPosY % grY + grY) % grY;

				destIndex = destPosY * grX + destPosX;

				if (numpGOfromGRtoGO[i] < maxPFtoGOInput)
				{
					pGOfromGRtoGO[i][numpGOfromGRtoGO[i]] = destIndex;
					numpGOfromGRtoGO[i]++;

					pGRfromGRtoGO[destIndex][numpGRfromGRtoGO[destIndex]] = i;
					numpGRfromGRtoGO[destIndex]++;
				}
			}
		}
	}

	int spanAAtoGOX = cp->int_params["span_aa_to_gox"];
	int spanAAtoGOY = cp->int_params["span_aa_to_goy"];
	int numPAAtoGO = cp->int_params["num_p_aa_to_go"];

	int maxAAtoGOAttempts = cp->int_params["max_aa_to_go_attempts"];
	int maxAAtoGOInput = cp->int_params["max_aa_to_go_input"];

	int spanArrayAAtoGOX[spanAAtoGOX + 1] = {0};
	int spanArrayAAtoGOY[spanAAtoGOY + 1] = {0};
	int xCoorsAAGO[numPAAtoGO] = {0};
	int yCoorsAAGO[numPAAtoGO] = {0};

	for (int i = 0; i < spanAAtoGOX + 1; i++)
	{
		spanArrayAAtoGOX[i] = i - (spanAAtoGOX / 2);
	}

	for (int i = 0; i < spanAAtoGOY + 1; i++)
	{
		spanArrayAAtoGOY[i] = i - (spanAAtoGOY / 2);
	}
		
	for (int i = 0; i < numPAAtoGO; i++)
	{
		xCoorsAAGO[i] = spanArrayAAtoGOX[i % (spanAAtoGOX + 1)];
		yCoorsAAGO[i] = spanArrayAAtoGOY[i / (spanAAtoGOX + 1)];
	}
	
	int rAASpanInd[numPAAtoGO] = {0};
	for (int i = 0; i < numPAAtoGO; i++) rAASpanInd[i] = i;

	for (int attempts = 0; attempts < maxAAtoGOAttempts; attempts++)
	{
		for (int i = 0; i < numGO; i++)
		{
			srcPosX = i % goX;
			srcPosY = i / goX;

			std::random_shuffle(rAASpanInd, rAASpanInd + numPAAtoGO);
			
			for (int j = 0; j < maxAAtoGOInput; j++)
			{
				destPosX = xCoorsAAGO[rAASpanInd[j]]; 
				destPosY = yCoorsAAGO[rAASpanInd[j]];
				
				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);

				destPosX = (destPosX % grX + grX) % grX;
				destPosY = (destPosY % grY + grY) % grY;
						
				destIndex = destPosY * grX + destPosX;
				
				if (numpGOfromGRtoGO[i] < maxAAtoGOInput + maxPFtoGOInput)
				{
					pGOfromGRtoGO[i][numpGOfromGRtoGO[i]] = destIndex;
					numpGOfromGRtoGO[i]++;

					pGRfromGRtoGO[destIndex][numpGRfromGRtoGO[destIndex]] = i;
					numpGRfromGRtoGO[destIndex]++;
				}
			}
		}
	}

	int gr_go_input_sum = 0;
	
	for (int i = 0; i < numGO; i++)
	{
		gr_go_input_sum += numpGOfromGRtoGO[i];
	}

	std::cout << "[INFO]: Total number of granule to golgi inputs: " << gr_go_input_sum << std::endl;

	int gr_go_output_sum = 0;

	for (int i = 0; i < numGR; i++)
	{
		gr_go_output_sum += numpGRfromGRtoGO[i];
	}

	std::cout << "[INFO]: Total number of granule to golgi outputs: " << gr_go_output_sum << std::endl;
}

void InNetConnectivityState::connectGOGL(ConnectivityParams *cp, CRandomSFMT &randGen)
{
	// using old connectivity alg for now , cannot generalize (do not always know
	// at least both array bounds for 2D arrays at compile time)
	// gl -> go
	int goX = cp->int_params["go_x"];
	int goY = cp->int_params["go_y"];
	int numGO = cp->int_params["num_go"];

	int glX = cp->int_params["gl_x"];
	int glY = cp->int_params["gl_y"];
	int numGL = cp->int_params["num_gl"];

	int spanGLtoGOX = cp->int_params["span_gl_to_go_x"];
	int spanGLtoGOY = cp->int_params["span_gl_to_go_y"];

	int maxNumPGOfromGLtoGO = cp->int_params["max_num_p_go_from_gl_to_go"];
	int lowNumPGLfromGLtoGO = cp->int_params["low_num_p_gl_from_gl_to_go"];

	int maxNumPGLfromGLtoGO = cp->int_params["max_num_p_gl_from_gl_to_go"];

	int maxGLtoGOAttempts = cp->int_params["max_gl_to_go_attempts"];
	int lowGLtoGOAttempts = cp->int_params["low_gl_to_go_attempts"];

	float gridXScaleSrctoDest = (float)goX / (float)glX;
	float gridYScaleSrctoDest = (float)goY / (float)glY;

	bool srcConnected[numGO] = {false};

	for (int i = 0; i < maxNumPGOfromGLtoGO; i++)
	{
		int srcNumConnected = 0;
		while (srcNumConnected < numGO)
		{
			int srcIndex = randGen.IRandom(0, numGO - 1);
			if (!srcConnected[srcIndex])
			{
				int srcPosX = srcIndex % goX;
				int srcPosY = (int)(srcIndex / goX);

				int tempDestNumConLim = lowNumPGLfromGLtoGO;

				for (int attempts = 0; attempts < maxGLtoGOAttempts; attempts++)
				{
					int destPosX;
					int destPosY;
					int destIndex;

					if (attempts == lowGLtoGOAttempts)
					{
						tempDestNumConLim = maxNumPGLfromGLtoGO;
					}

					destPosX = (int)round(srcPosX / gridXScaleSrctoDest);
					destPosY = (int)round(srcPosY / gridXScaleSrctoDest);

					// should multiply spans by 1 for full coverage
					destPosX += round((randGen.Random() - 0.5) * spanGLtoGOX);
					destPosY += round((randGen.Random() - 0.5) * spanGLtoGOY);

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
		std::fill(srcConnected, srcConnected + numGO, false);
	}

	std::cout << "[INFO]: Finished making gl go connections." << std::endl;
	std::cout << "[INFO]: Starting on go gl connections..." << std::endl;

	// go --> gl

	int spanGOtoGLX = cp->int_params["span_go_to_gl_x"];
	int spanGOtoGLY = cp->int_params["span_go_to_gl_y"];
	int numPGOtoGL  = cp->int_params["num_p_go_to_gl"];

	float amplGOtoGL     = cp->float_params["ampl_go_to_gl"];
	float stdDevGOtoGLML = cp->float_params["std_dev_go_to_gl_ml"];
	float stdDevGOtoGLS  = cp->float_params["std_dev_go_to_gl_s"];

	//int maxNumPGOfromGLtoGO = cp->int_params["max_num_p_go_from_gl_to_go"];
	//int lowNumPGLfromGLtoGO = cp->int_params["low_num_p_gl_from_gl_to_go"];

	int maxGOtoGLAttempts = cp->int_params["max_go_to_gl_attempts"];
	int lowGOtoGLAttempts = cp->int_params["low_go_to_gl_attempts"];

	int initialGOInput = cp->int_params["initial_go_input"];
	int maxNumPGLfromGOtoGL = cp->int_params["max_num_p_gl_from_go_to_gl"];

	int spanArrayGOtoGLX[spanGOtoGLX + 1] = {0};
	int spanArrayGOtoGLY[spanGOtoGLY + 1] = {0};
	int xCoorsGOGL[numPGOtoGL] = {0};
	int yCoorsGOGL[numPGOtoGL] = {0};
	float pConGOGL[numPGOtoGL] = {0.0};

	// Make span Array
	for (int i = 0; i < spanGOtoGLX + 1; i++)
	{
		spanArrayGOtoGLX[i] = i - (spanGOtoGLX / 2);
	}

	for (int i = 0; i < spanGOtoGLY + 1; i++)
	{
		spanArrayGOtoGLY[i] = i - (spanGOtoGLY / 2);
	}
		
	for (int i = 0; i < numPGOtoGL; i++)
	{
		xCoorsGOGL[i] = spanArrayGOtoGLX[i % (spanGOtoGLX + 1)];
		yCoorsGOGL[i] = spanArrayGOtoGLY[i / (spanGOtoGLX + 1)];
	}

	// Probability of connection as a function of distance
	for (int i = 0; i < numPGOtoGL; i++)
	{
		float PconX = (xCoorsGOGL[i] * xCoorsGOGL[i])
			/ (2 * stdDevGOtoGLML * stdDevGOtoGLML);
		float PconY = (yCoorsGOGL[i] * yCoorsGOGL[i])
			/ (2 * stdDevGOtoGLS * stdDevGOtoGLS);
		pConGOGL[i] = amplGOtoGL * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < numPGOtoGL; i++)
	{
		if ((xCoorsGOGL[i] == 0) && (yCoorsGOGL[i] == 0)) pConGOGL[i] = 0;
	}

	//Make Random Golgi cell Index Array
	int rGOInd[numGO] = {0};
	for (int i = 0; i < numGO; i++) rGOInd[i] = i;

	//Make Random Span Array
	int rGOSpanInd[numPGOtoGL] = {0};
	for (int i = 0; i < numPGOtoGL; i++) rGOSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int attempts = 0; attempts < maxGOtoGLAttempts; attempts++)
	{
		std::random_shuffle(rGOInd, rGOInd + numGO);
		
		// Go through each golgi cell 
		for (int i = 0; i < numGO; i++)
		{
			//Select GO Coordinates from random index array: Complete
			srcPosX = rGOInd[i] % goX;
			srcPosY = rGOInd[i] / goX;
			
			std::random_shuffle(rGOSpanInd, rGOSpanInd + numPGOtoGL);
			
			for (int j = 0; j < numPGOtoGL; j++)   
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
					
				if (numpGOfromGOtoGL[rGOInd[i]] >= initialGOInput + attempts) break; 
				if (randGen.Random() >= 1 - pConGOGL[rGOSpanInd[j]] && 
						numpGLfromGOtoGL[destIndex] < maxNumPGLfromGOtoGL) 
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

	for (int i = 0; i < numGL; i++)
	{
		if (numpGLfromGOtoGL[i] < maxNumPGLfromGOtoGL) shitCounter++;
		totalGOGL += numpGLfromGOtoGL[i];
	}

	std::cout << "Empty Glomeruli Counter: " << shitCounter << std::endl;
	std::cout << "Total GO -> GL: " << totalGOGL << std::endl;
	std::cout << "avg Num  GO -> GL Per GL: " << (float)totalGOGL / (float)numGL << std::endl;
}

void InNetConnectivityState::connectGOGODecayP(ConnectivityParams *cp, CRandomSFMT &randGen)
{
	int goX = cp->int_params["go_x"];
	int goY = cp->int_params["go_y"];

	int spanArrayGOtoGOsynX[cp->int_params["span_go_to_go_x"] + 1] = {0};
	int spanArrayGOtoGOsynY[cp->int_params["span_go_to_go_y"] + 1] = {0};
	int xCoorsGOGOsyn[cp->int_params["num_p_go_to_go"]] = {0};
	int yCoorsGOGOsyn[cp->int_params["num_p_go_to_go"]] = {0};
	float Pcon[cp->int_params["num_p_go_to_go"]] = {0};

	bool **conGOGOBoolOut = allocate2DArray<bool>
	(
		cp->int_params["num_go"],
		cp->int_params["num_go"]
	);
	// TODO: change to memset - is faster
	std::fill(conGOGOBoolOut[0], conGOGOBoolOut[0] + cp->int_params["num_go"] * cp->int_params["num_go"], false);

	for (int i = 0; i < cp->int_params["span_go_to_go_x"] + 1; i++)
   	{
		spanArrayGOtoGOsynX[i] = i - (cp->int_params["span_go_to_go_x"] / 2);
	}

	for (int i = 0; i < cp->int_params["span_go_to_go_y"] + 1; i++)
   	{
		spanArrayGOtoGOsynY[i] = i - (cp->int_params["span_go_to_go_y"] / 2);
	}
		
	for (int i = 0; i < cp->int_params["num_p_go_to_go"]; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (SPAN_GO_TO_GO_X + 1)];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (SPAN_GO_TO_GO_X + 1)];
	}

	for (int i = 0; i < cp->int_params["num_p_go_to_go"]; i++)
	{
		float PconX = (xCoorsGOGOsyn[i] * xCoorsGOGOsyn[i])
			/ (2 * (cp->float_params["std_dev_go_to_go"] * cp->float_params["std_dev_go_to_go"]));
		float PconY = (yCoorsGOGOsyn[i] * yCoorsGOGOsyn[i])
			/ (2 * (cp->float_params["std_dev_go_to_go"] * cp->float_params["std_dev_go_to_go"]));
		Pcon[i] = cp->float_params["ampl_go_to_go"] * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < cp->int_params["num_p_go_to_go"]; i++) 
	{
		if ((xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0)) Pcon[i] = 0;
	}
	
	int rGOGOSpanInd[cp->int_params["num_p_go_to_go"]] = {0};
	for (int i = 0; i < cp->int_params["num_p_go_to_go"]; i++) rGOGOSpanInd[i] = i;

	for (int attempts = 0; attempts < cp->int_params["max_go_to_go_attempts"]; attempts++) 
	{
		for (int i = 0; i < cp->int_params["num_go"]; i++) 
		{
			int srcPosX = i % goX;
			int srcPosY = i / goX;
			
			std::random_shuffle(rGOGOSpanInd, rGOGOSpanInd + cp->int_params["num_p_go_to_go"]);
			
			for (int j = 0; j < cp->int_params["num_p_go_to_go"]; j++)
		   	{	
				int destPosX = srcPosX + xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
				int destPosY = srcPosY + yCoorsGOGOsyn[rGOGOSpanInd[j]];

				destPosX = (destPosX % goX + goX) % goX;
				destPosY = (destPosY % goY + goY) % goY;
						
				int destIndex = destPosY * goX + destPosX;
			
				// Normal One	
				if ((bool)cp->int_params["go_go_recip_cons"]
					&& !(bool)cp->int_params["reduce_base_recip_go_go"]
					&& randGen.Random()>= 1 - Pcon[rGOGOSpanInd[j]]
					&& !conGOGOBoolOut[i][destIndex]
					&& numpGOGABAOutGOGO[i] < cp->int_params["num_con_go_to_go"] 
					&& numpGOGABAInGOGO[destIndex] < cp->int_params["num_con_go_to_go"]) 
				{
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;
							
					if (randGen.Random() <= cp->float_params["p_recip_go_go"]
							&& !conGOGOBoolOut[destIndex][i]
							&& numpGOGABAOutGOGO[destIndex] < cp->int_params["num_con_go_to_go"] 
							&& numpGOGABAInGOGO[i] < cp->int_params["num_con_go_to_go"]) 
					{
						pGOGABAOutGOGO[destIndex][numpGOGABAOutGOGO[destIndex]] = i;
						numpGOGABAOutGOGO[destIndex]++;
						
						pGOGABAInGOGO[i][numpGOGABAInGOGO[i]] = destIndex;
						numpGOGABAInGOGO[i]++;
						
						conGOGOBoolOut[destIndex][i] = true;
					}
				}
			
				if ((bool)cp->int_params["go_go_recip_cons"]
					&& (bool)cp->int_params["reduce_base_recip_go_go"]
					&& randGen.Random() >= 1 - Pcon[rGOGOSpanInd[j]]
					&& !conGOGOBoolOut[i][destIndex] && (!conGOGOBoolOut[destIndex][i]
					|| randGen.Random() <= cp->float_params["p_recip_lower_base_go_go"])
					&& numpGOGABAOutGOGO[i] < cp->int_params["num_con_go_to_go"] 
					&& numpGOGABAInGOGO[destIndex] < cp->int_params["num_con_go_to_go"]) 
				{	
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;	
				}

				if (!(bool)cp->int_params["go_go_recip_cons"]
					&& !(bool)cp->int_params["reduce_base_recip_go_go"]
					&& randGen.Random() >= 1 - Pcon[rGOGOSpanInd[j]]
					&& (!conGOGOBoolOut[i][destIndex]) && !conGOGOBoolOut[destIndex][i]
					&& numpGOGABAOutGOGO[i] < cp->int_params["num_con_go_to_go"] 
					&& numpGOGABAInGOGO[destIndex] < cp->int_params["num_con_go_to_go"])
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

	for (int i = 0; i < cp->int_params["num_go"]; i++)
	{
		totalGOGOcons += numpGOGABAInGOGO[i];
	}

	std::cout << "Total GOGO connections: " << totalGOGOcons << std::endl;
	std::cout << "Average GOGO connections:	" << (float)totalGOGOcons / float(cp->int_params["num_go"]) << std::endl;
	std::cout << cp->int_params["num_go"] << std::endl;

	int recipCounter = 0;

	for (int i = 0; i < cp->int_params["num_go"]; i++)
	{
		for (int j = 0; j < numpGOGABAInGOGO[i]; j++)
		{
			for (int k = 0; k < numpGOGABAOutGOGO[i]; k++)
			{
				if (pGOGABAInGOGO[i][j] == pGOGABAOutGOGO[i][k]
					&& pGOGABAInGOGO[i][j] != INT_MAX 
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

void InNetConnectivityState::connectGOGO_GJ(ConnectivityParams *cp, CRandomSFMT &randGen)
{
	int goX = cp->int_params["go_x"];
	int goY = cp->int_params["go_y"];

	int spanArrayGOtoGOgjX[cp->int_params["span_go_to_go_gj_x"] + 1] = {0};
	int spanArrayGOtoGOgjY[cp->int_params["span_go_to_go_gj_y"] + 1] = {0};
	int xCoorsGOGOgj[cp->int_params["num_p_go_to_go_gj"]] = {0};
	int yCoorsGOGOgj[ cp->int_params["num_p_go_to_go_gj"]] = {0};

	float gjPCon[cp->int_params["num_p_go_to_go_gj"]] = {0.0};
	float gjCC[cp->int_params["num_p_go_to_go_gj"]] = {0.0};

	bool **gjConBool = allocate2DArray<bool>
	(
		cp->int_params["num_go"],
		cp->int_params["num_go"]
	);
	std::fill(gjConBool[0], gjConBool[0] + cp->int_params["num_go"] * cp->int_params["num_go"], false);

	for (int i = 0; i < cp->int_params["span_go_to_go_gj_x"] + 1; i++)
	{
		spanArrayGOtoGOgjX[i] = i - (cp->int_params["span_go_to_go_gj_x"] / 2);
	}

	for (int i = 0; i < cp->int_params["span_go_to_go_gj_y"] + 1; i++)
	{
		spanArrayGOtoGOgjY[i] = i - (cp->int_params["span_go_to_go_gj_y"] / 2);
	}

	for (int i = 0; i < cp->int_params["num_p_go_to_go_gj"]; i++)
	{
		xCoorsGOGOgj[i] = spanArrayGOtoGOgjX[i % (cp->int_params["span_go_to_go_gj_x"] + 1)];
		yCoorsGOGOgj[i] = spanArrayGOtoGOgjY[i / (cp->int_params["span_go_to_go_gj_x"] + 1)];
	}

	// "In Vivo additions"
	for (int i = 0; i < cp->int_params["num_p_go_to_go_gj"]; i++)
	{
		float gjPConX = exp(((abs(xCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );	
		float gjPConY = exp(((abs(yCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );
		gjPCon[i] = ((-1745.0 + (1836.0 / (1 + (gjPConX + gjPConY)))) * 0.01);
		
		float gjCCX = exp(abs(xCoorsGOGOgj[i] * 100.0) / 190.0);
		float gjCCY = exp(abs(yCoorsGOGOgj[i] * 100.0) / 190.0);
		gjCC[i] = (-2.3 + (23.0 / ((gjCCX + gjCCY)/2.0))) * 0.09;
	}

	// Remove self connection 
	for (int i = 0; i < cp->int_params["num_p_go_to_go_gj"]; i++)
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

	for (int i = 0; i < cp->int_params["num_go"]; i++)
	{
		srcPosX = i % goX;
		srcPosY = i / goX;
		
		for (int j = 0; j < cp->int_params["num_p_go_to_go_gj"]; j++)
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

void InNetConnectivityState::translateMFGL(ConnectivityParams *cp)
{

	// Mossy fiber to Granule
	
	for (int i = 0; i < cp->int_params["num_gr"]; i++)
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
	
	for (int i = 0; i < cp->int_params["num_gr"]; i++)
	{
		grMFInputCounter += numpGRfromMFtoGR[i];
	}

	std::cout << "Total MF inputs: " << grMFInputCounter << std::endl;

	// Mossy fiber to Golgi	
	
	for (int i = 0; i < cp->int_params["num_go"]; i++)
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

void InNetConnectivityState::translateGOGL(ConnectivityParams *cp)
{
	for (int i = 0; i < cp->int_params["num_gr"]; i++)
	{
		for (int j = 0; j < cp->int_params["max_num_p_gr_from_go_to_gr"]; j++)
		{
			for (int k = 0; k < cp->int_params["max_num_p_gl_from_go_to_gl"]; k++)
			{	
				if (numpGRfromGOtoGR[i] < cp->int_params["max_num_p_gr_from_go_to_gr"])
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
	for (int i = 0; i < cp->int_params["num_gr"]; i++) totalGOGR += numpGRfromGOtoGR[i];
	
	std::cout << "total GO->GR: " << totalGOGR << std::endl;
	std::cout << "GO->GR Per GR: " << (float)totalGOGR / (float)cp->int_params["num_gr"] << std::endl;
}

void InNetConnectivityState::assignGRDelays(ConnectivityParams *cp, unsigned int msPerStep)
{
	for (int i = 0; i < cp->int_params["num_gr"]; i++)
	{
		//calculate x coordinate of GR position
		int grPosX = i % cp->int_params["gr_x"];

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		int grBCPCSCDist = abs((int)(cp->int_params["gr_x"] / 2 - grPosX));
		pGRDelayMaskfromGRtoBSP[i] = 0x1 << (int)((grBCPCSCDist / cp->int_params["gr_pf_vel_in_gr_x_per_t_step"]
			+ cp->int_params["gr_af_delay_in_t_step"]) / msPerStep);

		for (int j = 0; j < numpGRfromGRtoGO[i]; j++)
		{
			int goPosX = (pGRfromGRtoGO[i][j] % cp->int_params["go_x"]) * (((float)cp->int_params["gr_x"]) / cp->int_params["go_x"]);
			int dfromGRtoGO = abs(goPosX - grPosX);

			if (dfromGRtoGO > cp->int_params["gr_x"] / 2)
			{
				if (goPosX < grPosX) dfromGRtoGO = goPosX + cp->int_params["gr_x"] - grPosX;
				else dfromGRtoGO = grPosX + cp->int_params["gr_x"] - goPosX;
			}
			pGRDelayMaskfromGRtoGO[i][j] = 0x1<< (int)((dfromGRtoGO / cp->int_params["gr_pf_vel_in_gr_x_per_t_step"]
				+ cp->int_params["gr_af_delay_in_t_step"]) / msPerStep);
		}
	}
}

