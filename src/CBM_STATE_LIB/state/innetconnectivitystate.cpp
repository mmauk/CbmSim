/*
 * innetconnectivitystate.cpp
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */
#include "state/innetconnectivitystate.h"

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
	haspGLfromMFtoGL = new bool[num_gl];

	numpGLfromGLtoGO = new int[num_gl];
	pGLfromGLtoGO    = allocate2DArray<int>(num_gl, max_num_p_gl_from_gl_to_go);

	numpGLfromGOtoGL = new int[num_gl];
	pGLfromGOtoGL    = allocate2DArray<int>(num_gl, max_num_p_gl_from_go_to_gl);

	numpGLfromGLtoGR = new int[num_gl];
	pGLfromGLtoGR    = allocate2DArray<int>(num_gl, max_num_p_gl_from_gl_to_gr);

	pGLfromMFtoGL    = new int[num_gl];

	numpMFfromMFtoGL = new int[num_mf];
	pMFfromMFtoGL    = allocate2DArray<int>(num_mf, max_num_p_mf_from_mf_to_gl);

	numpMFfromMFtoGR = new int[num_mf];
	pMFfromMFtoGR    = allocate2DArray<int>(num_mf, max_num_p_mf_from_mf_to_gr);

	numpMFfromMFtoGO = new int[num_mf];
	pMFfromMFtoGO    = allocate2DArray<int>(num_mf, max_num_p_mf_from_mf_to_go);

	//golgi
	numpGOfromGLtoGO = new int[num_go];
	pGOfromGLtoGO    = allocate2DArray<int>(num_go, max_num_p_go_from_gl_to_go);

	numpGOfromGOtoGL = new int[num_go];
	pGOfromGOtoGL    = allocate2DArray<int>(num_go, max_num_p_go_from_go_to_gl);

	numpGOfromMFtoGO = new int[num_go];
	pGOfromMFtoGO    = allocate2DArray<int>(num_go, max_num_p_go_from_mf_to_go);

	numpGOfromGOtoGR = new int[num_go];
	pGOfromGOtoGR    = allocate2DArray<int>(num_go, max_num_p_go_from_go_to_gr);

	numpGOfromGRtoGO = new int[num_go];
	pGOfromGRtoGO    = allocate2DArray<int>(num_go, max_num_p_go_from_gr_to_go);

	// coincidentally, numcongotogo == maxnumpgogabaingogo
	numpGOGABAInGOGO  = new int[num_go];
	pGOGABAInGOGO     = allocate2DArray<int>(num_go, num_con_go_to_go);

	numpGOGABAOutGOGO = new int[num_go];
	pGOGABAOutGOGO    = allocate2DArray<int>(num_go, num_con_go_to_go);

	// go <-> go gap junctions
	numpGOCoupInGOGO = new int[num_go];
	pGOCoupInGOGO    = allocate2DArray<int>(num_go, num_p_go_to_go_gj);

	numpGOCoupOutGOGO = new int[num_go];
	pGOCoupOutGOGO    = allocate2DArray<int>(num_go, num_p_go_to_go_gj);

	pGOCoupOutGOGOCCoeff = allocate2DArray<float>(num_go, num_p_go_to_go_gj);
	pGOCoupInGOGOCCoeff  = allocate2DArray<float>(num_go, num_p_go_to_go_gj);

	//granule
	pGRDelayMaskfromGRtoBSP = new ct_uint32_t[num_gr];

	numpGRfromGLtoGR = new int[num_gr];
	pGRfromGLtoGR    = allocate2DArray<int>(num_gr, max_num_p_gr_from_gl_to_gr);

	numpGRfromGRtoGO = new int[num_gr];
	pGRfromGRtoGO    = allocate2DArray<int>(num_gr, max_num_p_gr_from_gr_to_go);

	pGRDelayMaskfromGRtoGO  = allocate2DArray<int>(num_gr, max_num_p_gr_from_gr_to_go);

	numpGRfromGOtoGR = new int[num_gr];
	pGRfromGOtoGR    = allocate2DArray<int>(num_gr, max_num_p_gr_from_go_to_gr);

	numpGRfromMFtoGR = new int[num_gr];
	pGRfromMFtoGR    = allocate2DArray<int>(num_gr, max_num_p_gr_from_mf_to_gr);
}

void InNetConnectivityState::initializeVals()
{
	std::fill(haspGLfromMFtoGL, haspGLfromMFtoGL + num_gl, false);

	std::fill(numpGLfromGLtoGO, numpGLfromGLtoGO + num_gl, 0);
	std::fill(pGLfromGLtoGO[0], pGLfromGLtoGO[0]
		+ num_gl * max_num_p_gl_from_gl_to_go, 0);

	std::fill(numpGLfromGOtoGL, numpGLfromGOtoGL + num_gl, 0);
	std::fill(pGLfromGOtoGL[0], pGLfromGOtoGL[0]
		+ num_gl * max_num_p_gl_from_go_to_gl, 0);

	std::fill(numpGLfromGLtoGR, numpGLfromGLtoGR + num_gl, 0);
	std::fill(pGLfromGLtoGR[0], pGLfromGLtoGR[0]
		+ num_gl * max_num_p_gl_from_gl_to_gr, 0);

	std::fill(pGLfromMFtoGL, pGLfromMFtoGL + num_gl, 0);

	std::fill(numpMFfromMFtoGL, numpMFfromMFtoGL + num_mf, 0);
	std::fill(pMFfromMFtoGL[0], pMFfromMFtoGL[0]
		+ num_mf * max_num_p_mf_from_mf_to_gl, 0);

	std::fill(numpMFfromMFtoGR, numpMFfromMFtoGR + num_mf, 0);
	std::fill(pMFfromMFtoGR[0], pMFfromMFtoGR[0]
		+ num_mf * max_num_p_mf_from_mf_to_gr, 0);

	std::fill(numpMFfromMFtoGO, numpMFfromMFtoGO + num_mf, 0);
	std::fill(pMFfromMFtoGO[0], pMFfromMFtoGO[0]
		+ num_mf * max_num_p_mf_from_mf_to_go, 0);

	std::fill(numpGOfromGLtoGO, numpGOfromGLtoGO + num_go, 0);
	std::fill(pGOfromGLtoGO[0], pGOfromGLtoGO[0]
		+ num_go * max_num_p_go_from_gl_to_go, 0);

	std::fill(numpGOfromGOtoGL, numpGOfromGOtoGL + num_go, 0);
	std::fill(pGOfromGOtoGL[0], pGOfromGOtoGL[0]
		+ num_go * max_num_p_go_from_go_to_gl, 0);

	std::fill(numpGOfromMFtoGO, numpGOfromMFtoGO + num_go, 0);
	std::fill(pGOfromMFtoGO[0], pGOfromMFtoGO[0]
		+ num_go * max_num_p_go_from_mf_to_go, 0);

	std::fill(numpGOfromGOtoGR, numpGOfromGOtoGR + num_go, 0);
	std::fill(pGOfromGOtoGR[0], pGOfromGOtoGR[0]
		+ num_go * max_num_p_go_from_go_to_gr, 0);

	std::fill(numpGOfromGRtoGO, numpGOfromGRtoGO + num_go, 0);
	std::fill(pGOfromGRtoGO[0], pGOfromGRtoGO[0]
		+ num_go * max_num_p_go_from_gr_to_go, 0);

	std::fill(numpGOGABAInGOGO, numpGOGABAInGOGO + num_go, 0);
	std::fill(pGOGABAInGOGO[0], pGOGABAInGOGO[0]
		+ num_go * num_con_go_to_go, INT_MAX);

	std::fill(numpGOGABAOutGOGO, numpGOGABAOutGOGO + num_go, 0);
	std::fill(pGOGABAOutGOGO[0], pGOGABAOutGOGO[0]
		+ num_go * num_con_go_to_go, INT_MAX);

	std::fill(numpGOCoupInGOGO, numpGOCoupInGOGO + num_go, 0);
	std::fill(pGOCoupInGOGO[0], pGOCoupInGOGO[0]
		+ num_go * num_p_go_to_go_gj, 0);

	std::fill(numpGOCoupOutGOGO, numpGOCoupOutGOGO + num_go, 0);
	std::fill(pGOCoupOutGOGO[0], pGOCoupOutGOGO[0]
		+ num_go * num_p_go_to_go_gj, 0);

	std::fill(pGOCoupOutGOGOCCoeff[0], pGOCoupOutGOGOCCoeff[0]
		+ num_go * num_p_go_to_go_gj, 0.0);

	std::fill(pGOCoupInGOGOCCoeff[0], pGOCoupInGOGOCCoeff[0]
		+ num_go * num_p_go_to_go_gj, 0.0);

	std::fill(pGRDelayMaskfromGRtoBSP, pGRDelayMaskfromGRtoBSP + num_gr, 0);

	std::fill(numpGRfromGLtoGR, numpGRfromGLtoGR + num_gr, 0);
	std::fill(pGRfromGLtoGR[0], pGRfromGLtoGR[0]
		+ num_gr * max_num_p_gr_from_gl_to_gr, 0);

	std::fill(numpGRfromGRtoGO, numpGRfromGRtoGO + num_gr, 0);
	std::fill(pGRfromGRtoGO[0], pGRfromGRtoGO[0]
		+ num_gr * max_num_p_gr_from_gr_to_go, 0);

	std::fill(pGRDelayMaskfromGRtoGO[0], pGRDelayMaskfromGRtoGO[0]
		+ num_gr * max_num_p_gr_from_gr_to_go, 0);

	std::fill(numpGRfromGOtoGR, numpGRfromGOtoGR + num_gr, 0);
	std::fill(pGRfromGOtoGR[0], pGRfromGOtoGR[0]
		+ num_gr * max_num_p_gr_from_go_to_gr, 0);

	std::fill(numpGRfromMFtoGR, numpGRfromMFtoGR + num_gr, 0);
	std::fill(pGRfromMFtoGR[0], pGRfromMFtoGR[0]
		+ num_gr * max_num_p_gr_from_mf_to_gr, 0);
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
	rawBytesRW((char *)haspGLfromMFtoGL, num_gl * sizeof(bool), read, file);
	rawBytesRW((char *)numpGLfromGLtoGO, num_gl * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGO[0],
		num_gl * max_num_p_gl_from_gl_to_go * sizeof(int), read, file);
	rawBytesRW((char *)numpGLfromGOtoGL, num_gl * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGOtoGL[0],
		num_gl * max_num_p_gl_from_go_to_gl * sizeof(int), read, file);
	rawBytesRW((char *)numpGLfromGLtoGR, num_gl * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromGLtoGR[0],
		num_gl * max_num_p_gl_from_gl_to_gr * sizeof(int), read, file);
	rawBytesRW((char *)pGLfromMFtoGL, num_gl * sizeof(int), read, file);

	//mossy fibers
	rawBytesRW((char *)numpMFfromMFtoGL, num_mf * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGL[0],
		num_mf * max_num_p_mf_from_mf_to_gl * sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGR, num_mf * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGR[0],
		num_mf * max_num_p_mf_from_mf_to_gr * sizeof(int), read, file);

	rawBytesRW((char *)numpMFfromMFtoGO, num_mf * sizeof(int), read, file);
	rawBytesRW((char *)pMFfromMFtoGO[0],
		num_mf * max_num_p_mf_from_mf_to_go * sizeof(int), read, file);

	//golgi
	rawBytesRW((char *)numpGOfromGLtoGO, num_go * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGLtoGO[0],
		num_go * max_num_p_go_from_gl_to_go * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGL, num_go * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGL[0],
		num_go * max_num_p_go_from_go_to_gl * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromMFtoGO, num_go * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromMFtoGO[0],
		num_go * max_num_p_go_from_mf_to_go * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGOtoGR, num_go * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGOtoGR[0],
		num_go * max_num_p_go_from_go_to_gr * sizeof(int), read, file);

	rawBytesRW((char *)numpGOfromGRtoGO, num_go * sizeof(int), read, file);
	rawBytesRW((char *)pGOfromGRtoGO[0],
		num_go * max_num_p_go_from_gr_to_go * sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAInGOGO, num_go * sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAInGOGO[0],
		num_go * num_con_go_to_go * sizeof(int), read, file);

	rawBytesRW((char *)numpGOGABAOutGOGO, num_go * sizeof(int), read, file);
	rawBytesRW((char *)pGOGABAOutGOGO[0],
		num_go * num_con_go_to_go * sizeof(int), read, file);
	
	rawBytesRW((char *)numpGOCoupInGOGO, num_go*sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupInGOGO[0],
		num_go * num_p_go_to_go_gj * sizeof(int), read, file);

	rawBytesRW((char *)numpGOCoupOutGOGO, num_go * sizeof(int), read, file);
	rawBytesRW((char *)pGOCoupOutGOGO[0],
		num_go * num_p_go_to_go_gj * sizeof(int), read, file);

	rawBytesRW((char *)pGOCoupOutGOGOCCoeff[0],
		num_go * num_p_go_to_go_gj * sizeof(float), read, file);
	rawBytesRW((char *)pGOCoupInGOGOCCoeff[0],
		num_go * num_p_go_to_go_gj * sizeof(float), read, file);

	//granule
	rawBytesRW((char *)pGRDelayMaskfromGRtoBSP, num_gr * sizeof(ct_uint32_t), read, file);

	rawBytesRW((char *)numpGRfromGLtoGR, num_gr * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGLtoGR[0],
		num_gr * max_num_p_gr_from_gl_to_gr * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGRtoGO, num_gr * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGRtoGO[0],
		num_gr * max_num_p_gr_from_gr_to_go * sizeof(int), read, file);
	rawBytesRW((char *)pGRDelayMaskfromGRtoGO[0],
		num_gr * max_num_p_gr_from_gr_to_go * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromGOtoGR, num_gr * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromGOtoGR[0],
		num_gr * max_num_p_gr_from_go_to_gr * sizeof(int), read, file);

	rawBytesRW((char *)numpGRfromMFtoGR, num_gr * sizeof(int), read, file);
	rawBytesRW((char *)pGRfromMFtoGR[0],
		num_gr * max_num_p_gr_from_mf_to_gr * sizeof(int), read, file);
}


void InNetConnectivityState::connectMFGL_noUBC()
{
	// define span and coord arrays locally
	int spanArrayMFtoGLX[span_mf_to_gl_x + 1] = {0};
	int spanArrayMFtoGLY[span_mf_to_gl_y + 1] = {0};
	int xCoorsMFGL[num_p_mf_to_gl] = {0};
	int yCoorsMFGL[num_p_mf_to_gl] = {0};

	// fill span arrays and coord arrays
	for (int i = 0; i < span_mf_to_gl_x + 1; i++)
	{
		spanArrayMFtoGLX[i] = i - (span_mf_to_gl_x / 2);
	}

	for (int i = 0; i < span_mf_to_gl_y + 1; i++)
	{
		spanArrayMFtoGLY[i] = i - (span_mf_to_gl_y / 2);
	}
		
	for (int i = 0; i < num_p_mf_to_gl; i++)
	{
		xCoorsMFGL[i] = spanArrayMFtoGLX[i % (span_mf_to_gl_x + 1)];
		yCoorsMFGL[i] = spanArrayMFtoGLY[i / (span_mf_to_gl_y + 1)]; /* should this be the x one? */
	}

	// scale factors from one cell coord to another
	float gridXScaleSrctoDest = (float)mf_x / (float)gl_x; 
	float gridYScaleSrctoDest = (float)mf_y/ (float)gl_y; 

	// random mf index array, supposedly to even out distribution of connections
	int rMFInd[num_mf] = {0};
	for (int i = 0; i < num_mf; i++) rMFInd[i] = i;
	std::random_shuffle(rMFInd, rMFInd + num_mf);

	// fill random span array with linear indices
	int rMFSpanInd[num_p_mf_to_gl] = {0};
	for (int i = 0; i < num_p_mf_to_gl; i++) rMFSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	// attempt to make connections
	for (int attempts = 0; attempts < max_mf_to_gl_attempts; attempts++)
	{
		std::random_shuffle(rMFInd, rMFInd + num_mf);
		// for each attempt, loop through all presynaptic cells
		for (int i = 0; i < num_mf; i++)
		{
			//Select MF Coordinates from random index array: Complete
			srcPosX = rMFInd[i] % mf_x;
			srcPosY = rMFInd[i] / mf_x;
			
			std::random_shuffle(rMFSpanInd, rMFSpanInd + num_p_mf_to_gl);
			// for each presynaptic cell, attempt to make up to initial output + max attempts
			// connections.
			for (int j = 0; j < num_p_mf_to_gl; j++)
			{
				// calculation of which gl cell this mf is connecting to
				destPosX = xCoorsMFGL[rMFSpanInd[j]]; 
				destPosY = yCoorsMFGL[rMFSpanInd[j]];

				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);
				
				destPosX = (destPosX % gl_x + gl_x) % gl_x;
				destPosY = (destPosY % gl_y + gl_y) % gl_y;
				
				destIndex = destPosY * gl_x + destPosX;
				// break if we hit this dynamic con limit
				if (numpMFfromMFtoGL[rMFInd[i]] == (initial_mf_output + attempts)) break;
				
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
	for (int i = 0; i < num_mf; i++) count += numpMFfromMFtoGL[i];
	
	std::cout << "[INFO]: Total number of Mossy Fiber to Glomeruli connections: " << count << std::endl;
	std::cout << "[INFO]: Correct number: " << num_gl << std::endl;
}

void InNetConnectivityState::connectGLGR(CRandomSFMT &randGen)
{
	float gridXScaleStoD = (float)gr_x / (float)gl_x;
	float gridYScaleStoD = (float)gr_y / (float)gl_y; /* unused :/ */

	bool srcConnected[num_gr] = {false};

	for (int i = 0; i < max_num_p_gr_from_gl_to_gr; i++)
	{
		int srcNumConnected = 0;
		memset(srcConnected, false, num_gr * sizeof(bool));

		while (srcNumConnected < num_gr)
		{
			int srcIndex = randGen.IRandom(0, num_gr - 1);
			if (!srcConnected[srcIndex])
			{
				int srcPosX = srcIndex % gr_x;
				int srcPosY = (int)(srcIndex / gr_x);

				int tempDestNumConLim = low_num_p_gl_from_gl_to_gr;

				for (int attempts = 0; attempts < max_gl_to_gr_attempts; attempts++)
				{
					if (attempts == low_gl_to_gr_attempts) tempDestNumConLim = max_num_p_gl_from_gl_to_gr;

					int destPosX = (int)round(srcPosX / gridXScaleStoD);
					int destPosY = (int)round(srcPosY / gridXScaleStoD);

					// again, should add 1 to spans
					destPosX += round((randGen.Random() - 0.5) * span_gl_to_gr_x);
					destPosY += round((randGen.Random() - 0.5) * span_gl_to_gr_y);

					destPosX = (destPosX % gl_x + gl_x) % gl_x;
					destPosY = (destPosY % gl_y + gl_y) % gl_y;

					int destIndex = destPosY * gl_x + destPosX;

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
	for (int i = 0; i < num_gl; i++)
	{
		count += numpGLfromGLtoGR[i];
	}

	std::cout << "[INFO]: Total number of Glomeruli to Granule connections: " << count << std::endl; 
	std::cout << "[INFO]: Correct number: " << num_gr * max_num_p_gr_from_gl_to_gr << std::endl;
}

void InNetConnectivityState::connectGRGO()
{
	int spanArrayPFtoGOX[span_pf_to_go_x + 1] = {0};
	int spanArrayPFtoGOY[span_pf_to_go_y + 1] = {0};
	int xCoorsPFGO[num_p_pf_to_go] = {0};
	int yCoorsPFGO[num_p_pf_to_go] = {0};

	//PARALLEL FIBER TO GOLGI 
	for (int i = 0; i < span_pf_to_go_x + 1; i++)
	{
		spanArrayPFtoGOX[i] = i - (span_pf_to_go_x / 2);
	}

	for (int i = 0; i < span_pf_to_go_y + 1; i++)
	{
		spanArrayPFtoGOY[i] = i - (span_pf_to_go_y / 2);
	}
		
	for (int i = 0; i < num_p_pf_to_go; i++)
	{
		xCoorsPFGO[i] = spanArrayPFtoGOX[i % (span_pf_to_go_x + 1)];
		yCoorsPFGO[i] = spanArrayPFtoGOY[i / (span_pf_to_go_x + 1)];
	}

	float gridXScaleSrctoDest = (float)go_x / (float)gr_x; 
	float gridYScaleSrctoDest = (float)go_y / (float)gr_y; 

	int rPFSpanInd[num_p_pf_to_go] = {0};
	for (int i = 0; i < num_p_pf_to_go; i++) rPFSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int attempts = 0; attempts < max_pf_to_go_attempts; attempts++)
	{
		for (int i = 0; i < num_go; i++)
		{
			srcPosX = i % go_x;
			srcPosY = i / go_x;

			std::random_shuffle(rPFSpanInd, rPFSpanInd + num_p_pf_to_go);
			for (int j = 0; j < max_pf_to_go_input; j++)
			{
				destPosX = xCoorsPFGO[rPFSpanInd[j]]; 
				destPosY = yCoorsPFGO[rPFSpanInd[j]];
				
				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);
				
				destPosX = (destPosX % gr_x + gr_x) % gr_x;
				destPosY = (destPosY % gr_y + gr_y) % gr_y;

				destIndex = destPosY * gr_x + destPosX;

				if (numpGOfromGRtoGO[i] < max_pf_to_go_input)
				{
					pGOfromGRtoGO[i][numpGOfromGRtoGO[i]] = destIndex;
					numpGOfromGRtoGO[i]++;

					pGRfromGRtoGO[destIndex][numpGRfromGRtoGO[destIndex]] = i;
					numpGRfromGRtoGO[destIndex]++;
				}
			}
		}
	}

	int maxAAtoGOAttempts = max_aa_to_go_attempts;
	int maxAAtoGOInput = max_aa_to_go_input;

	int spanArrayAAtoGOX[span_aa_to_go_x + 1] = {0};
	int spanArrayAAtoGOY[span_aa_to_go_y + 1] = {0};
	int xCoorsAAGO[num_p_aa_to_go] = {0};
	int yCoorsAAGO[num_p_aa_to_go] = {0};

	for (int i = 0; i < span_aa_to_go_x + 1; i++)
	{
		spanArrayAAtoGOX[i] = i - (span_aa_to_go_x / 2);
	}

	for (int i = 0; i < span_aa_to_go_y + 1; i++)
	{
		spanArrayAAtoGOY[i] = i - (span_aa_to_go_y / 2);
	}
		
	for (int i = 0; i < num_p_aa_to_go; i++)
	{
		xCoorsAAGO[i] = spanArrayAAtoGOX[i % (span_aa_to_go_x + 1)];
		yCoorsAAGO[i] = spanArrayAAtoGOY[i / (span_aa_to_go_x + 1)];
	}
	
	int rAASpanInd[num_p_aa_to_go] = {0};
	for (int i = 0; i < num_p_aa_to_go; i++) rAASpanInd[i] = i;

	for (int attempts = 0; attempts < max_aa_to_go_attempts; attempts++)
	{
		for (int i = 0; i < num_go; i++)
		{
			srcPosX = i % go_x;
			srcPosY = i / go_x;

			std::random_shuffle(rAASpanInd, rAASpanInd + num_p_aa_to_go);
			
			for (int j = 0; j < max_aa_to_go_input; j++)
			{
				destPosX = xCoorsAAGO[rAASpanInd[j]]; 
				destPosY = yCoorsAAGO[rAASpanInd[j]];
				
				destPosX += (int)round(srcPosX / gridXScaleSrctoDest);
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);

				destPosX = (destPosX % gr_x + gr_x) % gr_x;
				destPosY = (destPosY % gr_y + gr_y) % gr_y;
						
				destIndex = destPosY * gr_x + destPosX;
				
				if (numpGOfromGRtoGO[i] < max_aa_to_go_input + max_pf_to_go_input)
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
	
	for (int i = 0; i < num_go; i++)
	{
		gr_go_input_sum += numpGOfromGRtoGO[i];
	}

	std::cout << "[INFO]: Total number of granule to golgi inputs: " << gr_go_input_sum << std::endl;

	int gr_go_output_sum = 0;

	for (int i = 0; i < num_gr; i++)
	{
		gr_go_output_sum += numpGRfromGRtoGO[i];
	}

	std::cout << "[INFO]: Total number of granule to golgi outputs: " << gr_go_output_sum << std::endl;
}

void InNetConnectivityState::connectGOGL(CRandomSFMT &randGen)
{
	// using old connectivity alg for now , cannot generalize (do not always know
	// at least both array bounds for 2D arrays at compile time)
	// gl -> go
	float gridXScaleSrctoDest = (float)go_x / (float)gl_x;
	float gridYScaleSrctoDest = (float)go_y / (float)gl_y;

	bool srcConnected[num_go] = {false};

	for (int i = 0; i < max_num_p_go_from_gl_to_go; i++)
	{
		int srcNumConnected = 0;
		while (srcNumConnected < num_go)
		{
			int srcIndex = randGen.IRandom(0, num_go - 1);
			if (!srcConnected[srcIndex])
			{
				int srcPosX = srcIndex % go_x;
				int srcPosY = (int)(srcIndex / go_x);

				int tempDestNumConLim = low_num_p_gl_from_gl_to_go;

				for (int attempts = 0; attempts < max_gl_to_go_attempts; attempts++)
				{
					int destPosX;
					int destPosY;
					int destIndex;

					if (attempts == low_gl_to_go_attempts)
					{
						tempDestNumConLim = max_num_p_gl_from_gl_to_go;
					}

					destPosX = (int)round(srcPosX / gridXScaleSrctoDest);
					destPosY = (int)round(srcPosY / gridXScaleSrctoDest);

					// should multiply spans by 1 for full coverage
					destPosX += round((randGen.Random() - 0.5) * span_gl_to_go_x);
					destPosY += round((randGen.Random() - 0.5) * span_gl_to_go_y);

					destPosX = (destPosX % gl_x + gl_x) % gl_x;
					destPosY = (destPosY % gl_y + gl_y) % gl_y;

					destIndex = destPosY * gl_x + destPosX;

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
		memset(srcConnected, false, num_go * sizeof(bool));
	}

	std::cout << "[INFO]: Finished making gl go connections." << std::endl;
	std::cout << "[INFO]: Starting on go gl connections..." << std::endl;

	// go --> gl
	int spanArrayGOtoGLX[span_go_to_gl_x + 1] = {0};
	int spanArrayGOtoGLY[span_go_to_gl_y + 1] = {0};
	int xCoorsGOGL[num_p_go_to_gl] = {0};
	int yCoorsGOGL[num_p_go_to_gl] = {0};
	float pConGOGL[num_p_go_to_gl] = {0.0};

	// Make span Array
	for (int i = 0; i < span_go_to_gl_x + 1; i++)
	{
		spanArrayGOtoGLX[i] = i - (span_go_to_gl_x / 2);
	}

	for (int i = 0; i < span_go_to_gl_y + 1; i++)
	{
		spanArrayGOtoGLY[i] = i - (span_go_to_gl_y / 2);
	}
		
	for (int i = 0; i < num_p_go_to_gl; i++)
	{
		xCoorsGOGL[i] = spanArrayGOtoGLX[i % (span_go_to_gl_x + 1)];
		yCoorsGOGL[i] = spanArrayGOtoGLY[i / (span_go_to_gl_x + 1)];
	}

	// Probability of connection as a function of distance
	for (int i = 0; i < num_p_go_to_gl; i++)
	{
		float PconX = (xCoorsGOGL[i] * xCoorsGOGL[i])
			/ (2 * std_dev_go_to_gl_ml * std_dev_go_to_gl_ml);
		float PconY = (yCoorsGOGL[i] * yCoorsGOGL[i])
			/ (2 * std_dev_go_to_gl_s* std_dev_go_to_gl_s);
		pConGOGL[i] = ampl_go_to_gl * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < num_p_go_to_gl; i++)
	{
		if ((xCoorsGOGL[i] == 0) && (yCoorsGOGL[i] == 0)) pConGOGL[i] = 0;
	}

	//Make Random Golgi cell Index Array
	int rGOInd[num_go] = {0};
	for (int i = 0; i < num_go; i++) rGOInd[i] = i;

	//Make Random Span Array
	int rGOSpanInd[num_p_go_to_gl] = {0};
	for (int i = 0; i < num_p_go_to_gl; i++) rGOSpanInd[i] = i;

	int srcPosX;
	int srcPosY;
	int destPosX;
	int destPosY;
	int destIndex;

	for (int attempts = 0; attempts < max_go_to_gl_attempts; attempts++)
	{
		std::random_shuffle(rGOInd, rGOInd + num_go);
		
		// Go through each golgi cell 
		for (int i = 0; i < num_go; i++)
		{
			//Select GO Coordinates from random index array: Complete
			srcPosX = rGOInd[i] % go_x;
			srcPosY = rGOInd[i] / go_x;
			
			std::random_shuffle(rGOSpanInd, rGOSpanInd + num_p_go_to_gl);
			
			for (int j = 0; j < num_p_go_to_gl; j++)   
			{
				// relative position of connection
				destPosX = xCoorsGOGL[rGOSpanInd[j]];
				destPosY = yCoorsGOGL[rGOSpanInd[j]];

				destPosX += (int)round(srcPosX / gridXScaleSrctoDest); 
				destPosY += (int)round(srcPosY / gridYScaleSrctoDest);

				destPosX = (destPosX % gl_x + gl_x) % gl_x;  
				destPosY = (destPosY % gl_y + gl_y) % gl_y;
				
				// Change position to Index
				destIndex = destPosY * gl_x + destPosX;
					
				if (numpGOfromGOtoGL[rGOInd[i]] >= initial_go_input + attempts) break; 
				if (randGen.Random() >= 1 - pConGOGL[rGOSpanInd[j]] && 
						numpGLfromGOtoGL[destIndex] < max_num_p_gl_from_go_to_gl) 
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

	for (int i = 0; i < num_gl; i++)
	{
		if (numpGLfromGOtoGL[i] < max_num_p_gl_from_go_to_gl) shitCounter++;
		totalGOGL += numpGLfromGOtoGL[i];
	}

	std::cout << "[INFO]: Empty Glomeruli Counter: " << shitCounter << std::endl;
	std::cout << "[INFO]: Total number of golgi to glomerulus connections: " << totalGOGL << std::endl;
	std::cout << "[INFO]: Average number of golgi to glomerulus connections per glomerulus: " << (float)totalGOGL / (float)num_gl << std::endl;
}

void InNetConnectivityState::connectGOGODecayP(CRandomSFMT &randGen)
{
	int spanArrayGOtoGOsynX[span_go_to_go_x + 1] = {0};
	int spanArrayGOtoGOsynY[span_go_to_go_y + 1] = {0};
	int xCoorsGOGOsyn[num_p_go_to_go] = {0};
	int yCoorsGOGOsyn[num_p_go_to_go] = {0};
	float Pcon[num_p_go_to_go] = {0};

	bool **conGOGOBoolOut = allocate2DArray<bool>(num_go, num_go);
	memset(conGOGOBoolOut[0], false, num_go * num_go * sizeof(bool));

	for (int i = 0; i < span_go_to_go_x + 1; i++)
   	{
		spanArrayGOtoGOsynX[i] = i - (span_go_to_go_x / 2);
	}

	for (int i = 0; i < span_go_to_go_y + 1; i++)
   	{
		spanArrayGOtoGOsynY[i] = i - (span_go_to_go_y / 2);
	}
		
	for (int i = 0; i < num_p_go_to_go; i++)
	{
		xCoorsGOGOsyn[i] = spanArrayGOtoGOsynX[i % (span_go_to_go_x + 1)];
		yCoorsGOGOsyn[i] = spanArrayGOtoGOsynY[i / (span_go_to_go_x + 1)];
	}

	for (int i = 0; i < num_p_go_to_go; i++)
	{
		float PconX = (xCoorsGOGOsyn[i] * xCoorsGOGOsyn[i])
			/ (2 * std_dev_go_to_go * std_dev_go_to_go);
		float PconY = (yCoorsGOGOsyn[i] * yCoorsGOGOsyn[i])
			/ (2 * std_dev_go_to_go * std_dev_go_to_go);
		Pcon[i] = ampl_go_to_go * exp(-(PconX + PconY));
	}
	
	// Remove self connection 
	for (int i = 0; i < num_p_go_to_go; i++) 
	{
		if ((xCoorsGOGOsyn[i] == 0) && (yCoorsGOGOsyn[i] == 0)) Pcon[i] = 0;
	}
	
	int rGOGOSpanInd[num_p_go_to_go] = {0};
	for (int i = 0; i < num_p_go_to_go; i++) rGOGOSpanInd[i] = i;

	for (int attempts = 0; attempts < max_go_to_go_attempts; attempts++) 
	{
		for (int i = 0; i < num_go; i++) 
		{
			int srcPosX = i % go_x;
			int srcPosY = i / go_x;
			
			std::random_shuffle(rGOGOSpanInd, rGOGOSpanInd + num_p_go_to_go);
			
			for (int j = 0; j < num_p_go_to_go; j++)
			{
				int destPosX = srcPosX + xCoorsGOGOsyn[rGOGOSpanInd[j]]; 
				int destPosY = srcPosY + yCoorsGOGOsyn[rGOGOSpanInd[j]];

				destPosX = (destPosX % go_x + go_x) % go_x;
				destPosY = (destPosY % go_y + go_y) % go_y;

				int destIndex = destPosY * go_x + destPosX;
			
				// Normal One
				if ((bool)go_go_recip_cons && !(bool)reduce_base_recip_go_go
					&& randGen.Random()>= 1 - Pcon[rGOGOSpanInd[j]]
					&& !conGOGOBoolOut[i][destIndex]
					&& numpGOGABAOutGOGO[i] < num_con_go_to_go 
					&& numpGOGABAInGOGO[destIndex] < num_con_go_to_go) 
				{
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;
							
					if (randGen.Random() <= p_recip_go_go
							&& !conGOGOBoolOut[destIndex][i]
							&& numpGOGABAOutGOGO[destIndex] < num_con_go_to_go 
							&& numpGOGABAInGOGO[i] < num_con_go_to_go) 
					{
						pGOGABAOutGOGO[destIndex][numpGOGABAOutGOGO[destIndex]] = i;
						numpGOGABAOutGOGO[destIndex]++;
						
						pGOGABAInGOGO[i][numpGOGABAInGOGO[i]] = destIndex;
						numpGOGABAInGOGO[i]++;
						
						conGOGOBoolOut[destIndex][i] = true;
					}
				}
			
				if ((bool)go_go_recip_cons && (bool)reduce_base_recip_go_go
					&& randGen.Random() >= 1 - Pcon[rGOGOSpanInd[j]]
					&& !conGOGOBoolOut[i][destIndex] && (!conGOGOBoolOut[destIndex][i]
					|| randGen.Random() <= p_recip_lower_base_go_go)
					&& numpGOGABAOutGOGO[i] < num_con_go_to_go 
					&& numpGOGABAInGOGO[destIndex] < num_con_go_to_go) 
				{
					pGOGABAOutGOGO[i][numpGOGABAOutGOGO[i]] = destIndex;
					numpGOGABAOutGOGO[i]++;
					
					pGOGABAInGOGO[destIndex][numpGOGABAInGOGO[destIndex]] = i;
					numpGOGABAInGOGO[destIndex]++;
					
					conGOGOBoolOut[i][destIndex] = true;
				}

				if (!(bool)go_go_recip_cons && !(bool)reduce_base_recip_go_go
					&& randGen.Random() >= 1 - Pcon[rGOGOSpanInd[j]]
					&& (!conGOGOBoolOut[i][destIndex]) && !conGOGOBoolOut[destIndex][i]
					&& numpGOGABAOutGOGO[i] < num_con_go_to_go 
					&& numpGOGABAInGOGO[destIndex] < num_con_go_to_go)
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

	for (int i = 0; i < num_go; i++)
	{
		totalGOGOcons += numpGOGABAInGOGO[i];
	}

	std::cout << "[INFO]: Total number of golgi to golgi connections: " << totalGOGOcons << std::endl;
	std::cout << "[INFO]: Average number of golgi to golgi connections per golgi: " << (float)totalGOGOcons / (float)num_go << std::endl;

	int recipCounter = 0;

	for (int i = 0; i < num_go; i++)
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
	std::cout << "[INFO]: Fraction of reciprocal connections: " << fracRecip << std::endl;
	delete2DArray<bool>(conGOGOBoolOut);
}

void InNetConnectivityState::connectGOGO_GJ(CRandomSFMT &randGen)
{
	int spanArrayGOtoGOgjX[span_go_to_go_gj_x + 1] = {0};
	int spanArrayGOtoGOgjY[span_go_to_go_gj_y + 1] = {0};
	int xCoorsGOGOgj[num_p_go_to_go_gj] = {0};
	int yCoorsGOGOgj[num_p_go_to_go_gj] = {0};

	float gjPCon[num_p_go_to_go_gj] = {0.0};
	float gjCC[num_p_go_to_go_gj]   = {0.0};

	bool **gjConBool = allocate2DArray<bool>(num_go, num_go);
	memset(gjConBool[0], false, num_go * num_go * sizeof(bool));

	for (int i = 0; i < span_go_to_go_gj_x + 1; i++)
	{
		spanArrayGOtoGOgjX[i] = i - (span_go_to_go_gj_x / 2);
	}

	for (int i = 0; i < span_go_to_go_gj_y + 1; i++)
	{
		spanArrayGOtoGOgjY[i] = i - (span_go_to_go_gj_y / 2);
	}

	for (int i = 0; i < num_p_go_to_go_gj; i++)
	{
		xCoorsGOGOgj[i] = spanArrayGOtoGOgjX[i % (span_go_to_go_gj_x + 1)];
		yCoorsGOGOgj[i] = spanArrayGOtoGOgjY[i / (span_go_to_go_gj_y + 1)];
	}

	// "In Vivo additions"
	for (int i = 0; i < span_go_to_go_gj_x; i++)
	{
		float gjPConX = exp(((abs(xCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );	
		float gjPConY = exp(((abs(yCoorsGOGOgj[i]) * 36.0) - 267.0) / 39.0 );
		gjPCon[i] = ((-1745.0 + (1836.0 / (1 + (gjPConX + gjPConY)))) * 0.01);
		
		float gjCCX = exp(abs(xCoorsGOGOgj[i] * 100.0) / 190.0);
		float gjCCY = exp(abs(yCoorsGOGOgj[i] * 100.0) / 190.0);
		gjCC[i] = (-2.3 + (23.0 / ((gjCCX + gjCCY)/2.0))) * 0.09;
	}

	// Remove self connection 
	for (int i = 0; i < num_p_go_to_go_gj; i++)
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

	for (int i = 0; i < num_go; i++)
	{
		srcPosX = i % go_x;
		srcPosY = i / go_x;
		
		for (int j = 0; j < num_p_go_to_go_gj; j++)
		{
			destPosX = srcPosX + xCoorsGOGOgj[j]; 
			destPosY = srcPosY + yCoorsGOGOgj[j];

			destPosX = (destPosX % go_x + go_x) % go_x;
			destPosY = (destPosY % go_y + go_y) % go_y;
					
			destIndex = destPosY * go_x + destPosX;

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
	for (int i = 0; i < num_gr; i++)
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
	
	for (int i = 0; i < num_gr; i++)
	{
		grMFInputCounter += numpGRfromMFtoGR[i];
	}

	std::cout << "[INFO]: Total MF inputs: " << grMFInputCounter << std::endl;

	// Mossy fiber to Golgi
	for (int i = 0; i < num_go; i++)
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
	for (int i = 0; i < num_gr; i++)
	{
		for (int j = 0; j < max_num_p_gr_from_go_to_gr; j++)
		{
			for (int k = 0; k < max_num_p_gl_from_go_to_gl; k++)
			{
				if (numpGRfromGOtoGR[i] < max_num_p_gr_from_go_to_gr)
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
	for (int i = 0; i < num_gr; i++) totalGOGR += numpGRfromGOtoGR[i];
	
	std::cout << "[INFO]: Total golgi to granule connections: " << totalGOGR << std::endl;
	std::cout << "[INFO]: Average number of golgi to granule connections per granule: " << (float)totalGOGR / (float)num_gr << std::endl;
}

void InNetConnectivityState::assignGRDelays(unsigned int msPerStep)
{
	for (int i = 0; i < num_gr; i++)
	{
		//calculate x coordinate of GR position
		int grPosX = i % gr_x;

		//calculate distance of GR to BC, PC, and SC, and assign time delay.
		int grBCPCSCDist = abs((int)(gr_x / 2 - grPosX));
		pGRDelayMaskfromGRtoBSP[i] = 0x1 << (int)((grBCPCSCDist / gr_pf_vel_in_gr_x_per_t_step
			+ gr_af_delay_in_t_step) / msPerStep);

		for (int j = 0; j < numpGRfromGRtoGO[i]; j++)
		{
			int goPosX = (pGRfromGRtoGO[i][j] % go_x) * ((float)gr_x / go_x);
			int dfromGRtoGO = abs(goPosX - grPosX);

			if (dfromGRtoGO > gr_x / 2)
			{
				if (goPosX < grPosX) dfromGRtoGO = goPosX + gr_x - grPosX;
				else dfromGRtoGO = grPosX + gr_x - goPosX;
			}
			pGRDelayMaskfromGRtoGO[i][j] = 0x1<< (int)((dfromGRtoGO / gr_pf_vel_in_gr_x_per_t_step 
				+ gr_af_delay_in_t_step) / msPerStep);
		}
	}
}

