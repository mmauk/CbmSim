/*
 * connectivityparams.h
 *
 *  Created on: Oct 15, 2012
 *      Author: varicella
 */

#ifndef CONNECTIVITYPARAMS_H_
#define CONNECTIVITYPARAMS_H_


#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <cstdlib>

#include <stdDefinitions/pstdint.h>
#include <memoryMgmt/dynamic2darray.h>


struct ConnectivityParams
{
	/*
	 * a couple notes:
	 * 		- NUM_CELL type variables are not in original conParams_binChoice2_1.txt
	 * 		- NUM_P_CELLA_TO_CELLB type variables not in original conParams_binChoice2_t.txt
	 */

	// cell numbers
	const int MF_X = 128;
	const int MF_Y = 32;
	const int NUM_MF = 4096;

	const int GL_X = 512;
	const int GL_Y = 128;
	const int NUM_GL = 65536;

	const int GR_X = 2048;
	const int GR_Y = 512;
	const int NUM_GR = 1048576;

	const int GO_X = 128;
	const int GO_Y = 32;
	const int NUM_GO = 4096; 

	const int UBC_X = 64;
	const int UBC_Y = 16;
	const int NUM_UBC = 1024;

	// mf -> gl (uses Joe's Con alg (06/01/2022))
	const int SPAN_MF_TO_GL_X = 512; 
	const int SPAN_MF_TO_GL_Y = 128;
	const int NUM_P_MF_TO_GL = 66177;

	// before was 40 as a conservative size (06/07/2022)
	const int MAX_NUM_P_MF_FROM_MF_TO_GL = 16;

	// NOTE: not a maxnump as alg is adaptive
	const int INITIAL_MF_OUTPUT = 14;
	const int MAX_MF_TO_GL_ATTEMPTS = 4;

	// gl -> gr (uses connectCommon (06/01/2022))
	const int SPAN_GL_TO_GR_X = 4;
	const int SPAN_GL_TO_GR_Y = 4;
	const int NUM_P_GL_TO_GR = 25;

	const int LOW_NUM_P_GL_FROM_GL_TO_GR = 80;
	const int MAX_NUM_P_GR_FROM_GL_TO_GR = 5;
	const int MAX_NUM_P_GL_FROM_GL_TO_GR = 200;

	const int LOW_GL_TO_GR_ATTEMPTS = 20000;
	const int MAX_GL_TO_GR_ATTEMPTS = 50000;

	// gr -> go (both pf and aa use Joe's Con Alg (06/01/2022))
	const int SPAN_PF_TO_GO_X = 2048;
	const int SPAN_PF_TO_GO_Y = 150;
	const int NUM_P_PF_TO_GO = 309399;

	const int MAX_NUM_P_GO_FROM_GR_TO_GO = 5000;
	const int MAX_NUM_P_GR_FROM_GR_TO_GO = 50;

	// one value we play around with
	const int MAX_PF_TO_GO_INPUT = 3750;
	const int MAX_PF_TO_GO_ATTEMPTS = 5;

	const int SPAN_AA_TO_GO_X = 150; /* was 201 before tf? */
	const int SPAN_AA_TO_GO_Y = 150; /* same w this one */
	const int NUM_P_AA_TO_GO = 22801;

	// one value we play around with
	const int MAX_AA_TO_GO_INPUT = 1250;
	const int MAX_AA_TO_GO_ATTEMPTS = 15;

	// go <-> go (uses Joe's Con Alg (06/01/2022))
	const int SPAN_GO_TO_GO_X = 10;	
	const int SPAN_GO_TO_GO_Y = 10;	
	const int NUM_P_GO_TO_GO = 121;

	const int NUM_CON_GO_TO_GO = 12;
	
	const float AMPL_GO_TO_GO = 0.35f;
	const float STD_DEV_GO_TO_GO = 1.95;
	const float P_RECIP_GO_GO = 1.0f;
	const float P_RECIP_LOWER_BASE_GO_GO = 0.0f;
	const bool GO_GO_RECIP_CONS = true;
	const bool REDUCE_BASE_RECIP_GO_GO = false;

	const int MAX_GO_TO_GO_ATTEMPTS = 200;

	// go <-> go gap junctions
	const int SPAN_GO_TO_GO_GJ_X = 8;	
	const int SPAN_GO_TO_GO_GJ_Y = 8;	
	const int NUM_P_GO_TO_GO_GJ = 81;

	// go -> gl (uses Joe's Con Alg (06/01/2022))
	const int SPAN_GO_TO_GL_X = 56;
	const int SPAN_GO_TO_GL_Y = 56;
	const int NUM_P_GO_TO_GL = 3249;

	const int MAX_NUM_P_GL_FROM_GO_TO_GL = 1;
	const int MAX_NUM_P_GO_FROM_GO_TO_GL = 64;

	const float AMPL_GO_TO_GL = 0.01f;
	const float STD_DEV_GO_TO_GL_ML = 100f;
	const float STD_DEV_GO_TO_GL_S = 100f;

	const int MAX_GO_TO_GL_ATTEMPTS = 100;

	// gl -> go (uses connectCommon (06/01/2022))
	// NOTE: span values taken from spanGODecDenOnGL*
	const int SPAN_GL_TO_GO_X = 12;
	const int SPAN_GL_TO_GO_Y = 12;
	const int NUM_P_GL_TO_GO = 169;

	const int LOW_NUM_P_GL_FROM_GL_TO_GO = 1;
	const int MAX_NUM_P_GL_FROM_GL_TO_GO = 1;
	const int MAX_NUM_P_GO_FROM_GL_TO_GO = 16;

	const int INITIAL_GO_INPUT = 1;

	const int LOW_GL_TO_GO_ATTEMPTS = 20000;
	const int MAX_GL_TO_GO_ATTEMPTS = 50000;

	// TRANSLATIONS
	// go -> gr
	const int MAX_NUM_P_GO_FROM_GO_TO_GR = 12800; /* might not use */
	const int MAX_NUM_P_GR_FROM_GO_TO_GR = 3;

	// mf -> gr
	const int MAX_NUM_P_GR_FROM_MF_TO_GR = 5;	
	const int MAX_NUM_P_MF_FROM_MF_TO_GR = 4000; /* 20 * MAX_NUM_P_GL_FROM_GL_TO_GR */

	// mf -> go
	const int MAX_NUM_P_GO_FROM_MF_TO_GO = 16; /* was 40 before (06/07/2022) */
	const int MAX_NUM_P_MF_FROM_MF_TO_GO = 20; /* 20 * MAX_NUM_P_GL_FROM_GL_TO_GO */


	// delay mask vars
	const ct_uint32_t GR_PF_VEL_IN_GR_X_PER_T_STEP = 147;
	const ct_uint32_t GR_AF_DELAY_IN_T_STEP = 1;
	

	//int ubcX;
	//int ubcY;
	//int numUBC;

	//int spanGLtoUBCX;
	//int spanGLtoUBCY;
	//int numpGLtoUBC;
	//
	//int spanUBCtoGLX;
	//int spanUBCtoGLY;
	//int numpUBCtoGL;

	////glomeruli
	//int maxnumpGLfromGLtoGR;
	//int lownumpGLfromGLtoGR;
	//ct_uint32_t maxnumpGLfromGLtoGO;
	//ct_uint32_t maxnumpGLfromGOtoGL;
	////end glomeruli

	////mossy fiber
	//int numMF; //read in as power of 2
	//int mfX;
	//int mfY;
	//
	//int spanMFtoGLX;
	//int spanMFtoGLY;
	//int numpMFtoGL;

	//ct_uint32_t maxnumpMFfromMFtoGO; //derived = numGLOutPerMF*maxNumGODenPerGL
	//int maxnumpMFfromMFtoGR; //derived = numGLOutPerMF*maxNumGRDenPerGL

	//ct_uint32_t numpMFfromMFtoNC;

	////end mossy fibers

	////golgi cells
	//int goX; //read in as power of 2
	//int goY; //read in as power of 2

	//int numGO; //derived = goX*goY
	//int numConGOGO;


	//int spanGOGOsynX;
	//int spanGOGOsynY;
	//int numpGOGOsyn; 
	//float sigmaGOGOsynML;
	//float sigmaGOGOsynS;
	//float peakPconGOGOsyn;
	//float pRecipGOGOsyn;
	//int maxGOGOsyn;

	//int spanGOGLX;
	//int spanGOGLY;
	//int numpGOGL;

	//int maxnumpGOfromGRtoGO;

	//int maxnumpGOfromGLtoGO;
	//int maxnumpGOfromMFtoGO; //derived = maxNumGLInPerGO
	//ct_uint32_t maxnumpGOfromGOtoGL;
	//ct_uint32_t maxnumpGOfromGOtoGR; //derived = maxNumGLOutPerGO*maxNumGRDenPerGL

	//ct_uint32_t spanGODecDenOnGLX;
	//ct_uint32_t spanGODecDenOnGLY;

	//ct_uint32_t spanGOAscDenOnGRX;
	//ct_uint32_t spanGOAscDenOnGRY;

	//int spanGOtoGLX;
	//int spanGOtoGLY;
	//int numpGOtoGL;

	//ct_uint32_t spanGOAxonOnGLX;
	//ct_uint32_t spanGOAxonOnGLY;

	////go-go inhibition
	//int maxnumpGOGABAInGOGO;
	//int maxnumpGOGABAOutGOGO;
	//float **gogoGABALocalCon;

	////go-go coupling
	//int maxnumpGOCoupInGOGO;
	//int maxnumpGOCoupOutGOGO;
	//float **gogoCoupLocalCon;

	////end golgi cells

	////granule cells
	//int grX; //read in as power of 2
	//int grY; //read in as power of 2

	//int numGR; //derived = grX*grY
	//int numGRP2;

	//ct_uint32_t grPFVelInGRXPerTStep;
	//ct_uint32_t grAFDelayInTStep;
	//ct_uint32_t maxnumpGRfromGRtoGO;
	//int maxnumpGRfromGLtoGR;
	//ct_uint32_t maxnumpGRfromGOtoGR;
	//int maxnumpGRfromMFtoGR;

	//int spanGRDenOnGLX;
	//int spanGRDenOnGLY;
	////end granule cells

	////stellate cells
	//ct_uint32_t numSC; //read in as power of 2
	//ct_uint32_t numpSCfromGRtoSC; //derived = numGR/numSC
	//ct_uint32_t numpSCfromGRtoSCP2;
	//ct_uint32_t numpSCfromSCtoPC;//TODO: new
	////end stellate cells

	////purkinje cells
	//ct_uint32_t numPC; //read in as power of 2
	//ct_uint32_t numpPCfromGRtoPC; //derived = numGR/numPC
	//ct_uint32_t numpPCfromGRtoPCP2;
	//ct_uint32_t numpPCfromBCtoPC; //TODO new
	//ct_uint32_t numpPCfromPCtoBC; //TODO: new
	//ct_uint32_t numpPCfromSCtoPC; //TODO: new
	//ct_uint32_t numpPCfromPCtoNC; //TODO: new

	////basket cells
	//ct_uint32_t numBC; //read in as power of 2
	//ct_uint32_t numpBCfromGRtoBC; //derived = numGR/numBC
	//ct_uint32_t numpBCfromGRtoBCP2;
	//ct_uint32_t numpBCfromBCtoPC; //TODO: new
	//ct_uint32_t numpBCfromPCtoBC; //TODO: new

	////TODO: new below
	////nucleus cells
	//ct_uint32_t numNC;
	//ct_uint32_t numpNCfromPCtoNC;
	//ct_uint32_t numpNCfromNCtoIO;
	//ct_uint32_t numpNCfromMFtoNC;

	////inferior olivary cells
	//ct_uint32_t numIO;
	//ct_uint32_t numpIOfromIOtoPC;
	//ct_uint32_t numpIOfromNCtoIO;
	//ct_uint32_t numpIOInIOIO;
	//ct_uint32_t numpIOOutIOIO;


private:

	std::map<std::string, ct_uint32_t> paramMap;
};

#endif /* CONNECTIVITYPARAMS_H_ */
