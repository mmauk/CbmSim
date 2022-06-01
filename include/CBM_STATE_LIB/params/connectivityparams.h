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


class ConnectivityParams
{
public:
	ConnectivityParams();
	ConnectivityParams(std::fstream &infile);

	~ConnectivityParams();

	void writeParams(std::fstream &outfile);

	void showParams(std::ostream &outSt);

	ct_uint32_t getGOX();
	ct_uint32_t getGOY();
	ct_uint32_t getGRX();
	ct_uint32_t getGRY();
	ct_uint32_t getGLX();
	ct_uint32_t getGLY();

	ct_uint32_t getNumMF();
	ct_uint32_t getNumGO();
	ct_uint32_t getNumGR();
	ct_uint32_t getNumGL();
	ct_uint32_t getNumUBC();

	ct_uint32_t getNumSC();
	ct_uint32_t getNumBC();
	ct_uint32_t getNumPC();
	ct_uint32_t getNumNC();
	ct_uint32_t getNumIO();

	std::map<std::string, ct_uint32_t> getParamCopy();

	
	int ubcX;
	int ubcY;
	int numUBC;

	int spanGLtoUBCX;
	int spanGLtoUBCY;
	int numpGLtoUBC;
	
	int spanUBCtoGLX;
	int spanUBCtoGLY;
	int numpUBCtoGL;

	//glomeruli
//	ct_uint32_t
	int glX; //read in as power of 2
	int glY; //read in as power of 2
	int numGL; //derived = glX*glY

	int maxnumpGLfromGLtoGR;
	int lownumpGLfromGLtoGR;
	ct_uint32_t maxnumpGLfromGLtoGO;
	ct_uint32_t maxnumpGLfromGOtoGL;
	//end glomeruli

	//mossy fiber
	int numMF; //read in as power of 2
	int mfX;
	int mfY;
	
	int spanMFtoGLX;
	int spanMFtoGLY;
	int numpMFtoGL;

	ct_uint32_t maxnumpMFfromMFtoGO; //derived = numGLOutPerMF*maxNumGODenPerGL
	int maxnumpMFfromMFtoGR; //derived = numGLOutPerMF*maxNumGRDenPerGL

	ct_uint32_t numpMFfromMFtoNC;

	//end mossy fibers

	//golgi cells
	int goX; //read in as power of 2
	int goY; //read in as power of 2

	int numGO; //derived = goX*goY
	int numConGOGO;


	int spanGOGOsynX;
	int spanGOGOsynY;
	int numpGOGOsyn; 
	float sigmaGOGOsynML;
	float sigmaGOGOsynS;
	float peakPconGOGOsyn;
	float pRecipGOGOsyn;
	int maxGOGOsyn;

	int spanGOGLX;
	int spanGOGLY;
	int numpGOGL;

	int maxnumpGOfromGRtoGO;

	int maxnumpGOfromGLtoGO;
	int maxnumpGOfromMFtoGO; //derived = maxNumGLInPerGO
	ct_uint32_t maxnumpGOfromGOtoGL;
	ct_uint32_t maxnumpGOfromGOtoGR; //derived = maxNumGLOutPerGO*maxNumGRDenPerGL

	ct_uint32_t spanGODecDenOnGLX;
	ct_uint32_t spanGODecDenOnGLY;

	ct_uint32_t spanGOAscDenOnGRX;
	ct_uint32_t spanGOAscDenOnGRY;

	int spanGOtoGLX;
	int spanGOtoGLY;
	int numpGOtoGL;

	ct_uint32_t spanGOAxonOnGLX;
	ct_uint32_t spanGOAxonOnGLY;

	//go-go inhibition
	int maxnumpGOGABAInGOGO;
	int maxnumpGOGABAOutGOGO;
	float **gogoGABALocalCon;

	//go-go coupling
	int maxnumpGOCoupInGOGO;
	int maxnumpGOCoupOutGOGO;
	float **gogoCoupLocalCon;

	//end golgi cells

	//granule cells
	int grX; //read in as power of 2
	int grY; //read in as power of 2

	int numGR; //derived = grX*grY
	int numGRP2;

	ct_uint32_t grPFVelInGRXPerTStep;
	ct_uint32_t grAFDelayInTStep;
	ct_uint32_t maxnumpGRfromGRtoGO;
	int maxnumpGRfromGLtoGR;
	ct_uint32_t maxnumpGRfromGOtoGR;
	int maxnumpGRfromMFtoGR;

	int spanGRDenOnGLX;
	int spanGRDenOnGLY;
	//end granule cells

	//stellate cells
	ct_uint32_t numSC; //read in as power of 2
	ct_uint32_t numpSCfromGRtoSC; //derived = numGR/numSC
	ct_uint32_t numpSCfromGRtoSCP2;
	ct_uint32_t numpSCfromSCtoPC;//TODO: new
	//end stellate cells

	//purkinje cells
	ct_uint32_t numPC; //read in as power of 2
	ct_uint32_t numpPCfromGRtoPC; //derived = numGR/numPC
	ct_uint32_t numpPCfromGRtoPCP2;
	ct_uint32_t numpPCfromBCtoPC; //TODO new
	ct_uint32_t numpPCfromPCtoBC; //TODO: new
	ct_uint32_t numpPCfromSCtoPC; //TODO: new
	ct_uint32_t numpPCfromPCtoNC; //TODO: new

	//basket cells
	ct_uint32_t numBC; //read in as power of 2
	ct_uint32_t numpBCfromGRtoBC; //derived = numGR/numBC
	ct_uint32_t numpBCfromGRtoBCP2;
	ct_uint32_t numpBCfromBCtoPC; //TODO: new
	ct_uint32_t numpBCfromPCtoBC; //TODO: new

	//TODO: new below
	//nucleus cells
	ct_uint32_t numNC;
	ct_uint32_t numpNCfromPCtoNC;
	ct_uint32_t numpNCfromNCtoIO;
	ct_uint32_t numpNCfromMFtoNC;

	//inferior olivary cells
	ct_uint32_t numIO;
	ct_uint32_t numpIOfromIOtoPC;
	ct_uint32_t numpIOfromNCtoIO;
	ct_uint32_t numpIOInIOIO;
	ct_uint32_t numpIOOutIOIO;


private:

	std::map<std::string, ct_uint32_t> paramMap;
};

#endif /* CONNECTIVITYPARAMS_H_ */
