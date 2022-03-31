#ifndef _CONTROL_H
#define _CONTROL_H

#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include "setsim.h"

#include "stdDefinitions/pstdint.h"

#include "interfaces/cbmstate.h"
#include "interfaces/iconnectivityparams.h"
#include "interfaces/iactivityparams.h"
#include "state/innetconnectivitystate.h"

#include "interface/cbmsimcore.h"

#include "ecmfpopulation.h"
#include "poissonregencells.h"
#include "interfaces/ectrialsdata.h"
#include "eyelidintegrator.h"

#include "state/innetactivitystate.h"

class Control 
{

	public:
		Control();
		~Control();

		// Objects
		SetSim simulation;	
		
		EyelidIntegrator* joeeyelidInt;
		ECTrialsData* joeData;

		float *mfBackRate;
		ct_uint8_t **mfRaster_chunk;
		const ct_uint8_t *mfSpks;
		
		//InNetConnectivityState cs;
		const ct_uint8_t *bcSpks, *scSpks, *pcSpks, *ncSpks;
		const float *gogoG, *grgoG, *mfgoG;
		const float *grpcWeights;
		
		float **mfNoiseRatesLearn, **mfNoiseRates;
		float *mfR;
			
		float findMed(int *arr);

		// Training Parameters
		int msPreCS = 400;
		int msPostCS = 400;

		const ct_uint8_t *mfAP;
		int tts;
		int trial; 	
		int numGO = 4096;
		int numGR = 1048576;
		//int numUBC = 1024;

		ct_uint8_t *grPSTHPreCS;
		ct_uint8_t *grPSTHCS;

		int csStart = 2000;
		int csPhasicSize = 50;

		int trialTime = 10000;
			
		int numPC = 32;
		int numBC = 128;
		int numSC = 512;
		int numNC = 8;
		int numIO = 4;
		const ct_uint8_t* grSpks;
		const float *mfGO;
		const float *grGO;
		const float *goGO;
		float **grGOconductancePSTH;
		float **mfGOconductancePSTH;

		float **allGRGORaster;
		float **allGOGORaster;
		float **goGOgRaster;
		float **mfGOgRaster;
		float **grGOgPSTH; 

		//ct_uint8_t **allUBCRaster;
		ct_uint8_t **allGORaster;
		ct_uint8_t **allGORaster1;
		ct_uint8_t **allGORaster2;
		ct_uint8_t **allGORaster3;
		ct_uint8_t **allGORaster4;
		ct_uint8_t **allGORaster5;
		ct_uint8_t **allGORaster6;
		ct_uint8_t **allGORaster7;
		ct_uint8_t **allGORaster8;

		ct_uint8_t **allMFPSTH;
		ct_uint8_t **allGOPSTH;
		//float **allGOPSTH;
		ct_uint8_t **allGRPSTH;
		ct_uint8_t **activeGRPSTH;


		ct_uint8_t **allPCRaster;
		ct_uint8_t **allPCRaster1;
		ct_uint8_t **allPCRaster2;
		ct_uint8_t **allPCRaster3;
		ct_uint8_t **allPCRaster4;
		ct_uint8_t **allPCRaster5;
		ct_uint8_t **allPCRaster6;
		ct_uint8_t **allPCRaster7;
		ct_uint8_t **allPCRaster8;
		ct_uint8_t **allPCRaster9;
		ct_uint8_t **allPCRaster10;
		ct_uint8_t **allPCRaster11;
		ct_uint8_t **allPCRaster12;
		float **allGORaster_gogoG; 
		ct_uint8_t **allNCRaster;
		ct_uint8_t **allBCRaster;
		ct_uint8_t **allSCRaster;
		ct_uint8_t **allIORaster;
		float **eyelidPos;

		float **allGOInhInput;
		float **allGOExInput;

		float **allGOgSumMFGO;
		float **allGOgSumGOGO;
		float **allGOgSumGRGO;
		float **allGOgGOGOcouple;
		float **allGOVoltage;
		ct_uint8_t *granuleCSSpkCount;
		ct_uint8_t **activeGRRaster;
		ct_uint8_t **preGRRaster;
		ct_uint8_t **preGRPSTHPreCS;
		ct_uint8_t **preGRPSTHCS;

		ct_uint8_t **allGORaster_Trial;
		ct_uint8_t *goSpkCount_Trial;

		const ct_uint8_t* goSpks; 
		void runTuningSimulation(int tuningTrials, int numTrials, int simNum, int csSize,
				float csFracMFs, float goMin);
		void runSimulation(int tuningTrials, int numTrials, int simNum, int csSize, float csFracMFs,
				float goMin);
		void runSimulationWithGRdata(int fileNum, int goRecipParam, int numTuningTrials,
				int numGrDetectionTrials, int numTrainingTrials, int simNum, int csSize, float csFracMFs,
				float goMin, float GOGR, float GRGO, float MFGO, float csMinRate, float csMaxRate,
				float gogoW, int inputStrength, int inputWeight_two, float spillFrac);

		float **activeGRgISum;
		void train(int selectState, int filename, int ISIs, int numTrials, int numCon,
				int tunedConNumber);

		void countGOSpikes(int *goSpkCounter, float &medTrials);
		void fillRasterArrays(int rasterCounter);
		void write2DCharArray(std::string outFileName, ct_uint8_t **inArr,
				unsigned int numRow, unsigned int numCol);

		int* getGRIndicies(float CStonicMFfrac);
		int getNumGRIndicies(float CStonicMFfrac);

};

#endif /*_CONTROL_H*/
