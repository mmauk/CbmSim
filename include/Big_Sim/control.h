#ifndef _CONTROL_H
#define _CONTROL_H

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>

#include "stdDefinitions/pstdint.h"
#include "interfaces/cbmstate.h"
#include "params/connectivityparams.h"
#include "interfaces/iactivityparams.h"
#include "state/innetconnectivitystate.h"
#include "state/innetactivitystate.h"
#include "interface/cbmsimcore.h"
#include "ecmfpopulation.h"
#include "poissonregencells.h"
#include "interfaces/ectrialsdata.h"
#include "eyelidintegrator.h"

class Control 
{
	public:
		// TODO: create training parameter files
		Control();	
		Control(std::string actParamFile);
		~Control();

		// Objects
		ActivityParams ap;	

		// TODO: make these guys non pointers, somehow!
		CBMState *simState;
		CBMSimCore *simCore;
		ECMFPopulation *mfFreq;
		PoissonRegenCells *mfs;

		const float *grgoG, *mfgoG;
		const ct_uint8_t *mfAP;

		// params that I do not know how to categorize
		float goMin = 0.26;
		float spillFrac	= 0.15;	
		float gogoW = 0.0125;
		float inputStrength = 0.0;

		// sim params
		int gpuIndex = 0;
	   	int gpuP2	 = 2;	

		// Training Parameters
		int numTrainingTrials 	   = 10;
		int homeoTuningTrials 	   = 0;
		int granuleActDetectTrials = 0;

		int msPreCS = 400;
		int msPostCS = 400;

		int csStart = 2000; // begin at 2s?
		int csPhasicSize = 50;

		int csLength = 2000; // with duration 2s?
							 
		// mzone stuff
		int numMZones = 1;

		// MFFreq params (formally in Simulation::getMFs, Simulation::getMFFreq)
		int mfRandSeed = 3;
		float threshDecayTau = 4.0;

		float nucCollFrac = 0.02;
		float csMinRate = 100.0; // uh look at tonic lol
 		float csMaxRate = 110.0;		

		float CSTonicMFFrac = 0.05;
		float tonicFreqMin  = 100.0;
		float tonicFreqMax  = 110.0;

		float CSPhasicMFFrac = 0.0;
		float phasicFreqMin  = 200.0;
		float phasicFreqMax  = 250.0;
		
		float contextMFFrac  = 0.0;
		float contextFreqMin = 20.0;
		float contextFreqMax = 50.0;

		float bgFreqMin   = 10.0;
		float csbgFreqMin = 10.0;
		float bgFreqMax   = 30.0;
		float csbgFreqMax = 30.0;

		bool collaterals_off = false;
		bool secondCS 		 = true;

		float fracImport  = 0.0;
		float fracOverlap = 0.2;

		int trialTime = 5000; /* wild that this is here */
			
		//const ct_uint8_t* grSpks;
		//const float *mfGO;
		//const float *grGO;
		//const float *goGO;
		//float **grGOconductancePSTH;
		//float **mfGOconductancePSTH;

		//float **allGRGORaster;
		//float **allGOGORaster;
		//float **goGOgRaster;
		//float **mfGOgRaster;
		//float **grGOgPSTH; 

		//ct_uint8_t **allMFPSTH;
		ct_uint8_t **allGOPSTH;
		//ct_uint8_t **allGRPSTH;
		//ct_uint8_t **activeGRPSTH;

		//ct_uint8_t **allGORaster;
		ct_uint8_t **allPCRaster;
		ct_uint8_t **allNCRaster;
		ct_uint8_t **allBCRaster;
		ct_uint8_t **allSCRaster;
		//ct_uint8_t **allIORaster;

		//float **allGORaster_gogoG; 
		//float **eyelidPos;
		//float **activeGRgISum;

		//float **allGOInhInput;
		//float **allGOExInput;

		//float **allGOgSumMFGO;
		//float **allGOgSumGOGO;
		//float **allGOgSumGRGO;
		//float **allGOgGOGOcouple;
		//float **allGOVoltage;
		//ct_uint8_t *granuleCSSpkCount;
		//ct_uint8_t **activeGRRaster;
		//ct_uint8_t **preGRRaster;
		//ct_uint8_t **preGRPSTHPreCS;
		//ct_uint8_t **preGRPSTHCS;

		//ct_uint8_t **allGORaster_Trial;
		//ct_uint8_t *goSpkCount_Trial;
		const ct_uint8_t* goSpks; 

		void initializeOutputArrays(int csLength, int numTrainingTrials);

		void runTrials(int simNum, float GOGR, float GRGO, float MFGO);

		void saveOutputArraysToFile(int goRecipParam, int simNum);

		void countGOSpikes(int *goSpkCounter, float &medTrials);

		void fillRasterArrays(CBMSimCore *simCore, int rasterCounter);

		// this should be in CXX Tools or 2D array...
		void write2DCharArray(std::string outFileName, ct_uint8_t **inArr,
			unsigned int numRow, unsigned int numCol);
		void deleteOutputArrays();
};

#endif /*_CONTROL_H*/

