#ifndef _CONTROL_H
#define _CONTROL_H

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <ctime>

#include "commandLine/commandline.h"
#include "stdDefinitions/pstdint.h"
#include "params/connectivityparams.h"
#include "params/activityparams.h"
#include "interfaces/cbmstate.h"
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
		Control();
		Control(parsed_build_file &p_file);
		Control(std::string sim_file_name);
		~Control();

		// Objects
		CBMState *simState     = NULL;
		CBMSimCore *simCore    = NULL;
		ECMFPopulation *mfFreq = NULL;
		PoissonRegenCells *mfs = NULL;

		enum vis_mode sim_vis_mode = NO_VIS;
		bool output_arrays_initialized = false; /* temporary, going to refactor soon */
		bool sim_is_paused = false;
		std::string inStateFileName = "";

		const float *grgoG, *mfgoG;
		const ct_uint8_t *mfAP;

		// params that I do not know how to categorize
		float goMin = 0.26; 
		float spillFrac = 0.15; // go->gr synapse, part of build

		// weight parameters
		float weightScale = 0.3275;
		float mfgoW = 0.00350 * weightScale;
		float grgoW = 0.00056 * weightScale;
		float gogrW = 0.01050;
		float gogoW = 0.01250;
		float inputStrength = 0.0;

		// sim params -> TODO: place in simcore
		int gpuIndex = 0;
		int gpuP2    = 2;

		// Training Parameters -> TODO: deprecate in gui runExperiment
		int numTrainingTrials      = 10;
		int homeoTuningTrials      = 0;
		int granuleActDetectTrials = 0;

		int msPreCS = 1500;
		int msPostCS = 1000;

		int csStart = 1500; // begin at 2s?
		int csPhasicSize = 50;

		int csLength = 500; // with duration 2s?

		// mzone stuff -> TODO: place in build file down the road
		int numMZones = 1; 

		// TODO: place in experiment trial down the road
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
	
		// separate set of contextual MF due to position of rabbit
		float contextMFFrac  = 0.0;
		float contextFreqMin = 20.0; 
		float contextFreqMax = 50.0;

		float bgFreqMin   = 10.0;
		float csbgFreqMin = 10.0;
		float bgFreqMax   = 30.0;
		float csbgFreqMax = 30.0;

		bool collaterals_off = false;
		bool secondCS        = true;

		float fracImport  = 0.0;
		float fracOverlap = 0.2;

		int trialTime = 5000; /* wild that this is here */
	
		// raster stuff -> TODO: place in experiment file down the road
		// closer TODO: construct these within the runExperiment loop:
		//              depends upon the trial
		int PSTHColSize = csLength + msPreCS + msPostCS;
		int rasterColumnSize = PSTHColSize * numTrainingTrials;

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
		//ct_uint8_t **activeGRPSTH;

		//ct_uint8_t **allGORaster;
		ct_uint8_t **allGRPSTH;
		ct_uint8_t **allPCRaster;
		ct_uint8_t **allNCRaster;
		ct_uint8_t **allSCRaster;
		ct_uint8_t **allBCRaster;
		//ct_uint8_t **allIORaster;
		ct_uint8_t **allGOPSTH;
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

		void build_sim(parsed_build_file &p_file);
		
		void init_activity_params(std::string actParamFile); // TODO: deprecate

		void init_sim_state(std::string stateFile); // TODO: deprecate

		void save_sim_state_to_file(std::string outStateFile); 

		void save_sim_to_file(std::string outSimFile);

		void initializeOutputArrays();

		void runExperiment(experiment &experiment);

		void runTrials(int simNum, float GOGR, float GRGO, float MFGO); // TODO: deprecate

		void saveOutputArraysToFile(int goRecipParam, int trial, std::tm *local_time, int simNum);

		void countGOSpikes(int *goSpkCounter, float &medTrials);

		void fillOutputArrays(CBMSimCore *simCore, int trial, int PSTHCounter, int rasterCounter);

		// this should be in CXX Tools or 2D array...
		void write2DCharArray(std::string outFileName, ct_uint8_t **inArr,
			unsigned int numRow, unsigned int numCol);
		void deleteOutputArrays();

		void construct_control(enum vis_mode sim_vis_mode); // TODO: deprecate
};

#endif /*_CONTROL_H*/

