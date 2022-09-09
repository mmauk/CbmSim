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
#include "bits/bits.h"

// TODO: place in a common place, as gui uses a constant like this too
#define NUM_CELL_TYPES 8
#define NUM_RUN_STATES 3
// convenience enum for indexing cell arrays
enum {MF, GR, GO, BC, SC, PC, IO, DCN};

struct cell_spike_sums
{ ct_uint32_t num_cells;
	ct_uint32_t non_cs_spike_sum;
	ct_uint32_t cs_spike_sum;
	ct_uint32_t *non_cs_spike_counter;
	ct_uint32_t *cs_spike_counter;
};

struct cell_firing_rates
{
	float non_cs_mean_fr;
	float non_cs_median_fr;
	float cs_mean_fr;
	float cs_median_fr;
};

enum sim_run_state {NOT_IN_RUN, IN_RUN_NO_PAUSE, IN_RUN_PAUSE};

class Control 
{
	public:
		Control(parsed_build_file &p_file);
		Control(enum vis_mode sim_vis_mode);
		Control(char ***argv, enum vis_mode sim_vis_mode);
		~Control();

		// Objects
		experiment expt; // TODO: find a better scheme for this
		CBMState *simState     = NULL;
		CBMSimCore *simCore    = NULL;
		ECMFPopulation *mfFreq = NULL;
		PoissonRegenCells *mfs = NULL;

		enum vis_mode sim_vis_mode = NO_VIS;
		bool internal_arrays_initialized = false; /* temporary, going to refactor soon */
		bool output_arrays_initialized = false;
		bool spike_sums_initialized = false;
		enum sim_run_state run_state = NOT_IN_RUN; 
		std::string inStateFileName = "";
		std::string inSimFileName = "";

		const float *grgoG, *mfgoG, *gogrG, *mfgrG;
		const ct_uint8_t *mfAP;

		// params that I do not know how to categorize
		float goMin = 0.26; 
		float spillFrac = 0.15; // go->gr synapse, part of build
		float inputStrength = 0.0;

		// sim params -> TODO: place in simcore
		int gpuIndex = 0;
		int gpuP2    = 2;

		// Training Parameters -> TODO: deprecate in gui runExperiment
		int numTrainingTrials      = 100;
		int homeoTuningTrials      = 0;
		int granuleActDetectTrials = 0;

		int trial = 0;
		int msPreCS = 400; // was 1500 (08/09/2022)
		int msPostCS = 400; // was 1000 (08/09/2022)

		int csStart = 2000; // was 1500 (08/09/2022)
		int csPhasicSize = 50;

		int csLength = 2000; 

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
	
		struct cell_spike_sums spike_sums[NUM_CELL_TYPES] = {{}};
		struct cell_firing_rates firing_rates[NUM_CELL_TYPES] = {{}};
		const ct_uint8_t *cell_spikes[NUM_CELL_TYPES];

		int gr_indices[4096] = {0};

		ct_uint8_t **all_mf_rast_internal;
		ct_uint8_t **all_go_rast_internal;
		ct_uint8_t **sample_gr_rast_internal;
		ct_uint8_t **all_pc_rast_internal;
		ct_uint8_t **all_nc_rast_internal;
		ct_uint8_t **all_sc_rast_internal;
		ct_uint8_t **all_bc_rast_internal;
		ct_uint8_t **all_io_rast_internal;

		/* initialized within initializeOutputArrays */
		ct_uint32_t all_mf_rast_size;   
		ct_uint32_t all_go_rast_size;
		ct_uint32_t sample_gr_rast_size;
		ct_uint32_t all_pc_rast_size;  
		ct_uint32_t all_nc_rast_size;  
		ct_uint32_t all_sc_rast_size;  
		ct_uint32_t all_bc_rast_size;  
		ct_uint32_t all_io_rast_size; 

		ct_uint8_t *allMFRaster;
		ct_uint8_t *allGORaster;
		ct_uint8_t *sampleGRRaster;
		ct_uint8_t *allPCRaster;
		ct_uint8_t *allNCRaster;
		ct_uint8_t *allSCRaster;
		ct_uint8_t *allBCRaster;
		ct_uint8_t *allIORaster;

		float *sample_pfpc_syn_weights;

		float **all_pc_vm_rast_internal;
		float **all_nc_vm_rast_internal;
		float **all_io_vm_rast_internal;

		//TODO: get mfdcn weights too 
		//ct_uint8_t **allGOPSTH;
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

		void build_sim();
		
		void init_experiment(std::string in_expt_filename);
		void init_sim(std::string in_sim_filename);

		void save_sim_state_to_file(std::string outStateFile); /* TODO: deprecate, what else do we use for? */
		void save_sim_to_file(std::string outSimFile);
		void save_pfpc_weights_to_file(std::string out_pfpc_file);
		void load_pfpc_weights_from_file(std::string in_pfpc_file);
		void save_mfdcn_weights_to_file(std::string out_mfdcn_file);
		void load_mfdcn_weights_from_file(std::string in_mfdcn_file);

		void save_gr_psth_to_file(std::string out_gr_psth_file);
		void save_go_psth_to_file(std::string out_go_psth_file);
		void save_pc_psth_to_file(std::string out_pc_psth_file);
		void save_nc_psth_to_file(std::string out_nc_psth_file);
		void save_io_psth_to_file(std::string out_io_psth_file);
		void save_bc_psth_to_file(std::string out_bc_psth_file);
		void save_sc_psth_to_file(std::string out_sc_psth_file);
		void save_mf_psth_to_file(std::string out_mf_psth_file);

		void initialize_spike_sums();
		void initialize_rast_internal();
		void initializeOutputArrays();
		void runExperiment(struct gui *gui);

		void saveOutputArraysToFile();

		void countGOSpikes(int *goSpkCounter, float &medTrials);
		void update_spike_sums(int tts);
		void reset_spike_sums();
		void calculate_firing_rates();
		void fill_rast_internal(int PSTHCounter);
		void reset_rast_internal();
		void fillOutputArrays();

		// this should be in CXX Tools or 2D array...
		void write2DCharArray(std::string outFileName, ct_uint8_t **inArr,
			unsigned int numRow, unsigned int numCol);
		void delete_rast_internal();
		void delete_spike_sums();
		void deleteOutputArrays();
};

#endif /*_CONTROL_H*/

