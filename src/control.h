#ifndef _CONTROL_H
#define _CONTROL_H

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <ctime>

#include "commandline.h"
#include <cstdint>
#include "connectivityparams.h"
#include "activityparams.h"
#include "cbmstate.h"
#include "innetconnectivitystate.h"
#include "innetactivitystate.h"
#include "cbmsimcore.h"
#include "ecmfpopulation.h"
#include "poissonregencells.h"
#include "bits.h"

// TODO: place in a common place, as gui uses a constant like this too
#define NUM_CELL_TYPES 8

/** convenience indexing cell arrays
 * used for indexing raster arrays cell num arrays, and name arrays
 */
enum cell_id {MF, GR, GO, BC, SC, PC, IO, NC};

/** used for the gui in Control::runSession
 * an identifier to hold the state of the current run of the session
 */
enum sim_run_state {NOT_IN_RUN, IN_RUN_NO_PAUSE, IN_RUN_PAUSE};

/** used to collect and sum all cell spikes
 * non_cs_spike_counter is used for the period before the cs period.
 * arrays run in the order of cell_id enum.
 */
struct cell_spike_sums
{
	uint32_t num_cells;
	uint32_t non_cs_spike_sum;
	uint32_t cs_spike_sum;
	uint32_t *non_cs_spike_counter;
	uint32_t *cs_spike_counter;
};

/** used to compute mean and median firing rates
 * is currently used in the Array of Structures (AoS)
 * design. 
 */
struct cell_firing_rates
{
	float non_cs_mean_fr;
	float non_cs_median_fr;
	float cs_mean_fr;
	float cs_median_fr;
};

/** The class that controls the simulation
 *
 * All of the behaviour of the simulation, from the number of trials
 * to the types of mossy fiber input, is specified here. Output arrays are
 * also included in this class, in addition instances of all the classes
 * that afford the simulation its functionality, including the state and the
 * core of the simulation.
 *
 */
class Control 
{
	public:
		Control(parsed_commandline &p_cl);
		~Control();

		trials_data td; /* used for ordering and type of trials */
		CBMState *simState     = NULL; /* contains the connectivity and activity states. Is copied over to the gpus */
		CBMSimCore *simCore    = NULL; /* contains interface code to the gpus in order to copy arrays and run the sim */
		ECMFPopulation *mfFreq = NULL; /* separate state class containing the frequencies of all mossy fibers */
		PoissonRegenCells *mfs = NULL; /* separate state class containing the spikes of all mossy fibers */

		bool trials_data_initialized = false; /* used in the destructor */ 
		bool sim_initialized         = false; /* used in the constructor and gui-related functions */

		bool internal_arrays_initialized = false; /* used in the destructor */ 
		bool output_arrays_initialized   = false; /* used in the destructor */
		bool spike_sums_initialized      = false; /* used to check whether spike sums should be calculated */
		enum sim_run_state run_state     = NOT_IN_RUN; /* used in the gui to hide or make visible buttons in gui */

		std::string visual_mode          = ""; /* raw string from parsed_commandline */
		std::string run_mode             = ""; /* raw string from parsed commandline. either 'build' or 'run' */
		std::string curr_build_file_name = ""; /* build file used to create current sim. is empty in run mode */ 
		std::string curr_sess_file_name  = ""; /* session file used to dictate trial type and order. is empty in build mode */
		std::string curr_sim_file_name   = ""; /* current input simulation file. is empty in build mode */
		std::string out_sim_file_name    = ""; /* output of the requested session run. */

		// TODO: place in .sess file
		float spillFrac = 0.15; /* the fraction of go->gr synapses that is spillover from the main post-synaptic target */ 

		int gpuIndex = 0; /* the starting index for counting which gpus will be used */
		int num_gpu  = 2; /* the number of gpus to be used */

		int trial = 0; /* trial number tracker. used as a class attrib for dedicated functions in gui */

		int csPhasicSize     = 50; /* transient cs time period, measured in ms */
		int numMZones        = 1; /* number of microzones within the simulation */
		int mfRandSeed       = 3; /* random seed used in generating mf frequencies and dcn collateral connectivities */
		float nucCollFrac    = 0.02;
		//float csMinRate = 100.0; // uh look at tonic lol
		//float csMaxRate = 110.0;

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

		int trialTime = 0; /* was 5000 (10/17/2022) */
	
		// raster measurement params. 
		int msPreCS = 0; // was 400 (10/17/2022) was 1500 (08/09/2022)
		int msPostCS = 0; // was 400 (10/17/2022) was 1000 (08/09/2022)
		int PSTHColSize = 0; // derived param, from msPreCS, msPostCS and csLength 

		enum plasticity pf_pc_plast;
		enum plasticity mf_nc_plast;

		std::string rf_names[NUM_CELL_TYPES];

		std::string pf_pc_weights_file = "";
		std::string mf_nc_weights_file = "";

		struct cell_spike_sums spike_sums[NUM_CELL_TYPES];
		struct cell_firing_rates firing_rates[NUM_CELL_TYPES];
		const uint8_t *cell_spikes[NUM_CELL_TYPES];

		const float *grgoG, *mfgoG, *gogrG, *mfgrG;
		float *sample_pfpc_syn_weights;
		const uint8_t *mfAP, *goSpks;
		
		const uint8_t *cell_spks[NUM_CELL_TYPES];
		int rast_cell_nums[NUM_CELL_TYPES];
		uint8_t **rast_internal[NUM_CELL_TYPES];

		uint32_t rast_sizes[NUM_CELL_TYPES]; // initialized in initializeOutputArrays
		uint8_t *rast_output[NUM_CELL_TYPES];

		float **all_pc_vm_rast_internal;
		float **all_nc_vm_rast_internal;
		float **all_io_vm_rast_internal;

		void build_sim();

		void set_plasticity_modes(parsed_commandline &p_cl);
		void init_sim(parsed_sess_file &s_file, std::string in_sim_filename);
		void reset_sim(std::string in_sim_filename);

		void save_sim_to_file(std::string outSimFile);
		void save_pfpc_weights_to_file(std::string out_pfpc_file);
		void load_pfpc_weights_from_file(std::string in_pfpc_file);
		void save_mfdcn_weights_to_file(std::string out_mfdcn_file);
		void load_mfdcn_weights_from_file(std::string in_mfdcn_file);

		void save_raster_to_file(std::string raster_file_name, enum cell_id);

		void get_raster_filenames(std::map<std::string, std::string> &raster_files);
		void get_weights_filenames(std::map<std::string, std::string> &weights_files);
		void initialize_rast_cell_nums();
		void initialize_cell_spikes();
		void initialize_spike_sums();
		void initialize_rast_internal();
		void initializeOutputArrays();

		void runSession(struct gui *gui);

		void reset_spike_sums();
		void reset_rast_internal();
		void resetOutputArrays();

		void countGOSpikes(int *goSpkCounter, float &medTrials);
		void update_spike_sums(int tts, float onset_cs, float offset_cs);
		void calculate_firing_rates(float onset_cs, float offset_cs);
		void fill_rast_internal(int PSTHCounter);
		void fillOutputArrays();
		void saveOutputArraysToFile();

		void delete_rast_internal();
		void delete_spike_sums();
		void deleteOutputArrays();
};

#endif /*_CONTROL_H*/

