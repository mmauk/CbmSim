/** @file control.h
 *  @brief Acts as the main controller of actions related to building
 *  simulations (i.e. establishing connectivity) in addition to running
 *  trials on the built simulation.
 *
 *  @author Sean Gallogly (sean.gallo@austin.utexas.edu)
 *  @bug TODO report back on bugs
 */

#ifndef _CONTROL_H
#define _CONTROL_H

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <string>

#include "activityparams.h"
#include "cbmsimcore.h"
#include "cbmstate.h"
#include "commandline.h"
#include "connectivityparams.h"
#include "ecmfpopulation.h"
#include "info_file.h"
#include "innetactivitystate.h"
#include "innetconnectivitystate.h"

// TODO: place in a common place, as gui uses a constant like this too
#define NUM_CELL_TYPES 8
#define NUM_WEIGHTS_TYPES 2
#define NUM_SAVE_OPTS 19
#define NUM_SYN_CONS 12

// TODO:  should put these in file_utility.h
const std::string RAST_EXT[NUM_CELL_TYPES] = {".mfr", ".grr", ".gor", ".bcr",
                                              ".scr", ".pcr", ".ior", ".ncr"};
const std::string PSTH_EXT[NUM_CELL_TYPES] = {".mfp", ".grp", ".gop", ".bcp",
                                              ".scp", ".pcp", ".iop", ".ncp"};
const std::string WEIGHTS_EXT[NUM_WEIGHTS_TYPES] = {".pfpcw", ".mfncw"};
const std::string SYN_CONS_EXT[NUM_SYN_CONS] = {
    ".mfgr", ".grgo", ".mfgo", ".gogo", ".gogr", ".bcpc",
    ".scpc", ".pcbc", ".pcnc", ".ioio", ".ncio", ".mfnc"};

/* convenience enum for indexing output data type */
enum save_opts {
  MF_RAST,
  GR_RAST,
  GO_RAST,
  BC_RAST,
  SC_RAST,
  PC_RAST,
  IO_RAST,
  NC_RAST,
  MF_PSTH,
  GR_PSTH,
  GO_PSTH,
  BC_PSTH,
  SC_PSTH,
  PC_PSTH,
  IO_PSTH,
  NC_PSTH,
  PFPC,
  MFNC,
  SIM
};

/*
 * convenience enum for indexing cell arrays
 */
enum cell_id { MF, GR, GO, BC, SC, PC, IO, NC };

/*
 * convenience array for getting string representations of the cell ids
 */
const std::string CELL_IDS[NUM_CELL_TYPES] = {"MF", "GR", "GO", "BC",
                                              "SC", "PC", "IO", "NC"};
// TODO this same entity exists in different translation unit. create
// common one
const std::string SYN_CONS_IDS[NUM_SYN_CONS] = {"MFGR", "GRGO", "MFGO", "GOGO",
                                                "GOGR", "BCPC", "SCPC", "PCBC",
                                                "PCNC", "IOIO", "NCIO", "MFNC"};

/*
 * sums and counters used to support cell population firing rate calculation
 */
struct cell_spike_sums {
  uint32_t non_cs_spike_sum;
  uint32_t cs_spike_sum;
  uint32_t *non_cs_spike_counter;
  uint32_t *cs_spike_counter;
};

/*
 * a convenience structure for placing cell population firing rate descriptive
 * statistics into: these values are calculated at the end of every trial if the
 * GUI is in use.
 */
struct cell_firing_rates {
  float non_cs_mean_fr;
  float non_cs_median_fr;
  float cs_mean_fr;
  float cs_median_fr;
};

/*
 * convenience enum used to track the state of the program relative to
 * simulation use. This enum is used to populate the info file.
 */
enum sim_run_state { NOT_IN_RUN, IN_RUN_NO_PAUSE, IN_RUN_PAUSE };

/** @class Control control.h "src/control.h"
 *  @brief Controlling class that handles building simulations, running
 *  simulations, and loading data from file and to file.
 */
class Control {
public:
  /**
   *  @brief Contstructor used to create an object of the control class from
   *  a parsed commandline object.
   *
   *  @param p_cl A reference to a struct of type parsed_commandline.
   */
  Control(parsed_commandline &p_cl);

  /**
   *  @brief Default constructor. Handles de-allocating of all memory allocated
   *  on the heap.
   */
  ~Control();

  // Objects
  parsed_sess_file s_file;
  trials_data td;
  info_file_data if_data;
  CBMState *simState = NULL;
  CBMSimCore *simCore = NULL;
  ECMFPopulation *mfs = NULL;
  // PoissonRegenCells *mfs = NULL;

  /* temporary state check vars */
  bool use_gui = false;
  bool trials_data_initialized = false;
  bool sim_initialized = false;

  bool raster_arrays_initialized = false;
  bool psth_arrays_initialized = false;
  bool spike_sums_initialized = false;

  bool data_out_dir_created = false;
  bool out_sim_filename_created = false;
  bool out_info_filename_created = false;
  bool out_biv_filename_created = false;
  bool raster_filenames_created = false;
  bool psth_filenames_created = false;

  bool pfpc_weights_filenames_created = false;
  bool mfnc_weights_filenames_created = false;

  bool con_arrs_filenames_created = false;

  enum sim_run_state run_state = NOT_IN_RUN;

  /* params that I do not know how to categorize */
  float goMin = 0.26;
  float spillFrac = 0.15; // go->gr synapse, part of build
  float inputStrength = 0.0;

  // sim params -> TODO: place in simcore
  uint32_t gpuIndex = 2;
  uint32_t gpuP2 = 2;

  uint32_t trial;
  uint32_t raster_counter;

  uint32_t csPhasicSize = 50;

  // mzone stuff -> TODO: place in build file down the road
  uint32_t numMZones = 1;

  /* MFFreq params (formally in Simulation::getMFs, Simulation::getMFFreq) */
  uint32_t mfRandSeed = 3;
  float threshDecayTau = 4.0;

  float nucCollFrac = 0.02;

  float CSTonicMFFrac = 0.05;
  float tonicFreqMin = 100.0;
  float tonicFreqMax = 110.0;

  float CSPhasicMFFrac = 0.0;
  float phasicFreqMin = 200.0;
  float phasicFreqMax = 250.0;

  /* separate set of contextual MF due to position of rabbit */
  float contextMFFrac = 0.0;
  float contextFreqMin = 20.0;
  float contextFreqMax = 50.0;

  float bgFreqMin = 10.0;
  float csbgFreqMin = 10.0;
  float bgFreqMax = 30.0;
  float csbgFreqMax = 30.0;

  bool collaterals_off = false;
  bool secondCS = true;

  float fracImport = 0.0;
  float fracOverlap = 0.2;

  /* time related variables in-trial */
  uint32_t trialTime = 0;
  uint32_t msPreCS = 0;   // how much time before the cs do we collect data for
  uint32_t msPostCS = 0;  // how much time after the cs do we collect data for
  uint32_t msMeasure = 0; // total amount of time data is collected for

  /* plasticity types */
  enum plasticity pf_pc_plast = GRADED;
  enum plasticity mf_nc_plast = GRADED;

  /* input and output filenames */
  std::string sess_file_name = "";
  std::string data_out_path = "";
  std::string data_out_base_name = "";

  std::string out_sim_name = "";
  std::string out_info_name = "";
  std::string out_bvi_name = "";

  std::string rf_names[NUM_CELL_TYPES];
  std::string pf_names[NUM_CELL_TYPES];

  std::string pfpc_weights_file = "";
  std::string mfnc_weights_file = "";

  std::string pre_con_arrs_names[NUM_SYN_CONS];
  std::string post_con_arrs_names[NUM_SYN_CONS];

  /* instantiation of above structs for firing rate calculations in gui */
  struct cell_spike_sums spike_sums[NUM_CELL_TYPES];
  struct cell_firing_rates firing_rates[NUM_CELL_TYPES];

  /* extra conductance pointers obtained from innet and mzone */
  const float *grgoG, *mfgoG, *gogrG, *mfgrG;
  /* mossy fiber and golgi spike pointers for test data collection */
  const uint8_t *mfAP, *goSpks;

  /* time series data output related variables */
  const uint8_t *cell_spikes[NUM_CELL_TYPES];
  uint32_t rast_cell_nums[NUM_CELL_TYPES];
  uint8_t **rasters[NUM_CELL_TYPES];
  uint8_t **psths[NUM_CELL_TYPES];

  /* save functions for time series data */
  uint32_t rast_sizes[NUM_CELL_TYPES];
  std::function<void()> psth_save_funcs[NUM_CELL_TYPES];
  std::function<void()> rast_save_funcs[NUM_CELL_TYPES];

  /* voltage rasters for gui pc window */
  float **pc_vm_raster;
  float **nc_vm_raster;
  float **io_vm_raster;

  /**
   *  @brief connect synapses between all cell populations.
   */
  void build_sim();

  /**
   *  @brief Set plasticity class attributes.
   *  @param pfpc_plasticity String code for parallel fiber to purkinje cell
   *  plasticity.
   *  @param mfnc_plasticity String code for mossy fiber to deep nucleus cell
   *  plasticity.
   */
  void set_plasticity_modes(std::string pfpc_plasticity,
                            std::string mfnc_plasticity);
  /**
   *  @brief initialize session class attribute from file.
   *  @param sess_file String representing the filepath of the input session
   *  file.
   */
  void initialize_session(std::string sess_file);

  /**
   *  @brief initialize simulation cbm state and cbm core.
   *  @param in_sim_filename String representing the filepath of the input
   *  simulation file.
   */
  void init_sim(std::string in_sim_filename);

  /**
   *  @brief reset cbm state and cbm core to initial values from
   *  in_sim_filename.
   *  @param in_sim_filename String representing the filepath of the input
   *  simulation file used to initialize cbm state and cbm core.
   */
  void reset_sim(std::string in_sim_filename);

  /**
   *  @brief Save simulation to file in the form of a .sim binary file.
   */
  void save_sim_to_file();

  /**
   *  @brief Write run time and duration parameters to info file.
   *  @param out_buf Output buffer to send textual data to.
   */
  void write_header_info(std::fstream &out_buf);

  /**
   *  @brief Write command line specification information to info file.
   *  @param out_buf Output buffer to send textual data to.
   */
  void write_cmdline_info(std::fstream &out_buf);

  /**
   *  @brief Write specified session information to info file.
   *  @param out_buf Output buffer to send textual data to.
   */
  void write_sess_info(std::fstream &out_buf);

  /**
   *  @brief Write all simulation run information to text file.
   */
  void save_info_to_file();

  /**
   *  @brief Write all relevant info to bvi file.
   */
  void save_bvi_to_file();

  /**
   *  @brief Save the parallel fiber purkinje cell weights to binary file.
   */
  void save_pfpc_weights_to_file();

  /**
   *  @brief Save the parallel fiber purkinje cell weights to binary file at
   *  given trial.
   *  @param trial the trial to save the weights at.
   */
  void save_pfpc_weights_at_trial_to_file(uint32_t trial);

  /**
   *  @brief load the parallel fiber purkinje cell weights from binary file.
   *  @param in_pfpc_file input binary file of pfpc weights
   */
  void load_pfpc_weights_from_file(std::string in_pfpc_file);

  /**
   *  @brief Save the mossy fiber to deep nucleus weights to binary file.
   */
  void save_mfdcn_weights_to_file();

  /**
   *  @brief Load the mossy fiber to deep nucleus weights from binary file.
   *  @param in_mfdcn_file Input binary file of mfdcn weights
   */
  void load_mfdcn_weights_from_file(std::string in_mfdcn_file);

  /**
   *  @brief Create the full-path filename of the output simulation file
   */
  void create_out_sim_filename();

  /**
   *  @brief Create the full-path filename of the output info file
   */
  void create_out_info_filename();

  /**
   *  @brief Create the full-path filename of the output bvi file
   */
  void create_out_bvi_filename();

  /**
   *  @brief Create the full-path filenames of cmdline-specified rasters
   *  @param rast_map Reference to map of cell type to bool which encodes
   *  whether the cell type was specified at the cmdline
   */
  void create_raster_filenames(std::map<std::string, bool> &rast_map);

  /**
   *  @brief Create the full-path filenames of cmdline-specified psth
   *  @param psth_map Reference to map of cell type to bool which encodes
   *  whether the cell type was specified at the cmdline
   */
  void create_psth_filenames(std::map<std::string, bool> &psth_map);

  /**
   *  @brief Create the full-path filenames of cmdline-specified weights
   *  @param weights_map Reference to map of cell type to bool which encodes
   *  whether the synpatic connection was specified at the cmdline
   */
  void create_weights_filenames(std::map<std::string, bool> &weights_map);

  /**
   *  @brief Create the full-path filenames of cmdline-specified con arrs
   *  @param conn_arrs_map Reference to map of synapse type to bool which
   *  encodes whether the synpatic connection was specified at the cmdline
   */
  void create_con_arrs_filenames(std::map<std::string, bool> &conn_arrs_map);

  /* initialization functions for in-run data collection */
  void initialize_rast_cell_nums();
  void initialize_cell_spikes();
  void initialize_spike_sums();
  void initialize_rasters();
  void initialize_psths();
  void initialize_psth_save_funcs();
  void initialize_raster_save_funcs();

  /**
   *  @brief Runs all trials in trials_data attribute.
   *  @param gui Pointer to gui object. Taken as a param to pass data to and
   *  from this method.
   */
  void runSession(struct gui *gui);

  /* reset functions for data collected during a session */
  void reset_spike_sums();
  void reset_rasters();
  void reset_psths();

  /* 'legacy' function to keep running count of go spikes. Mainly used for
   * tuning.
   * */
  void countGOSpikes(int *goSpkCounter);

  /**
   *  @brief Update function for all cell types spike sums. Is run every ts.
   *  @param tts The current time step
   *  @param onset_cs Onset time of the cs in ms
   *  @param offset_cs Offset time of cs in ms
   */
  void update_spike_sums(int tts, float onset_cs, float offset_cs);

  /**
   *  @brief Update cell population firing rate statistics. Is run at the end of
   *  every trial
   *  @param onset_cs Onset time of the cs in ms
   *  @param offset_cs Offset time of cs in ms
   */
  void calculate_firing_rates(float onset_cs, float offset_cs);

  /**
   *  @brief fill raster arrays from cell spike arrays. Is run every ts.
   *  @param raster_counter Counts time since the beginning of the session.
   *  @param psth_counter Counts time since the beginning of the current trial.
   */
  void fill_rasters(uint32_t raster_counter, uint32_t psth_counter);

  /* similar function to fill_rasters but for psths */
  void fill_psths(uint32_t psth_counter);

  /* save data objects to file functions */
  void save_weights();
  void save_gr_rasters_at_trial_to_file(uint32_t trial);
  void save_rasters();
  void save_psths();
  /* NOTE: for now, saving 2d arrays, only from pre-synaptic side */
  void save_con_arrs();

  /* delete data objects from memory. Only run in destructor */
  void delete_rasters();
  void delete_psths();
  void delete_spike_sums();
};

#endif /*_CONTROL_H*/
