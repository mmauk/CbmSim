/*
 * File: commandline.h
 * Author: Sean Gallogly
 * Created on: 08/02/2022
 * 
 * Description:
 *     This is the interface file for parsing the command line args for cbm sim. 
 *     It includes the constants, enums, and functions required to parse the command
 *     line arguments according to a two-mode scheme for building simulations (i.e.
 *     creating new rabbits) and running simulations (i.e. training rabbits).
 */
#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_
#include <string>

#include "file_parse.h"
#include "experiment_file.h"


/* ============================ CONSTANTS =============================== */

const int BUILD_NUM_ARGS = 3;
const int RUN_NUM_ARGS   = 4;

const std::string INPUT_DATA_PATH = "../data/inputs/";
const std::string OUTPUT_DATA_PATH = "../data/outputs/";
const std::string DEFAULT_SIM_OUT_FILE = OUTPUT_DATA_PATH + "default_out_sim_file.sim";

const std::string EXPERIMENT_FILE_FIRST_LINE = "begin filetype experiment";
const std::string BUILD_FILE_FIRST_LINE      = "begin filetype build";


/* =============================== ENUMS ================================ */

enum build_args {BUILD_PROGRAM, BUILD_FILE, BUILD_OUT_SIM_FILE};
enum run_args {RUN_PROGRAM, EXPERIMENT_FILE, IN_SIM_FILE, RUN_OUT_SIM_FILE};

enum vis_mode {GUI, TUI, NO_VIS};
enum run_mode {BUILD, RUN, NO_RUN};

enum user_mode {FRIENDLY, VETERAN, NO_USER_MODE};

/* ============================ COMMANDLINE 2.0 =============================== */

typedef struct
{
	std::string vis_mode;
	std::string build_file;
	std::string experiment_file;
	std::string input_sim_file;
	std::string output_sim_file;
	std::map<std::string, std::string> raster_files;
} parsed_commandline;

void parse_commandline(int *argc, char ***argv, parsed_commandline &p_cl);

void print_parsed_commandline(parsed_commandline &p_cl);

/* ============================ COMMANDLINE 2.0 =============================== */

/* ======================= FUNCTION DECLARATIONS ======================= */

/*
 * Description:
 *     Checks whether the c-string arg is the name (without full path info)
 *     of a valid file. Does not check the type nor contents of the file.
 *
 */
bool is_file(const char *arg);

/*
 * Description:
 *     Obtains the run mode {RUN, BUILD} as a run_mode enum, declared above,
 *     from the argument arg. Does not check whether arg is a file or not:
 *     that check is assumed to be done before this.  
 *
 */
enum run_mode get_run_mode(const char *arg);

/*
 * Description:
 *     Obtains the visualization mode {TUI, GUI} as a vis_mode enum, declared above,
 *     from the argument arg. Does not check whether the arg is a file or not:
 *     that check is assumed to be done before this. It is assumed that this fnctn
 *     is used in a branch relating only to run mode, as we assume building involves
 *     only a TUI.
 *
 */
enum vis_mode get_vis_mode(const char *arg);

/*
 * Description:
 *     Main function which checks the number of arguments and validates them according
 *     to the following two-mode scheme:
 *
 *             1) Build Mode
 *
 *                 ./big_sim build_file out_sim_file
 *
 *             2) Run Mode
 *
 *                 ./big_sim experiment_file in_sim_file out_sim_file
 *     
 *     For either mode, none of the arguments is optional. If any argument is missing, is
 *     an invalid file, or has the wrong format (e.g. build_file or experiment_file) an
 *     error message is displayed and the program exits with non-zero status.
 *
 */
void validate_args_and_set_modes(int *argc, char ***argv,
	  enum vis_mode *sim_vis_mode, enum run_mode *sim_run_mode,
	  enum user_mode *sim_user_mode);

/*
 * Description:
 *     Parses the given build file into the parsed_build_file structure,
 *     defined in build_file.h. It is assumed that this function will be called
 *     in some branch relating to build mode. For more information on how a
 *     build file is parsed, please look at build_file.h and build_file.cpp.
 *
 */
void parse_build_args(char ***argv, parsed_build_file &p_file);

/*
 * Assigns the full input experiment file path name to in_file. Assumes the file
 * is contained in the correct order within *argv. Should be called after validate_args_and_set_modes
 *
 */
void get_in_expt_file(char ***argv, std::string &in_file);

/*
 * Description:
 *     Assigns the full input simulation file name, which is contained within one of argv,
 *     to the string in_file. Assumes that the input simulation file is given in argv and
 *     in the correct placement relative to the other command line args (see scheme above).
 *     This function should be called in a branch relating to run mode, as build mode does
 *     not require an input simulation file.
 *
 */
void get_in_sim_file(char ***argv, std::string &in_file);

/*
 * Description:
 *     Assigns the full output simulation file name, which is contained within one of argv,
 *     to the string out_file. Assumes that the output simulation file is given in argv and
 *     in the correct placement relative to the other command line args (see scheme above).    
 *
 */
void get_out_sim_file(int arg_index, char ***argv, std::string &out_file);

#endif /* COMMAND_LINE_H_ */

