#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_
#include <string>

#include "fileIO/build_file.h"
#include "fileIO/experiment_file.h"

const int BUILD_NUM_ARGS = 3;
const int RUN_NUM_ARGS   = 4;

const std::string INPUT_DATA_PATH = "../data/inputs/";
const std::string OUTPUT_DATA_PATH = "../data/outputs/";
const std::string DEFAULT_SIM_OUT_FILE = OUTPUT_DATA_PATH + "default_out_sim_file.sim";

const std::string EXPERIMENT_FILE_FIRST_LINE = "#Begin filetype experiment";
const std::string BUILD_FILE_FIRST_LINE      = "#begin filetype build";

enum build_args {BUILD_PROGRAM, BUILD_FILE, BUILD_OUT_SIM_FILE};
enum run_args   {RUN_PROGRAM, EXPERIMENT_FILE, IN_SIM_FILE, RUN_OUT_SIM_FILE};

enum vis_mode {GUI, TUI, NO_VIS};
enum run_mode {BUILD, RUN, NO_RUN};

bool is_file(const char *arg);
enum run_mode get_run_mode(const char *arg);
enum vis_mode get_vis_mode(const char *arg);
void validate_args_and_set_modes(int *argc, char ***argv,
	  enum vis_mode *sim_vis_mode, enum run_mode *sim_run_mode);

void parse_build_args(char ***argv, parsed_build_file &p_file);
void parse_experiment_args(char ***argv, experiment &exper);
void get_in_sim_file(char ***argv, std::string &in_file);
void get_out_sim_file(int arg_index, char ***argv, std::string &out_file);

/*
 * function: getCmdOption
 *
 * args: char** begin 		  - ptr to first char array in a range
 * 		 char** end 		  - ptr to last char array in a range
 * 		 const string& option - the command line opt as a string that 
 * 		 						we want to find
 *
 * return val: the command opt as a char ptr upon success, else 0 (NULL)
 */
char* getCmdOption(char** begin, char** end, const std::string& option);

/*
 * function: cmdOptionExists
 * 
 * args: char** begin		  - ptr to first char array in a range
 * 		 char** end			  - ptr to last char array in a range
 * 		 const string& option - the command line opt as a string
 *
 * return val: a boolean, true if the option was given, false if not
 */
bool cmdOptionExists(char** begin, char** end, const std::string& option);

#endif /* COMMAND_LINE_H_ */

