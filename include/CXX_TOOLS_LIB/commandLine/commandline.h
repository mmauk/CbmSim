#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_
#include <string>

#include "fileIO/build_file.h"
#include "fileIO/experiment_file.h"

// moved from main, should go somewhere better :/
const std::string INPUT_DATA_PATH = "../data/inputs/";
const std::string OUTPUT_DATA_PATH = "../data/outputs/";
const std::string DEFAULT_SIM_OUT_FILE = OUTPUT_DATA_PATH + "default_out_sim_file.sim";

enum vis_mode {GUI, TUI, NO_VIS};
enum run_mode {BUILD, RUN, NO_RUN};

void verify_and_assign_run_mode(int arg_index, int *argc, char ***argv, enum run_mode *sim_run_mode);
void verify_vis_mode(int arg_index, int *argc, char ***argv);
void assign_vis_mode(int arg_index, int *argc, char ***argv, enum vis_mode *sim_vis_mode);
void verify_is_file(int arg_index, char ***argv, std::string error_msg);
void verify_file_format(int arg_index, char ***argv, std::string first_line_test, std::string error_msg);
void validate_args_and_set_modes(int *argc, char ***argv,
	  enum vis_mode *sim_vis_mode, enum run_mode *sim_run_mode);

void parse_build_args(int *argc, char ***argv, parsed_build_file &p_file);
void parse_experiment_args(int *argc, char ***argv, experiment &exper);
void get_in_sim_file(int arg_index, int *argc, char ***argv, std::string &in_file);
void get_out_sim_file(int arg_index, int *argc, char ***argv, std::string &out_file);

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

