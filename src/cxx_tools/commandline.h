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

const std::string INPUT_DATA_PATH      = "../data/inputs/";
const std::string OUTPUT_DATA_PATH     = "../data/outputs/";
const std::string DEFAULT_SIM_OUT_FILE = OUTPUT_DATA_PATH + "default_out_sim_file.sim";

typedef struct
{
	std::string print_help;
	std::string vis_mode;
	std::string build_file;
	std::string session_file;
	std::string input_sim_file;
	std::string output_sim_file;
	std::string pfpc_plasticity;
	std::string mfnc_plasticity;
	std::map<std::string, std::string> raster_files;
	std::map<std::string, std::string> psth_files;
	std::map<std::string, std::string> weights_files;
} parsed_commandline;

void parse_commandline(int *argc, char ***argv, parsed_commandline &p_cl);

void validate_commandline(parsed_commandline &p_cl);

void parse_and_validate_parsed_commandline(int *argc, char ***argv, parsed_commandline &p_cl);

std::string parsed_commandline_to_str(parsed_commandline &p_cl);

std::ostream &operator<<(std::ostream &os, parsed_commandline &p_cl);


#endif /* COMMAND_LINE_H_ */

