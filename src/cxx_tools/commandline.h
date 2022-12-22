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
#include "file_utility.h"

/*
 * The main data structure for this module. collects all relevant commandline information
 * here, in string representations. the raster, psth, and weights files are maps from
 * cell ids (or synapse ids) to booleans, which will be used downstream to determine
 * what data structures and filenames need to be constructed downstream
 *
 */
typedef struct
{
	std::string cmd_name;
	std::string print_help;
	std::string vis_mode;
	std::string build_file;
	std::string session_file;
	std::string input_sim_file;
	std::string output_sim_file;
	std::string output_basename;
	std::string pfpc_plasticity;
	std::string mfnc_plasticity;
	std::map<std::string, bool> raster_files;
	std::map<std::string, bool> psth_files;
	std::map<std::string, bool> weights_files;
} parsed_commandline;

/*
 * Description:
 *     fills the reference to the parsed_commandline from the tokens found 
 *     in the commandline. Some validation is done at this step, but most
 *     validation of the filled p_cl struct is deferred to validate_commandline
 */
void parse_commandline(int *argc, char ***argv, parsed_commandline &p_cl);

/*
 * Description:
 *     performs the majority of validity checks on the filled struct p_cl. Most 
 *     invalid states that p_cl could be in cause the program to terminate. However,
 *     some options (such as not specifying the visual mode) have default values which
 *     are deferred to in cases in which their options are omitted entirely. For a detailed
 *     list of the commandline options run the executable with the '-h' flag.
 *
 */
void validate_commandline(parsed_commandline &p_cl);

/*
 * Description:
 *     a convenience wrapper for the functions 'parse_commandline' and 'validate_commandline'
 */
void parse_and_validate_parsed_commandline(int *argc, char ***argv, parsed_commandline &p_cl);

/*
 * Description:
 *     copies the data in 'from_p_cl' struct to 'to_p_cl' struct. Used only in copying data
 *     to the contained instance in info_file (current as of 12/21/2022)
 */
void cp_parsed_commandline(parsed_commandline &from_p_cl, parsed_commandline &to_p_cl);

/*
 * Description:
 *     converts the struct 'p_cl' into a string representation
 */
std::string parsed_commandline_to_str(parsed_commandline &p_cl);

/* Description:
 *     overloads the stream insertion operator (<<) for the parsed_commandline struct.
 *     With this overload, if you have a parsed_commandline called p_cmdline, you can print
 *     its contents like so:
 *
 *     std::cout << p_cmdline << std::endl;
 *
 */
std::ostream &operator<<(std::ostream &os, parsed_commandline &p_cl);

#endif /* COMMAND_LINE_H_ */

