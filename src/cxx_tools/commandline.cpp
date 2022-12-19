/*
 * File: commandline.cpp
 * Author: Sean Gallogly
 * Created on: 08/02/2022
 *
 * Description:
 *     This file implements the function prototypes in commandLine/commandline.h
 */

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <utility>
#include <cstdint>

#include "logger.h"
#include "commandline.h"

// TODO: place in some global place, or convenience file, as there are duplicates across
// translation units now
const int NUM_CELL_TYPES = 8;
const std::string CELL_IDS[NUM_CELL_TYPES] = {"MF", "GR", "GO", "BC", "SC", "PC", "IO", "NC"}; 

const std::vector<std::string> command_line_single_opts
{
	"--pfpc-off",
	"--mfnc-off",
	"--binary",
	"--cascade",
};

const std::vector<std::pair<std::string, std::string>> command_line_pair_opts 
{
	{ "-h", "--help"    },
	{ "-v", "--visual"  },
	{ "-b", "--build"   },
	{ "-s", "--session" },
	{ "-i", "--input"   },
	{ "-o", "--output"  },
	{ "-r", "--raster"  },
	{ "-p", "--psth"    },
	{ "-w", "--weights" }
};

bool is_cmd_opt(std::string in_str)
{
	for (auto opt : command_line_pair_opts)
	{
		if (in_str == opt.first || in_str == opt.second) return true;
	}
	return false;
}

int cmd_opt_exists(std::vector<std::string> &token_buf, std::string opt)
{
	return (std::find(token_buf.begin(), token_buf.end(), opt) != token_buf.end()) ? 1 : 0;
}

std::string get_opt_param(std::vector<std::string> &token_buf, std::string opt)
{
	auto tp = std::find(token_buf.begin(), token_buf.end(), opt);
	if (tp != token_buf.end() && tp++ != token_buf.end())
	{
		return std::string(*tp);
	}
	return "";
}

void fill_opt_map(std::map<std::string, bool> &opt_map, std::string opt, std::string param)
{
	// if the parameter is a single cell id or weight id, take case into account separately
	if (param.length() == 2 || param.length() == 4)
	{
		opt_map[param] = true;
	}
	else 
	{
		size_t prev_div = 0;
		size_t curr_div = param.find_first_of(',');
		while (curr_div != std::string::npos)
		{
			opt_map[param.substr(prev_div, curr_div-prev_div)] = true; // creates this new entry
			prev_div = curr_div + 1;
			curr_div = param.find(',', curr_div+1);
		}
		if (opt_map.empty()) // if we didn't fill the map at all, then we weren't able to find valid tokens
		{
			std::cerr << "[IO_ERROR]: Invalid parameter for option '" << opt << "'. Exiting...\n";
			exit(8);
		}
		opt_map[param.substr(prev_div, curr_div)] = true; // take into account last item missed
	}
}

void print_usage_info()
{
	std::cout << "Usage: ./cbm_sim [options]\n";
	std::cout << "Options:\n";
	std::cout << std::right << std::setw(10) << "\t-h, --help" << "\t\tprint this information and exit\n";
	std::cout << std::right << std::setw(20) << "\t-v, --visual [TUI|GUI]" << "\tspecify the visual mode the simulation is run in\n";
	std::cout << std::right << std::setw(20) << "\t-b, --build [FILE]" << "\tsets the simulation to build a bunny using FILE as the build file\n";
	std::cout << std::right << std::setw(20) << "\t-s, --session [FILE]" << "\tsets the simulation to run a session using FILE as the session file\n";
	std::cout << std::right << std::setw(20) << "\t-i, --input [FILE]" << "\tspecify the input simulation file\n";
	std::cout << std::right << std::setw(20) << "\t-o, --output [FILE]" << "\tspecify the output simulation file\n";
	std::cout << std::right << std::setw(10) << "\t--pfpc-off|--binary|--cascade" << "\tturns off or sets PFPC plasticity mode; options are mutually exclusive and work as follows:\n\n";
	std::cout << "\t\t\t\t \t--pfpc-off - turns PFPC plasticity off\n";
	std::cout << "\t\t\t\t \t--binary - turns PFPC plasticity on and sets the type of plasticity to 'dual' ie 'binary'\n";
	std::cout << "\t\t\t\t \t--cascade - turns PFPC plasticity on and sets the type of plasticity to 'cascade'\n\n";
	std::cout << std::right << std::setw(10) << "\t" << "\t\t\tif none of these options is given, PFPC plasticity is turned on and set to 'graded' by default\n\n";
	std::cout << std::right << std::setw(10) << "\t--mfnc-off" << "\t\tturns off MFNC plasticity; if not included, MFNC plasticity is turned on and set to 'graded' by default\n";
	std::cout << "\t-r, --raster {[CODE],[FILE]} space-separated list of cell types and raster files to be saved for that cell type. Possible CODEs are:\n\n";
	std::cout << "\t\t\t\t \tMF - Mossy Fiber\n";
	std::cout << "\t\t\t\t \tGR - Granule Cell\n";
	std::cout << "\t\t\t\t \tGO - Golgi Cell\n";
	std::cout << "\t\t\t\t \tSC - Stellate Cell\n";
	std::cout << "\t\t\t\t \tBC - Basket Cell\n";
	std::cout << "\t\t\t\t \tPC - Purkinje Cell\n";
	std::cout << "\t\t\t\t \tNC - Deep Nucleus Cell\n";
	std::cout << "\t\t\t\t \tIO - Inferior Olive Cell\n\n";
	std::cout << "\t-p, --psth {[CODE],[FILE]} space-separated list of cell types and psth files to be saved for that cell type. Possible CODEs are identical with those for rasters.\n\n";
	std::cout << "\t-w, --weights {[CODE],[FILE]} space-separated list of weights and weights files to be saved. Possible CODEs are:\n\n";
	std::cout << "\t\t\t\t  \tPFPC - parallel-fiber to purkinje synapse\n";
	std::cout << "\t\t\t\t  \tMFNC - mossy-fiber to deep nucleus synapse\n\n";
	std::cout << "Example usage:\n\n";
	std::cout << "1) uses file 'build_file.bld' to construct a bunny, which is saved to file 'bunny.sim':\n\n";
	std::cout << "\t./cbm_sim -b build_file.bld -o bunny.sim\n\n";
	std::cout << "2) uses file 'acquisition.sess' to train the input simulation 'bunny.sim' with PFPC plasticity on and set to graded and MFNC plasticity off:\n\n";
	std::cout << "\t./cbm_sim -s acquisition.sess -i bunny.sim --mfnc-off\n\n";
	std::cout << "3) uses file 'acquisition.sess' to train the input simulation 'bunny.sim' with PFPC and MFNC plasticity on and set to graded;\n";
	std::cout << "   PC, SC, and BC rasters are saved to files 'allPCRaster.bin' 'allSCRaster.bin' and 'allBCRaster.bin' respectively:\n\n";
	std::cout << "\t./cbm_sim -s acquisition.sess -i bunny.sim -r PC,allPCRaster SC,allSCRaster BC,allBCRaster\n\n";
}


void parse_commandline(int *argc, char ***argv, parsed_commandline &p_cl)
{
	std::vector<std::string> tokens;
	// get command line args in a vector for easier parsing.
	for (char** iter = *argv; iter != *argv + *argc; iter++)
	{
		tokens.push_back(std::string(*iter));
	}

	for (auto opt : command_line_single_opts)
	{
		if (cmd_opt_exists(tokens, opt) == 1)
		{
			if (opt == "--pfpc-off")
				p_cl.pfpc_plasticity = "off";
			else if (opt == "--mfnc-off")
				p_cl.mfnc_plasticity = "off";
			else if (opt == "--binary")
				p_cl.pfpc_plasticity = "binary";
			else if (opt == "--cascade")
				p_cl.pfpc_plasticity = "cascade";
		}
	}

	// first pass: get options and associated opt_params.
	for (auto opt : command_line_pair_opts)
	{
		int first_opt_exist = cmd_opt_exists(tokens, opt.first); 
		int second_opt_exist = cmd_opt_exists(tokens, opt.second);
		int opt_sum = first_opt_exist + second_opt_exist;
		std::string this_opt;
		std::string this_param;
		char opt_char_code;
		std::vector<std::string>::iterator curr_token_iter = tokens.begin();
		p_cl.cmd_name = *curr_token_iter; //TODO; double check this
		switch (opt_sum)
		{
			case 2:
				LOG_FATAL("Specified both short and long command line option.");
				LOG_FATAL("You can specify only one argument for each command line argument type. Exiting...");
				exit(9);
				break;
			case 1:
				this_opt = (first_opt_exist == 1) ? opt.first : opt.second;
				// both give the same thing, it is a matter of which exists, the
				// long or the short version.
				opt_char_code = (first_opt_exist == 1) ? this_opt[1] : this_opt[2];
				if (opt_char_code == 'h') p_cl.print_help = "help";
				else 
				{
					this_param = get_opt_param(tokens, this_opt);
					curr_token_iter = std::find(tokens.begin(), tokens.end(), this_param);
				}
				switch (opt_char_code)
				{
					case 'v':
						p_cl.vis_mode = this_param;
						break;
					case 'b':
						p_cl.build_file = this_param;
						break;
					case 's':
						p_cl.session_file = this_param;
						break;
					case 'i':
						p_cl.input_sim_file = this_param;
						break;
					case 'o':
						// FIXME: for now, regardless mode, add -o arg to both names. later, may deprecated one over other
						p_cl.output_sim_file = this_param;
						p_cl.output_basename = this_param;
						break;
					case 'r':
						fill_opt_map(p_cl.raster_files, this_opt, this_param);
						break;
					case 'p':
						fill_opt_map(p_cl.psth_files, this_opt, this_param);
						break;
					case 'w':
						fill_opt_map(p_cl.weights_files, this_opt, this_param);
						break;
				}
				break;
			case 0:
				continue;
				break;
		}
	}
}

// I am sorry in advance for this implementation. C++ doesn't have reflection. A shame.
bool p_cmdline_is_empty(parsed_commandline &p_cl)
{
	return p_cl.print_help.empty() && p_cl.vis_mode.empty() &&
	   p_cl.build_file.empty() && p_cl.session_file.empty() &&
	   p_cl.input_sim_file.empty() && p_cl.output_sim_file.empty() &&
	   p_cl.output_basename.empty() && p_cl.pfpc_plasticity.empty() &&
	   p_cl.mfnc_plasticity.empty() && p_cl.raster_files.empty() &&
	   p_cl.psth_files.empty() && p_cl.weights_files.empty();
}

void validate_commandline(parsed_commandline &p_cl)
{
	if (p_cmdline_is_empty(p_cl))
	{
		p_cl.vis_mode = "GUI";
		// FIXME: for now, will not assign to either -o options. no args, assume want run mode in gui
		// later, will defer run mode to only tui, as gui callbacks do not require such checks
		p_cl.mfnc_plasticity = "graded"; 
		p_cl.mfnc_plasticity = "graded";
	}
	else
	{
		/* for now, print usage info regardless of other arguments.
		 * in future, if there is a commandline error, print usage info then exit
		 */
		if (!p_cl.print_help.empty())
		{
			print_usage_info();
			exit(0);
		}
		if (!p_cl.build_file.empty())
		{
			if (!p_cl.session_file.empty())
			{
				LOG_FATAL("Cannot specify both session and build file. Exiting...");
				exit(5);
			}
			if (p_cl.output_sim_file.empty())
			{
				LOG_DEBUG("Output simulation file not specified. Using default value...");
				p_cl.output_sim_file = DEFAULT_SIM_OUT_FILE; 
			}
			else p_cl.output_sim_file = INPUT_DATA_PATH + p_cl.output_sim_file;
			if (p_cl.vis_mode.empty())
			{
				LOG_DEBUG("Visual mode not specified. Setting to default value of 'TUI'...");
				p_cl.vis_mode = "TUI";
			}
			else if (p_cl.vis_mode == "GUI")
			{
				LOG_FATAL("Cannot specify visual mode 'GUI' in build mode. Exiting...");
				exit(7);
			}
			if (!p_cl.input_sim_file.empty() || !p_cl.raster_files.empty())
			{
				LOG_DEBUG("Ignoring additional arguments in build mode.");
			}
			p_cl.build_file = INPUT_DATA_PATH + p_cl.build_file;
		}
		else if (!p_cl.session_file.empty())
		{
			if (!p_cl.build_file.empty())
			{
				LOG_FATAL("Cannot specify both build and session file. Exiting.");
				exit(6);
			}
			if (p_cl.output_basename.empty())
			{
				LOG_FATAL("You must specify an output basename. Exiting...");
				exit(7);
			}
			// TODO: make a more general algorithm which searches data/ dir for file with this name 
			if (!p_cl.input_sim_file.empty())
			{
				p_cl.input_sim_file = INPUT_DATA_PATH + p_cl.input_sim_file;
			}
			else
			{
				LOG_FATAL("No input simulation specified in run mode. Exiting...");
				exit(8);
			}
			if (p_cl.pfpc_plasticity.empty())
			{
				LOG_DEBUG("Turning PFPC plasticity on to default mode 'graded'...");
				p_cl.pfpc_plasticity = "graded";
			}
			else
			{
				// just notify user what we already set above
				if (p_cl.pfpc_plasticity == "dual") LOG_DEBUG("Turning PFPC plasticity on in 'dual' mode...");
				else if (p_cl.pfpc_plasticity == "cascade") LOG_DEBUG("Turning PFPC plasticity on in 'cascade' mode...");
				else if (p_cl.pfpc_plasticity == "off") LOG_DEBUG("Turning PFPC plasticity off..");
			}
			if (p_cl.mfnc_plasticity.empty())
			{
				LOG_DEBUG("Turning MFNC plasticity on to default mode 'graded'...");
				p_cl.mfnc_plasticity = "graded";
			}
			else if (p_cl.mfnc_plasticity == "off") LOG_DEBUG("Turning MFNC plasticity off...");
			if (p_cl.vis_mode.empty())
			{
				LOG_DEBUG("Visual mode not specified in run mode. Setting to default value of 'TUI'...");
				p_cl.vis_mode = "TUI";
			}
			p_cl.session_file = INPUT_DATA_PATH + p_cl.session_file;
		}
		else
		{
			LOG_FATAL("Run mode not specified. You must provide either {-b|--build} or {-s|--session} arguments. Exiting...");
			exit(7);
		}
	}
}

void parse_and_validate_parsed_commandline(int *argc, char ***argv, parsed_commandline &p_cl)
{
	parse_commandline(argc, argv, p_cl);
	validate_commandline(p_cl);
}

void cp_parsed_commandline(parsed_commandline &from_p_cl, parsed_commandline &to_p_cl)
{
	to_p_cl.cmd_name        = from_p_cl.cmd_name;
	to_p_cl.print_help      = from_p_cl.print_help;
	to_p_cl.vis_mode        = from_p_cl.vis_mode;
	to_p_cl.build_file      = from_p_cl.build_file;
	to_p_cl.session_file    = from_p_cl.session_file;
	to_p_cl.input_sim_file  = from_p_cl.input_sim_file;
	to_p_cl.output_sim_file = from_p_cl.output_sim_file;
	to_p_cl.output_basename = from_p_cl.output_basename;
	to_p_cl.pfpc_plasticity = from_p_cl.pfpc_plasticity;
	to_p_cl.mfnc_plasticity = from_p_cl.mfnc_plasticity;
	
	to_p_cl.raster_files  = from_p_cl.raster_files;
	to_p_cl.psth_files    = from_p_cl.psth_files;
	to_p_cl.weights_files = from_p_cl.weights_files;
}

std::string parsed_commandline_to_str(parsed_commandline &p_cl)
{
	std::stringstream p_cl_buf;
	p_cl_buf << "{ 'print_help', '" << p_cl.print_help << "' }\n";
	p_cl_buf << "{ 'vis_mode', '" << p_cl.vis_mode << "' }\n";
	p_cl_buf << "{ 'build_file', '" << p_cl.build_file << "' }\n";
	p_cl_buf << "{ 'session_file', '" << p_cl.session_file << "' }\n";
	p_cl_buf << "{ 'input_sim_file', '" << p_cl.input_sim_file << "' }\n";
	p_cl_buf << "{ 'output_sim_file', '" << p_cl.output_sim_file << "' }\n";
	p_cl_buf << "{ 'output_basename', '" << p_cl.output_basename << "' }\n";
	p_cl_buf << "{ 'pfpc_plasticity', '" << p_cl.pfpc_plasticity << "' }\n";
	p_cl_buf << "{ 'mfnc_plasticity', '" << p_cl.mfnc_plasticity << "' }\n";
	p_cl_buf << "{ 'raster_files' :\n";
	for (auto pair : p_cl.raster_files)
	{
		p_cl_buf << "{ '" << pair.first << "', '" << pair.second << "' }\n";
	}
	p_cl_buf << "}\n";
	p_cl_buf << "{ 'psth_files' :\n";
	for (auto pair : p_cl.psth_files)
	{
		p_cl_buf << "{ '" << pair.first << "', '" << pair.second << "' }\n";
	}
	p_cl_buf << "}\n";
	p_cl_buf << "{ 'weights_files' :\n";
	for (auto pair : p_cl.weights_files)
	{
		p_cl_buf << "{ '" << pair.first << "', '" << pair.second << "' }\n";
	}
	p_cl_buf << "}\n";
	return p_cl_buf.str();
}

std::ostream &operator<<(std::ostream &os, parsed_commandline &p_cl)
{
	return os << parsed_commandline_to_str(p_cl);
}

