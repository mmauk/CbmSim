/*
 * File: commandline.cpp
 * Author: Sean Gallogly
 * Created on: 08/02/2022
 *
 * Description:
 *     This file implements the function prototypes in commandLine/commandline.h
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <utility>
#include "commandline.h"
#include "pstdint.h"

const std::vector<std::pair<std::string, std::string>> command_line_opts{
	{ "-v", "--visual" },
	{ "-b", "--build" },
	{ "-s", "--session" },
	{ "-i", "--input" },
	{ "-o", "--output" },
	{ "-r", "--raster" },
	{ "-w", "--weights" }
};

bool is_cmd_opt(std::string in_str)
{
	for (auto opt : command_line_opts)
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

void parse_commandline(int *argc, char ***argv, parsed_commandline &p_cl)
{
	std::vector<std::string> tokens;
	// get command line args in a vector for easier parsing.
	for (char** iter = *argv; iter != *argv + *argc; iter++)
	{
		tokens.push_back(std::string(*iter));
	}
	// first pass: get options and associated opt_params.
	for (auto opt : command_line_opts)
	{
		int first_opt_exist = cmd_opt_exists(tokens, opt.first); 
		int second_opt_exist = cmd_opt_exists(tokens, opt.second);
		int opt_sum = first_opt_exist + second_opt_exist;
		std::string this_opt;
		std::string this_param;
		char opt_char_code;
		std::vector<std::string>::iterator curr_token_iter;
		size_t div;
		std::string raster_code, raster_file_name;
		std::string weights_code, weights_file_name;
		switch (opt_sum)
		{
			case 2:
				std::cerr << "[IO_ERROR]: Specified both short and long command line option. You can specify only one\n"
						  << "[IO_ERROR]: argumnet for each command line argument type. Exiting...\n";
				exit(9);
				break;
			case 1:
				this_opt = (first_opt_exist == 1) ? opt.first : opt.second;
				// both give the same thing, it is a matter of which exists, the
				// long or the short version.
				opt_char_code = (first_opt_exist == 1) ? this_opt[1] : this_opt[2];
				this_param = get_opt_param(tokens, this_opt);
				curr_token_iter = std::find(tokens.begin(), tokens.end(), this_param);
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
						p_cl.output_sim_file = this_param;
						break;
					case 'r':
						while (curr_token_iter != tokens.end() && !is_cmd_opt(*curr_token_iter))
						{
							div = curr_token_iter->find_first_of(',');
							if (div == std::string::npos)
							{
								std::cerr << "[IO_ERROR]: Comma not given for raster argument '" << *curr_token_iter << "'. Exiting...\n";
								exit(8);
								// we have a problem, so exit
							}
							raster_code = curr_token_iter->substr(0, div);
							raster_file_name = curr_token_iter->substr(div+1);
							p_cl.raster_files[raster_code] = raster_file_name;
							curr_token_iter++;
						}
						break;
					case 'w':
						while (curr_token_iter != tokens.end() && !is_cmd_opt(*curr_token_iter))
						{
							div = curr_token_iter->find_first_of(',');
							if (div == std::string::npos)
							{
								std::cerr << "[IO_ERROR]: Comma not given for weights argument '" << *curr_token_iter << "'. Exiting...\n";
								exit(8);
								// we have a problem, so exit
							}
							weights_code = curr_token_iter->substr(0, div);
							weights_file_name = curr_token_iter->substr(div+1);
							p_cl.weights_files[weights_code] = weights_file_name;
							curr_token_iter++;
						}
						break;
				}
				break;
			case 0:
				continue;
				break;
		}
	}
	validate_commandline(p_cl);
}

void validate_commandline(parsed_commandline &p_cl)
{
	if (!p_cl.build_file.empty())
	{
		if (!p_cl.session_file.empty())
		{
			std::cerr << "[IO_ERROR]: Cannot specify both session and build file. Exiting.\n";
			exit(5);
		}
		if (p_cl.output_sim_file.empty())
		{
			std::cout << "[INFO]: Output simulation file not specified. Using default value...\n";
			p_cl.output_sim_file = DEFAULT_SIM_OUT_FILE; 
		}
		else p_cl.output_sim_file = INPUT_DATA_PATH + p_cl.output_sim_file;
		if (!p_cl.vis_mode.empty() || !p_cl.input_sim_file.empty() || !p_cl.raster_files.empty())
		{
			std::cout << "[INFO]: Ignoring additional arguments in build mode.\n";
		}
		p_cl.build_file = INPUT_DATA_PATH + p_cl.build_file;
	}
	else if (!p_cl.session_file.empty())
	{
		if (!p_cl.build_file.empty())
		{
			std::cerr << "[IO_ERROR]: Cannot specify both build and session file. Exiting.\n";
			exit(6);
		}
		if (!p_cl.output_sim_file.empty())
		{
			p_cl.output_sim_file = INPUT_DATA_PATH + p_cl.output_sim_file;
		}
		if (!p_cl.input_sim_file.empty())
		{
			p_cl.input_sim_file = INPUT_DATA_PATH + p_cl.input_sim_file;
		}
		else
		{
			std::cerr << "[IO_ERROR]: No input simulation specified in run mode. Exiting...\n";
			exit(8);
		}
		if (!p_cl.raster_files.empty())
		{
			for (auto iter = p_cl.raster_files.begin(); iter != p_cl.raster_files.end(); iter++)
			{
				iter->second = OUTPUT_DATA_PATH + iter->second;
			}
		}
		if (!p_cl.weights_files.empty())
		{
			for (auto iter = p_cl.weights_files.begin(); iter != p_cl.weights_files.end(); iter++)
			{
				iter->second = OUTPUT_DATA_PATH + iter->second;
			}
		}
		if (p_cl.vis_mode.empty())
		{
			std::cout << "[INFO]: Visual mode not specified in run mode. Setting to default value of 'TUI'.\n";
			p_cl.vis_mode = "TUI";
		}
		p_cl.session_file = INPUT_DATA_PATH + p_cl.session_file;
	}
	else
	{
		std::cerr << "[IO_ERROR]: Run mode not specified. You must provide either the {-b|--build} or {-s|--session}\n"
				  << "[IO_ERROR]: arguments. Exiting...\n";
		exit(7);
	}
}

std::string parsed_commandline_to_str(parsed_commandline &p_cl)
{
	std::stringstream p_cl_buf;
	p_cl_buf << "{ 'vis_mode', '" << p_cl.vis_mode << "' }\n";
	p_cl_buf << "{ 'build_file', '" << p_cl.build_file << "' }\n";
	p_cl_buf << "{ 'session_file', '" << p_cl.session_file << "' }\n";
	p_cl_buf << "{ 'input_sim_file', '" << p_cl.input_sim_file << "' }\n";
	p_cl_buf << "{ 'output_sim_file', '" << p_cl.output_sim_file << "' }\n";
	for (auto pair : p_cl.raster_files)
	{
		p_cl_buf << "{ '" << pair.first << "', '" << pair.second << "' }\n";
	}
	for (auto pair : p_cl.weights_files)
	{
		p_cl_buf << "{ '" << pair.first << "', '" << pair.second << "' }\n";
	}
	return p_cl_buf.str();
}

std::ostream &operator<<(std::ostream &os, parsed_commandline &p_cl)
{
	return os << parsed_commandline_to_str(p_cl);
}

