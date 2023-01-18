/*
 * File: file_parse.cpp
 * Author: Sean Gallogly
 * Created on: 10/02/2022
 *
 * Description:
 *     This file implements the function prototypes in file_parse.h
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

#include "logger.h"
#include "file_parse.h"

// regex strings for matching variable identifiers and variable values
static const std::string var_id_regex_str = "[a-zA-Z_]{1}[a-zA-Z0-9_]*";
static const std::string var_val_regex_str = "[+-]?([0-9]*[.])?[0-9]*([e][+-]?[0-9]+)?";

// look-up table for lexemes, is used for printing lexemes to whatever stream you want
static std::map<lexeme, std::string> lex_string_look_up =
{
		{ NONE, "NONE" },
		{ BEGIN_MARKER, "BEGIN_MARKER" },
		{ END_MARKER, "END_MARKER"},
		{ REGION, "REGION" },
		{ REGION_TYPE, "REGION_TYPE" },
		{ TYPE_NAME, "TYPE_NAME" },
		{ VAR_IDENTIFIER, "VAR_IDENTIFIER" },
		{ VAR_VALUE, "VAR_VALUE" },
		{ DEF, "DEF" },
		{ DEF_TYPE, "DEF_TYPE" },
		{ SINGLE_COMMENT, "SINGLE_COMMENT" },
		{ DOUBLE_COMMENT_BEGIN, "DOUBLE_COMMENT_BEGIN" },
		{ DOUBLE_COMMENT_END, "DOUBLE_COMMENT_END" },
};

// definitions of tokens via their lexemes
static std::map<std::string, lexeme> token_defs =
{
		{ "begin", BEGIN_MARKER },
		{ "end", END_MARKER },
		{ "filetype", REGION },
		{ "section", REGION },
		{ "build", REGION_TYPE }, // might be deprecated
		{ "run", REGION_TYPE },
		{ "connectivity", REGION_TYPE },
		{ "activity", REGION_TYPE },
		{ "trial_def", REGION_TYPE },
		{ "mf_input", REGION_TYPE },
		{ "trial_spec", REGION_TYPE },
		{ "activity", REGION_TYPE },
		{ "int", TYPE_NAME },
		{ "float", TYPE_NAME },
		{ "[a-zA-Z_]{1}[a-zA-Z0-9_]*", VAR_IDENTIFIER },
		{ "[+-]?([0-9]*[.])?[0-9]*([e][+-]?[0-9]+)?", VAR_VALUE },
		{ "def", DEF },
		{ "trial", DEF_TYPE },
		{ "block", DEF_TYPE },
		{ "session", DEF_TYPE },
		{ "//", SINGLE_COMMENT },
		{ "/*", DOUBLE_COMMENT_BEGIN },
		{ "*/", DOUBLE_COMMENT_END },
};

/*
 * Implementation Notes:
 *     this function uses an input file stream to basically loop over
 *     the lines of the input file, creating a vector of strings which
 *     represents a tokenized line, and then adding this to a vector
 *     which represents the tokenized file. I make these changes in place
 *     (ie to an object passed in by reference) because the caller has
 *     ownership of the object at the end of the day, so it should create
 *     it too.
 */
void tokenize_file(std::string in_file, tokenized_file &t_file)
{
	std::ifstream in_file_buf(in_file.c_str());
	if (!in_file_buf.is_open())
	{
		LOG_FATAL("Could not open file '%s'. Exiting...", in_file.c_str());
		exit(3);
	}

	std::string line = "";
	while (getline(in_file_buf, line))
	{
		std::vector<std::string> line_tokens;
		if (line == "") continue;
		std::stringstream line_buf(line);
		std::string token;
		while (line_buf >> token)
		{
			line_tokens.push_back(token);
		}
		t_file.tokens.push_back(line_tokens);
	}
	in_file_buf.close();
}

/*
 * Implementation Notes:
 *     This function operates upon a tokenized file by going line-by-line, token-by-token
 *     and matching the token with its lexical-definition. The only tricky tokens are the
 *     variable identifiers and the variable values, as these are naturally variable from
 *     one variable to the next. First I search for the token as-is in the token definitions
 *     look-up table. This ensures that I'm not matching a constant lexical token with 
 *     a variable one. Then I attempt to match, in sequence, the raw_token with the regex
 *     pattern string associated with variable identifiers and variable values. Finally,
 *     I add the relevant data into each lexed_token, add that lexed_token to l_file's tokens vector.
 *     Note that I add an "artificial" EOL token after each line has been lexed. Used in parsing.
 */
void lex_tokenized_file(tokenized_file &t_file, lexed_file &l_file)
{
	std::regex var_id_regex(var_id_regex_str);
	std::regex var_val_regex(var_val_regex_str);
	for (auto line : t_file.tokens)
	{
		for (auto raw_token : line)
		{
			lexed_token l_token;
			auto entry_exists = token_defs.find(raw_token);
			if (entry_exists != token_defs.end())
			{
				l_token.lex = token_defs[raw_token];
			}
			else
			{
				if (std::regex_match(raw_token, var_id_regex))
				{
					l_token.lex = token_defs[var_id_regex_str];
				}
				else if (std::regex_match(raw_token, var_val_regex))
				{
					l_token.lex = token_defs[var_val_regex_str];
				}
			}
			l_token.raw_token = raw_token;
			l_file.tokens.push_back(l_token);
		}
		l_file.tokens.push_back({ NEW_LINE, "\n" }); /* push an artificial new line token at end of line */
	}
}

/*
 * Implementation Notes:
 *     meant to be called within parse_trial_section. Loops until the end lex is found. Is used for
 *     each of the types of defintions which are possible: 1) trial definitions, 2) block definitions,
 *     and 3) session definitions. To serve these 3 purposes we need to initialize a lot of temporary
 *     data structures, which may or may not be used depending on the current definition type being
 *     parsed. 
 *
 *     Collects the required information for the given definition type. In order to be compatible with
 *     "statements" within the definition consisting of either single trial or block identifiers, we
 *     add a "shadow token" within the l_file, which conceptually adds in the value of "1" for that trial
 *     or block identifier.
 */
static void parse_def(std::vector<lexed_token>::iterator &ltp,
                      lexed_file &l_file,
                      parsed_sess_file &s_file,
                      std::string def_type,
                      std::string def_label)
{
	std::pair<std::string, std::string> curr_pair = {};
	std::map<std::string, variable> curr_trial = {};
	std::vector<std::pair<std::string, std::string>> curr_block = {};
	lexeme prev_lex = NONE;
	variable curr_var = {};
	while (ltp->lex != END_MARKER)
	{
		switch (ltp->lex)
		{
			case TYPE_NAME:
				if (def_type == "trial") curr_var.type_name = ltp->raw_token;
				else
				{
					// TODO: report error
				}
				break;
			case VAR_IDENTIFIER:
				if (def_type == "trial")
				{
					if (prev_lex != TYPE_NAME)
					{
						// TODO: report error
					}
					else curr_var.identifier = ltp->raw_token;
				}
				else
				{
					/* disgusting hack: inserts a 'shadow token'
					 * so that if the prev identifier didnt include
					 * a value immediately proceeding it, a value
					 * is inserted into the vector to act as if there
					 * was one.
					 */
					if (curr_pair.first != "")
					{
						lexed_token shadow_token = { VAR_VALUE, "1" };
						ltp = l_file.tokens.insert(ltp, shadow_token);
						ltp--;
					}
					else curr_pair.first = ltp->raw_token;
				}
				break;
			case VAR_VALUE:
				if (prev_lex != VAR_IDENTIFIER
					&& prev_lex != NEW_LINE)
				{
					// TODO: report error
				}
				else
				{
					if (def_type == "trial")
					{
						curr_var.value = ltp->raw_token;
						curr_trial[curr_var.identifier] = curr_var;
						curr_var = {};
					}
					else
					{
						curr_pair.second = ltp->raw_token;
						if (def_type == "block")
						{
							curr_block.push_back(curr_pair);
						}
						else if (def_type == "session")
						{
							// TODO: should check if curr_pair.first is either a block or a trial identifier (at a later stage of processing)
							s_file.parsed_trial_info.session.push_back(curr_pair);
						}
						curr_pair = {};
					}
				}
				break;
			case SINGLE_COMMENT:
				while (ltp->lex != NEW_LINE) ltp++;
			break;
		}
		prev_lex = ltp->lex;
		ltp++;
		// ensures a single identifier in a def block is added to the right structure
		if (ltp->lex == END_MARKER
			&& def_type != "trial"
			&& curr_pair.first != ""
			&& curr_pair.second == "")
		{
			curr_pair.second = "1";
			if (def_type == "block") curr_block.push_back(curr_pair);
			else if (def_type == "session") s_file.parsed_trial_info.session.push_back(curr_pair);
		}
	}
	if (def_type == "trial") s_file.parsed_trial_info.trial_map[def_label] = curr_trial;
	else if (def_type == "block") s_file.parsed_trial_info.block_map[def_label] = curr_block;
}

/*
 * Implementation Notes:
 *     meant for parsing var sections in parsed_session files. loops until the end lex {END_MARKER, "end"} is
 *     found. Within the loop current variable attributes are collected and added to the (temporary) current section.
 *     Once the end marker is reached, the current section is added to the session file's parsed_var_sections map.
 */
static void parse_var_section(std::vector<lexed_token>::iterator &ltp,
                              lexed_file &l_file,
                              parsed_sess_file &s_file,
                              std::string region_type)
{
	parsed_var_section curr_section = {};
	variable curr_var = {};
	while (ltp->lex != END_MARKER)
	{
		if (ltp->lex == TYPE_NAME)
		{
			auto next_lt = std::next(ltp, 1);
			auto second_next_lt = std::next(ltp, 2);
			if (next_lt->lex == VAR_IDENTIFIER
				&& second_next_lt->lex == VAR_VALUE)
			{
				curr_var.type_name  = ltp->raw_token;
				curr_var.identifier = next_lt->raw_token;
				curr_var.value      = second_next_lt->raw_token;

				curr_section.param_map[next_lt->raw_token] = curr_var;
				curr_var = {};
				ltp += 2;
			}
		}

		else if (ltp->lex == SINGLE_COMMENT)
		{
			while (ltp->lex != NEW_LINE) ltp++;
		}
		ltp++;
	}
	s_file.parsed_var_sections[region_type] = curr_section;
}

/*
 * Implementation Notes:
 *     loops over lexes until END_MARKER is found. checks whether a 3-lex sequence definition header is found,
 *     and if so, calls parse_def. Ignores single comments by passing over the lexes until the new line lex is found.
 */
static void parse_trial_section(std::vector<lexed_token>::iterator &ltp, lexed_file &l_file, parsed_sess_file &s_file)
{
	while (ltp->lex != END_MARKER) // parse this section
	{
		if (ltp->lex == DEF)
		{
			auto next_lt = std::next(ltp, 1);
			auto second_next_lt = std::next(ltp, 2);
			if (next_lt->lex == DEF_TYPE
				&& second_next_lt->lex == VAR_IDENTIFIER)
			{
				ltp += 4;
				parse_def(ltp, l_file, s_file, next_lt->raw_token, second_next_lt->raw_token);
			}
			else {} // TODO: report error
		}
		else if (ltp->lex == SINGLE_COMMENT)
		{
			while (ltp->lex != NEW_LINE) ltp++;
		}
		ltp++;
	}
}

/*
 * Implementation Notes:
 *     a region is defined as a code block in a .sess file which begins with "begin" and ends with "end."
 *     This recursive function is run on regions in a parsed sess file: it checks the region type (there
 *     are two recognized region types) and calls parse_var_section or parse_trial_section, otherwise loops
 *     over lexes until it encounters a valid sequence of begin marker, region, and region type.
 *
 */
static void parse_region(std::vector<lexed_token>::iterator &ltp,
                         lexed_file &l_file,
                         parsed_sess_file &s_file,
                         std::string region_type)
{
	if (region_type == "mf_input"
		|| region_type == "activity"
		|| region_type == "trial_spec")
	{
		parse_var_section(ltp, l_file, s_file, region_type);
	}
	else if (region_type == "trial_def")
	{
		parse_trial_section(ltp, l_file, s_file);
	}
	else
	{
		while (ltp->lex != END_MARKER)
		{
			auto next_lt = std::next(ltp, 1);
			auto second_next_lt  = std::next(ltp, 2);
			if (ltp->lex == BEGIN_MARKER
				&& next_lt->lex == REGION
				&& second_next_lt->lex == REGION_TYPE)
			{
				ltp += 4;
				parse_region(ltp, l_file, s_file, second_next_lt->raw_token);
			}
			else if (ltp->lex == SINGLE_COMMENT)
			{
				while (ltp->lex != NEW_LINE) ltp++;
			}
			ltp++;
		}
	}
}

/*
 * Implementation Notes:
 *     the main function for parsing lexes within l_file, assuming l_file represents a
 *     lexed session file. First loops over lexes until it encounters the following sequence of tokens:
 *     "begin filetype run" TODO: change filetype to session. Then loops over lexes until trial section
 *     or variable section 3-lex sequence headers are found. If either is found, enters the appropriate
 *     parsing subroutine.
 *
 */
void parse_lexed_sess_file(lexed_file &l_file, parsed_sess_file &s_file)
{
	// NOTE: ltp stands for [l]exed [t]oken [p]ointer, not long-term potentiation!
	std::vector<lexed_token>::iterator ltp = l_file.tokens.begin();
	// parse the first region, ie the filetype region
	while (ltp->lex != BEGIN_MARKER)
	{
		if (ltp->lex == SINGLE_COMMENT)
		{
			while (ltp->lex != NEW_LINE) ltp++;
		}
		else if (ltp->lex != NEW_LINE)
		{
			LOG_FATAL("Unidentified token. Exiting...");
			exit(1);
		}
		ltp++;
	}
	auto next_lt = std::next(ltp, 1);
	auto second_next_lt  = std::next(ltp, 2);
	if (next_lt->lex == REGION)
	{
		if (next_lt->raw_token != "filetype")
		{
			LOG_FATAL("First interpretable line does not specify filetype. Exiting...");
			exit(2);
		}
		else if (second_next_lt->raw_token != "run")
		{
			LOG_FATAL("'%s' does not indicate a session file. Exiting...", second_next_lt->raw_token.c_str());
			exit(3);
		}
		else
		{
			ltp += 4;
			parse_region(ltp, l_file, s_file, second_next_lt->raw_token);
		}
	}
	else
	{
		LOG_FATAL("Unidentified token after '%s'. Exiting...", ltp->raw_token.c_str());
		exit(4);
	}
}

void cp_parsed_sess_file(parsed_sess_file &from_s_file, parsed_sess_file &to_s_file)
{
	to_s_file.parsed_trial_info.trial_map = from_s_file.parsed_trial_info.trial_map;
	to_s_file.parsed_trial_info.block_map = from_s_file.parsed_trial_info.block_map;
	to_s_file.parsed_trial_info.session = from_s_file.parsed_trial_info.session;
	to_s_file.parsed_var_sections = from_s_file.parsed_var_sections;
}

/*
 * Implementation Notes:
 *     meant for parsing var sections in parsed_build files. loops until the end lex {END_MARKER, "end"} is
 *     found. Within the loop current variable attributes are collected and added to the (temporary) current section.
 *     Once the end marker is reached, the current section is added to the build file's parsed_var_sections map.
 */
static void parse_var_section(std::vector<lexed_token>::iterator &ltp,
                              lexed_file &l_file,
                              parsed_build_file &b_file,
                              std::string region_type)
{
	parsed_var_section curr_section = {};
	variable curr_var = {};
	while (ltp->lex != END_MARKER)
	{
		if (ltp->lex == TYPE_NAME)
		{
			auto next_lt = std::next(ltp, 1);
			auto second_next_lt = std::next(ltp, 2);
			if (next_lt->lex == VAR_IDENTIFIER
				&& second_next_lt->lex == VAR_VALUE)
			{
				curr_var.type_name  = ltp->raw_token;
				curr_var.identifier = next_lt->raw_token;
				curr_var.value      = second_next_lt->raw_token;

				curr_section.param_map[next_lt->raw_token] = curr_var;
				curr_var = {};
				ltp += 2;
			}
		}
		else if (ltp->lex == SINGLE_COMMENT)
		{
			while (ltp->lex != NEW_LINE) ltp++;
		}
		ltp++;
	}
	b_file.parsed_var_sections[region_type] = curr_section;
}

/*
 * Implementation Notes:
 *     a region is defined as a code block in a .sess file which begins with "begin" and ends with "end."
 *     This recursive function is run on regions in a parsed build file: it checks the region type (there
 *     are two recognized region types) and calls parse_var_section, otherwise loops over lexes until it
 *     encounters a valid sequence of begin marker, region, and region type.
 */
static void parse_region(std::vector<lexed_token>::iterator &ltp,
                         lexed_file &l_file,
                         parsed_build_file &b_file,
                         std::string region_type)
{
	if (region_type == "connectivity")
	{
		parse_var_section(ltp, l_file, b_file, region_type);
	}
	else if (region_type == "activity")
	{
		parse_var_section(ltp, l_file, b_file, region_type);
	}
	else
	{
		while (ltp->lex != END_MARKER)
		{
			auto next_lt = std::next(ltp, 1);
			auto second_next_lt  = std::next(ltp, 2);
			if (ltp->lex == BEGIN_MARKER
				&& next_lt->lex == REGION
				&& second_next_lt->lex == REGION_TYPE)
			{
				ltp += 4;
				parse_region(ltp, l_file, b_file, second_next_lt->raw_token);
			}
			else if (ltp->lex == SINGLE_COMMENT)
			{
				while (ltp->lex != NEW_LINE) ltp++;
			}
			ltp++;
		}
	}
}

/*
 * Implementation Notes:
 *     This function operates upon a lexed file, converting that file into a parsed file.
 *     While a recursive solution is more elegant, I brute-forced it for now by looping
 *     over lines and then over tokens. I create an empty parsed_section at the beginning
 *     of the algorithm, which will be filled within a section and then cleared in
 *     preparation for the next section. Basically I match each lexed token with the relevant lexemes.
 *     Notice that I do not have a case for VAR_VALUE: this is because when I reach VAR_IDENTIFIER
 *     I must use the identifier in the param map to assign the VAR_VALUE. So I progress the lexed token
 *     iterator by two at the end of this operatation, either reaching a comment or EOL.
 *     One final note is that I needed to keep the END_MARKERs in as they helped solve the problem
 *     of finding when we are finished with a section.
 */
void parse_lexed_build_file(lexed_file &l_file, parsed_build_file &b_file)
{
	std::vector<lexed_token>::iterator ltp = l_file.tokens.begin();
	// parse the first region, ie the filetype region
	while (ltp->lex != BEGIN_MARKER)
	{
		if (ltp->lex == SINGLE_COMMENT)
		{
			while (ltp->lex != NEW_LINE) ltp++;
		}
		else if (ltp->lex != NEW_LINE)
		{
			LOG_FATAL("Unidentified token. Exiting...");
			exit(1);
		}
		ltp++;
	}
	auto next_lt = std::next(ltp, 1);
	auto second_next_lt  = std::next(ltp, 2);
	if (next_lt->lex == REGION)
	{
		if (next_lt->raw_token != "filetype")
		{
			LOG_FATAL("First interpretable line does not specify filetype. Exiting...");
			exit(2);
		}
		else if (second_next_lt->raw_token != "build")
		{
			LOG_FATAL("'%s' does not indicate a build file. Exiting...", second_next_lt->raw_token.c_str());
			exit(3);
		}
		else
		{
			ltp += 4;
			parse_region(ltp, l_file, b_file, second_next_lt->raw_token);
		}
	}
	else
	{
		LOG_FATAL("Unidentified token after '%s'. Exiting...", ltp->raw_token.c_str());
		exit(4);
	}
}

/*
 * Implementation Notes:
 *     allocates memory for arrays in td. Note that it is expected that
 *     num_trials would be determined from the .sess file via calculate_num_trials.
 *
 */
void allocate_trials_data(trials_data &td, uint32_t num_trials)
{
	td.trial_names     = (std::string *)calloc(num_trials, sizeof(std::string));
	td.use_css         = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
	td.cs_onsets       = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
	td.cs_lens         = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
	td.cs_percents     = (float *)calloc(num_trials, sizeof(float));
	td.use_uss         = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
	td.us_onsets       = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
}

/*
 * Implementation Notes:
 *     the main recursive function for determining the ordering of trial names from 
 *     .sess file. Loops through every element of the input map and either adds
 *     the name of the current trial at the next empty trial name in td.trial_names,
 *     or checks whether the element is an already defined block. If the current element
 *     is within pt_section.block_map, it recurses and checks elements within the block
 *     for a previous trial definition.
 *
 */
static void initialize_trial_names_helper(trials_data &td, parsed_trial_section &pt_section,
	  std::vector<std::pair<std::string, std::string>> &in_vec)
{
	for (auto vec_pair : in_vec)
	{
		if (pt_section.trial_map.find(vec_pair.first) != pt_section.trial_map.end())
		{
			auto first_empty = std::find(td.trial_names, td.trial_names + td.num_trials, "");
			auto initial_empty = first_empty;
			if (first_empty != td.trial_names + td.num_trials)
			{
				uint32_t this_num_trials = std::stoi(vec_pair.second);
				while (first_empty != initial_empty + this_num_trials)
				{
					/* mad sus */
					*first_empty = vec_pair.first;
					first_empty++;
				}
			}
		}
		else
		{
			for (uint32_t i = 0; i < std::stoi(vec_pair.second); i++)
			{
				if (pt_section.block_map.find(vec_pair.first) != pt_section.block_map.end())
				{
					initialize_trial_names_helper(td, pt_section, pt_section.block_map[vec_pair.first]);
				}
			}
		}
	}
}

/*
 * Implementation Notes:
 *     main entry point for initializing td array elements from pt_section.
 *     The trial names need to be initialized first, as they are used in a look-up
 *     table to obtain the rest of the relevant data for that trial.
 *
 */
void initialize_trials_data(trials_data &td, parsed_trial_section &pt_section)
{
	initialize_trial_names_helper(td, pt_section, pt_section.session);
	for (uint32_t i = 0; i < td.num_trials; i++)
	{
		td.use_css[i]         = std::stoi(pt_section.trial_map[td.trial_names[i]]["use_cs"].value);
		td.cs_onsets[i]       = std::stoi(pt_section.trial_map[td.trial_names[i]]["cs_onset"].value);
		td.cs_lens[i]         = std::stoi(pt_section.trial_map[td.trial_names[i]]["cs_len"].value);
		td.cs_percents[i]     = std::stof(pt_section.trial_map[td.trial_names[i]]["cs_percent"].value);
		td.use_uss[i]         = std::stoi(pt_section.trial_map[td.trial_names[i]]["use_us"].value);
		td.us_onsets[i]       = std::stoi(pt_section.trial_map[td.trial_names[i]]["us_onset"].value);
	}
}

void delete_trials_data(trials_data &td)
{
	free(td.trial_names);
	free(td.use_css);
	free(td.cs_onsets);
	free(td.cs_lens);
	free(td.cs_percents);
	free(td.use_uss);
	free(td.us_onsets);
}

/*
 * Implementation Notes:
 *     the main recursive algorithm for determining the number of trials specified from .sess
 *     file and encoded withing pt_section. Computes a sum over the current iteration,
 *     and for each level of recursion, finishes by multiplying the current sum by the previously
 *     obtained number of trials. Of course, only recurses if vec_pair.first is a valid definition
 *     of a block.
 */
void calculate_num_trials_helper(parsed_trial_section &pt_section,
		std::vector<std::pair<std::string, std::string>> &in_vec, uint32_t &temp_num_trials)
{
	uint32_t temp_sum = 0;
	for (auto vec_pair : in_vec)
	{
		uint32_t temp_count = std::stoi(vec_pair.second);
		if (pt_section.block_map.find(vec_pair.first ) != pt_section.block_map.end())
		{
			calculate_num_trials_helper(pt_section, pt_section.block_map[vec_pair.first], temp_count);
		}
		temp_sum += temp_count;
	}
	temp_num_trials *= temp_sum;
}

/*
 * Implementation Notes:
 *     the entry point for recursively calculating the number of trials from pt_section.
 *     Because sums computed at each recursive layer are multiplied by and assigned to the
 *     previous temp_num_trials, we need to "seed" this process with 1.
 *
 */
uint32_t calculate_num_trials(parsed_trial_section &pt_section)
{
	uint32_t temp_num_trials = 1;
	calculate_num_trials_helper(pt_section, pt_section.session, temp_num_trials);
	return temp_num_trials;
}

/*
 * Implementation Notes:
 *     Computes the number of trials and allocates/initializes td given the computed number of trials.
 *
 */
void translate_parsed_trials(parsed_sess_file &s_file, trials_data &td)
{
	td.num_trials = calculate_num_trials(s_file.parsed_trial_info);
	// no error checking on returned value, uh oh!
	allocate_trials_data(td, td.num_trials);
	initialize_trials_data(td, s_file.parsed_trial_info);
}

/*
 * Implementation Notes for to_str functions:
 *     all *to_str functions use std::stringstreams to create a str representation of the
 *     given data structure. For the std::ostream insertion operator overloads, the *to_str
 *     functions are called to return the string representation which is then inserted into
 *     the given argument std::ostream.
 *
 */
std::string tokenized_file_to_str(tokenized_file &t_file)
{
	std::stringstream tokenized_file_buf;
	tokenized_file_buf << "[\n";
	for (auto line : t_file.tokens)
	{
		for (auto token : line)
		{
			tokenized_file_buf << "['" << token << "'],\n";
		}
	}
	tokenized_file_buf << "]\n";
	return tokenized_file_buf.str();
}

std::string lexed_file_to_str(lexed_file &l_file)
{
	std::stringstream lexed_file_buf;
	lexed_file_buf << "[\n";
	for (auto token : l_file.tokens)
	{
		lexed_file_buf << "['";
		lexed_file_buf << lex_string_look_up[token.lex] << "', '";
		lexed_file_buf << token.raw_token << "'],\n";
	}
	lexed_file_buf << "]\n";
	return lexed_file_buf.str();
}

std::string parsed_build_file_to_str(parsed_build_file &b_file)
{
	std::stringstream build_file_buf;
	build_file_buf << "[\n";
	for (auto var_sec : b_file.parsed_var_sections)
	{
		build_file_buf << "{\n";
		for (auto pair : var_sec.second.param_map)
		{
			build_file_buf << "['" << pair.first << "': {'";
			build_file_buf << pair.second.type_name << "', '";
			build_file_buf << pair.second.identifier << "', '";
			build_file_buf << pair.second.value << "'}]\n";
		}
		build_file_buf << "}\n";
	}
	return build_file_buf.str();
}

std::string parsed_sess_file_to_str(parsed_sess_file &s_file)
{
	std::stringstream sess_file_buf;
	sess_file_buf << "[\n";
	for (auto var_sec : s_file.parsed_var_sections)
	{
		sess_file_buf << "{\n";
		for (auto pair : var_sec.second.param_map)
		{
			sess_file_buf << "['" << pair.first << "': {'";
			sess_file_buf << pair.second.type_name << "', '";
			sess_file_buf << pair.second.identifier << "', '";
			sess_file_buf << pair.second.value << "'}]\n";
		}
		sess_file_buf << "}\n";
	}

	sess_file_buf << "{\n";
	for (auto pair : s_file.parsed_trial_info.trial_map)
	{
		sess_file_buf << "{'" << pair.first << "': {'";
		for (auto vars : pair.second)
		{
			sess_file_buf << "{'" << vars.first << "': {'";
			sess_file_buf << vars.second.type_name << "', '";
			sess_file_buf << vars.second.identifier << "', '";
			sess_file_buf << vars.second.value << "'}}\n";
		}
		sess_file_buf << "}\n";
	}
	sess_file_buf << "}\n";
	
	sess_file_buf << "{\n";
	for (auto block : s_file.parsed_trial_info.block_map)
	{
		sess_file_buf << "['" << block.first << "' : {";
		for (auto t_pair : block.second)
		{
			sess_file_buf << "{'" << t_pair.first << "', '";
			sess_file_buf << t_pair.second << "'}\n";
		}
		sess_file_buf << "}]\n";
	}
	sess_file_buf << "}\n";

	sess_file_buf << "{\n";
	for (auto bt_pair : s_file.parsed_trial_info.session)
	{
		sess_file_buf << "{'" << bt_pair.first << "', '";
		sess_file_buf << bt_pair.second << "'}\n";
	}
	sess_file_buf << "}\n";

	return sess_file_buf.str();
}

std::ostream &operator<<(std::ostream &os, tokenized_file &t_file)
{
	return os << tokenized_file_to_str(t_file);
}

std::ostream &operator<<(std::ostream &os, lexed_file &l_file)
{
	return os << lexed_file_to_str(l_file);
}

std::ostream &operator <<(std::ostream &os, parsed_build_file &b_file)
{
	return os << parsed_build_file_to_str(b_file);
}

std::ostream &operator <<(std::ostream &os, parsed_sess_file &s_file)
{
	return os << parsed_sess_file_to_str(s_file);

}

