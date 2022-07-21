/*
 * File: build_file.cpp
 * Author: Sean Gallogly
 * Created on: 07/21/2022
 *
 * Description:
 *     This file implements the function prototypes in fileIO/build_file.h
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include "fileIO/build_file.h"

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
void tokenize_build_file(std::string b_file, tokenized_file &t_file)
{
	std::ifstream b_file_buf(b_file.c_str());
	if (!b_file_buf.is_open())
	{
		std::cerr << "[ERROR]: Could not open build file. Exiting" << std::endl;
		exit(3);
	}

	std::string line = "";
	while (getline(b_file_buf, line))
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
}

/*
 * Implementation Notes;
 *     This function operates upon a tokenized file by going line-by-line, token-by-token
 *     and matching the token with its lexical-definition. The only tricky tokens are the
 *     variable identifiers and the variable values, as these are naturally variable from
 *     one variable to the next. First I search for the token as-is in the token definitions
 *     look-up table. This ensures that I'm not matching a constant lexical token with 
 *     a variable one. Then I attempt to match, in sequence, the raw_token with the regex
 *     pattern string associated with variable identifiers and variable values. Finally,
 *     I add the relevant data into each lexed_token, add that lexed_token to the line,
 *     and then the line to the entire file.
 *
 */

void lex_tokenized_build_file(tokenized_file &t_file, lexed_file &l_file)
{
	std::regex var_id_regex(var_id_regex_str);
	std::regex var_val_regex(var_val_regex_str);
	for (auto line : t_file.tokens)
	{
		std::vector<lexed_token> line_tokens;
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
			line_tokens.push_back(l_token);
		}
		l_file.tokens.push_back(line_tokens);
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
 *     One final note is that I needed to keep in the END_MARKERs as they helped solve the problem
 *     of finding when we are finished with a section.
 *
 */
void parse_lexed_build_file(lexed_file &l_file, parsed_file &p_file)
{
	parsed_section curr_section = {};
	auto line = l_file.tokens.begin();
	while (line != l_file.tokens.end())
	{
		auto lexed_token = line->begin();
		while (lexed_token != line->end())
		{
			switch (lexed_token->lex)
			{
				case BEGIN_MARKER:
					lexed_token++;
					break;
				case REGION:
					if (lexed_token->raw_token == "filetype")
					{
						lexed_token++;
						p_file.file_type_label = lexed_token->raw_token;
					}
					else if (lexed_token->raw_token == "section")
					{
						lexed_token++;
						curr_section.section_label = lexed_token->raw_token;
					}
					lexed_token++;
					break;
				case VAR_IDENTIFIER:
					// TODO: check whether next token is indeed a VAR_VALUE
					curr_section.param_map[lexed_token->raw_token] = (lexed_token + 1)->raw_token;
					lexed_token += 2;
					break;
				case END_MARKER:
					if (curr_section.section_label != "")
					{
						p_file.parsed_sections.push_back(curr_section);
						curr_section.section_label = "";
						curr_section.param_map.clear();
					}
					lexed_token++;
					break;
				case SINGLE_COMMENT:
					lexed_token = line->end();
					break;
			}
		}
		line++;
	}
}

/*
 * Implementation Notes:
 *     Loops go brrrrrrr
 *
 */
void print_tokenized_build_file(tokenized_file &t_file)
{
	for (auto line : t_file.tokens)
	{
		for (auto token : line)
		{
			std::cout << "['" << token << "'], ";
		}
		std::cout << std::endl;
	}
}

void print_lexed_build_file(lexed_file &l_file)
{
	for (auto line : l_file.tokens)
	{
		for (auto token : line)
		{
				std::cout << "['";
				std::cout << lex_string_look_up[token.lex] << "', '";
				std::cout << token.raw_token << "'], ";
		}
		std::cout << std::endl;
	}
}

void print_parsed_build_file(parsed_file &p_file)
{
	std::cout << "[ 'filetype', " << "'" << p_file.file_type_label << "'" << "]" << std::endl;
	for (auto p_section : p_file.parsed_sections)
	{
		std::cout << "[ 'section', " << "'" << p_section.section_label << "'" << "]" << std::endl;
		for (auto iter = p_section.param_map.begin(); iter != p_section.param_map.end(); iter++)
		{
			std::cout << "[ '" << iter->first << "', '" << iter->second << "' ]" << std::endl;
		}
	}
}


