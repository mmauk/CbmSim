/*
 * File: file_parse.h
 * Author: Sean Gallogly
 * Created on: 10/01/2022
 * 
 * Description:
 *     This is the interface file for reading in a build file. It includes the 
 *     structures, enums, and tables necessary to tokenize, lex, and parse a build
 *     file. This file is more of a workflow and less of a class interface file in
 *     that the end result of this process is to obtain a structure of type
 *     parsed_file that contains all of the parameters that we need to build a 
 *     simulation from scratch, akin to "generating" a new rabbit on the fly.
 */
#ifndef FILE_PARSE_H_
#define FILE_PARSE_H_

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "pstdint.h"

// lexemes of the build file ie the fundamental meanings behind each token
typedef enum 
{
	NONE,
	BEGIN_MARKER, END_MARKER,
	REGION,
	REGION_TYPE,
	TYPE_NAME,
	VAR_IDENTIFIER, VAR_VALUE,
	DEF, DEF_TYPE, DEF_TYPE_LABEL, 
	NEW_LINE,
	SINGLE_COMMENT,
	DOUBLE_COMMENT_BEGIN, DOUBLE_COMMENT_END
} lexeme;

// a token that is identified by its lexeme
typedef struct
{
	lexeme lex;
	std::string raw_token;
} lexed_token;

typedef struct
{
	std::string type_name;
	std::string identifier;
	std::string value;
} variable;

typedef struct
{
	std::string first;
	std::string second;
} pair;

typedef struct 
{
	ct_uint32_t num_trials;
	std::string *trial_names;
	ct_uint32_t *use_css;
	ct_uint32_t *cs_onsets;
	ct_uint32_t *cs_lens;
	float       *cs_percents;
	ct_uint32_t *use_uss;
	ct_uint32_t *us_onsets;
} trials_data;

typedef struct
{
	std::unordered_map<std::string, variable> param_map;
} trial_2;

typedef struct
{
	std::vector<pair> trials;
} block;

typedef struct
{
	std::vector<pair> blocks;
} session;

typedef struct
{
	std::vector<pair> sessions;
} experiment_2;

/* 
 * a section within the build file that includes the label of the section
 * and the dictionary of parameters in that section
 *
 */
typedef struct
{
	std::unordered_map<std::string, variable> param_map;
} parsed_var_section;

//typedef struct
//{
//	std::unordered_map<std::string, trial_2> trial_map;
//	std::unordered_map<std::string, block> block_map;
//	std::unordered_map<std::string, session> session_map;
//	experiment_2 expt_info;
//} parsed_trial_section;

typedef struct
{
	std::unordered_map<std::string, std::unordered_map<std::string, variable>> trial_map;
	std::unordered_map<std::string, std::vector<pair>> block_map;
	std::unordered_map<std::string, std::vector<pair>> session_map;
	std::vector<pair> experiment; // <-- pairs of session identifier and number of sessions
} parsed_trial_section;

// represents the entire file contents, broken up line-by-line, token-by-token
typedef struct
{
   std::vector<std::vector<std::string>> tokens;
} tokenized_file;

/* 
 * represents the entire file contents broken up line-by-line, token-by-token,
 * with each token being identified by its lexeme
 *
 */
typedef struct
{
   std::vector<lexed_token> tokens;
} lexed_file;

typedef struct
{
	std::map<std::string, parsed_var_section> parsed_var_sections;
} parsed_build_file;

typedef struct
{
	parsed_trial_section parsed_trial_info;
	std::map<std::string, parsed_var_section> parsed_var_sections;
} parsed_expt_file;

/*
 * Description:
 *     takes in the string b_file representing the build file's name and constructs
 *     a tokenized file from it. See the above definition of tokenized_file.
 *
 */
void tokenize_file(std::string in_file, tokenized_file &t_file);

/*
 * Description:
 *     takes a tokenized file reference t_file and lexes it, i.e. assigns meaningful
 *     tags to the tokens. It does this by populating the lexed_file reference l_file.
 *
 */
void lex_tokenized_file(tokenized_file &t_file, lexed_file &l_file);

/*
 * Description:
 *     takes a lexed file reference l_file and parses it, i.e. takes each lexeme
 *     and adds it to the correct entry in either parsed_expt_file.parsed_trial_info or
 *     parsed_expt_file.parsed_var_sections
 *
 */
void parse_lexed_expt_file(lexed_file &l_file, parsed_expt_file &e_file);

/*
 * Description:
 *     takes a lexed file reference l_file and parses it, i.e. takes each lexeme
 *     and adds it to the correct entry in p_file.parsed_var_sections. See the above
 *     definition of parsed_build_file and parsed_section for more information.
 *
 */
void parse_lexed_build_file(lexed_file &l_file, parsed_build_file &p_file);

void translate_parsed_trials(parsed_expt_file &pe_file, trials_data &td);

void print_tokenized_file(tokenized_file &t_file);

void print_lexed_file(lexed_file &l_file);

void print_parsed_expt_file(parsed_expt_file &e_file);

void print_parsed_build_file(parsed_build_file &b_file);

#endif /* FILE_PARSE_H_ */

