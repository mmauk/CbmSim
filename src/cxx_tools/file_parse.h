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
#include <utility> /* for std::pair */
#include <vector>
#include <map>
#include <cstdint>

// lexemes of the input files ie the fundamental meanings behind each token
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
	uint32_t num_trials;
	std::string *trial_names;
	uint32_t *use_css;
	uint32_t *cs_onsets;
	uint32_t *cs_lens;
	float    *cs_percents;
	uint32_t *use_uss;
	uint32_t *us_onsets;
} trials_data;

/* 
 * a section within an input file that includes the label of the section
 * and the dictionary of parameters in that section
 *
 */
typedef struct
{
	std::map<std::string, variable> param_map;
} parsed_var_section;

typedef struct
{
	std::map<std::string, std::map<std::string, variable>> trial_map;
	std::map<std::string, std::vector<std::pair<std::string, std::string>>> block_map;
	std::vector<std::pair<std::string, std::string>> session;  // <-- pairs of block identifier and number of blocks 
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
} parsed_sess_file;

/*
 * Description:
 *     takes in the string in_file representing the input file's name and constructs
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
 *     and adds it to the correct entry in either parsed_sess_file.parsed_trial_info or
 *     parsed_sess_file.parsed_var_sections
 *
 */
void parse_lexed_sess_file(lexed_file &l_file, parsed_sess_file &s_file);

/*
 * Description:
 *     takes in parsed_session_file 'from_s_file' and copies its data into 'to_s_file'
 */
void cp_parsed_sess_file(parsed_sess_file &from_s_file, parsed_sess_file &to_s_file);

/*
 * Description:
 *     takes a lexed file reference l_file and parses it, i.e. takes each lexeme
 *     and adds it to the correct entry in p_file.parsed_var_sections. See the above
 *     definition of parsed_build_file and parsed_section for more information.
 *
 */
void parse_lexed_build_file(lexed_file &l_file, parsed_build_file &p_file);

/*
 * Description:
 *     allocates memory for the arrays within reference to trials_data td.
 *     Caller is responsible for the memory that is allocated in this function.
 *     (blah blah blah, fancy talk for if you call this function, you will want
 *      to call delete_trials_data at a later point to prevent memory leak)
 *
 */
void allocate_trials_data(trials_data &td, uint32_t num_trials);

/*
 * Description:
 *     initializes elements of arrays in td according to parsed trial definitions
 *     in pt_section. NOTE: "allocate_trials_data" must be run before this function
 *     is called.
 *
 */
void initialize_trials_data(trials_data &td, parsed_trial_section &pt_section);

/*
 * Description:
 *     translates trials section information in parsed sess file into td.
 *     td follows the structure of arrays (SoA) paradigm for efficiency in
 *     member element access. Data in td is used later in Control::runSession
 *
 */
void translate_parsed_trials(parsed_sess_file &s_file, trials_data &td);

/*
 * Description:
 *     deallocates memory for the data members of td. NOTE: allocate_trials_data
 *     must be called before this function is called.
 *
 */
void delete_trials_data(trials_data &td);

/*
 * Description of following 4 functions:
 *      following are used to print the given type to the console like
 *      you would use the stream insertion operator for atomic types and stl
 *      types for which an operator overload exists.
 * 
 * Example Usage:
 *
 *     std::cout << t_file << std::endl;
 */

std::ostream &operator<<(std::ostream &os, tokenized_file &t_file);

std::ostream &operator<<(std::ostream &os, lexed_file &l_file);

std::ostream &operator <<(std::ostream &os, parsed_build_file &b_file);

std::ostream &operator <<(std::ostream &os, parsed_sess_file &s_file);

#endif /* FILE_PARSE_H_ */

