/*
 * File: build_file.h
 * Author: Sean Gallogly
 * Created on: 07/21/2022
 * 
 * Description:
 *     This is the interface file for reading in a build file. It includes the 
 *     structures, enums, and tables necessary to tokenize, lex, and parse a build
 *     file. This file is more of a workflow and less of a class interface file in
 *     that the end result of this process is to obtain a structure of type
 *     parsed_file that contains all of the parameters that we need to build a 
 *     simulation from scratch, akin to "generating" a new rabbit on the fly.
 */
#ifndef BUILD_FILE_H_
#define BUILD_FILE_H_

#include <string>
#include <vector>
#include <map>

/* ================================ TYPES AND CONSTANTS ================================= */

// lexemes of the build file ie the fundamental meanings behind each token
enum lexeme {
	BEGIN_MARKER, END_MARKER,
	REGION,
	REGION_TYPE,
	TYPE_NAME,
	VAR_IDENTIFIER, VAR_VALUE,
	SINGLE_COMMENT,
	DOUBLE_COMMENT_BEGIN, DOUBLE_COMMENT_END
};

typedef enum lexeme lexeme;

// a token that is identified by its lexeme
struct lexed_token
{
	lexeme lex;
	std::string raw_token;
};

typedef struct lexed_token lexed_token;

struct variable
{
	std::string type_name;
	std::string identifier;
	std::string value;
};

typedef struct variable variable;

/* 
 * a section within the build file that includes the label of the section
 * and the dictionary of parameters in that section
 *
 */
struct parsed_section
{
	std::string section_label;
	std::map<std::string, variable> param_map;
};

typedef struct parsed_section parsed_section;

// represents the entire file contents, broken up line-by-line, token-by-token
struct tokenized_file
{
   std::vector<std::vector<std::string>> tokens;
};

typedef struct tokenized_file tokenized_file;

/* 
 * represents the entire file contents broken up line-by-line, token-by-token,
 * with each token being identified by its lexeme
 *
 */
struct lexed_file
{
   std::vector<std::vector<lexed_token>> tokens;
};

typedef struct lexed_file lexed_file;

/*
 * represents everything that we wanted to get out of the file. Contains the file
 * type as well as all of the parsed sections.
 *
 */
struct parsed_build_file
{
	std::string file_type_label;
	std::map<std::string, parsed_section> parsed_sections;
};

typedef parsed_build_file parsed_file;

/* ============================== FUNCTION DECLARATIONS ============================== */

/*
 * Description:
 *     takes in the string b_file representing the build file's name and constructs
 *     a tokenized file from it. See the above definition of tokenized_file.
 *
 */
void tokenize_build_file(std::string b_file, tokenized_file &t_file);

/*
 * Description:
 *     takes a tokenized file reference t_file and lexes it, i.e. assigns meaningful
 *     tags to the tokens. It does this by populating the lexed_file reference l_file.
 *
 */
void lex_tokenized_build_file(tokenized_file &t_file, lexed_file &l_file);

/*
 * Description:
 *     takes a lexed file reference l_file and parses it, i.e. takes each lexeme
 *     and adds it to the correct entry in p_file.parsed_sections. See the above
 *     definition of parsed_build_file and parsed_section for more information.
 *
 */
void parse_lexed_build_file(lexed_file &l_file, parsed_build_file &p_file);

void print_tokenized_build_file(tokenized_file &t_file);

void print_lexed_build_file(lexed_file &l_file);

void print_parsed_build_file(parsed_build_file &p_file);

#endif /* BUILD_FILE_H_ */

