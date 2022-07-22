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
struct parsed_file
{
	std::string file_type_label;
	std::map<std::string, parsed_section> parsed_sections;
};

typedef parsed_file parsed_file;

/* ============================== FORWARD FUNCTION DECLARATIONS ============================== */

void tokenize_build_file(std::string b_file, tokenized_file &t_file);

void lex_tokenized_build_file(tokenized_file &t_file, lexed_file &l_file);

void parse_lexed_build_file(lexed_file &l_file, parsed_file &p_file);

void print_tokenized_build_file(tokenized_file &t_file);

void print_lexed_build_file(lexed_file &l_file);

void print_parsed_build_file(parsed_file &p_file);


#define BUILD_FILE_H_
#endif /* BUILD_FILE_H_ */
