/*
 * File: file_parse.h
 * Author: Sean Gallogly
 * Created on: 10/01/2022
 *
 * Description:
 *     This is the interface file for reading in a build file. It includes the
 *     structures, enums, and tables necessary to tokenize, lex, and parse a
 * build file. This file is more of a workflow and less of a class interface
 * file in that the end result of this process is to obtain a structure of type
 *     parsed_file that contains all of the parameters that we need to build a
 *     simulation from scratch, akin to "generating" a new rabbit on the fly.
 */
#ifndef FILE_PARSE_H_
#define FILE_PARSE_H_

#include "json.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <utility> /* for std::pair */
#include <vector>

using json = nlohmann::json;

typedef struct {
  uint32_t num_trials;
  std::string *trial_names;
  uint32_t *use_css;
  uint32_t *cs_onsets;
  uint32_t *cs_lens;
  float *cs_percents;
  uint32_t *use_uss;
  uint32_t *us_onsets;
} trials_data;

/*
 * Description:
 *     allocates memory for the arrays within reference to trials_data td.
 *     Caller is responsible for the memory that is allocated in this function.
 *     (blah blah blah, fancy talk for if you call this function, you will want
 *      to call delete_trials_data at a later point to prevent memory leak)
 *
 */
void allocate_trials_data(trials_data &td, std::string s_file_name);

/*
 * Description:
 *     translates trials section information in parsed sess file into td.
 *     td follows the structure of arrays (SoA) paradigm for efficiency in
 *     member element access. Data in td is used later in Control::runSession
 *
 */
void translate_trials(std::string s_file_name, trials_data &td);

/*
 * Description:
 *     deallocates memory for the data members of td. NOTE: allocate_trials_data
 *     must be called before this function is called.
 *
 */
void delete_trials_data(trials_data &td);

#endif /* FILE_PARSE_H_ */
