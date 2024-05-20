/*
 * File: file_parse.cpp
 * Author: Sean Gallogly
 * Created on: 10/02/2022
 *
 * Description:
 *     This file implements the function prototypes in file_parse.h
 */

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

#include "file_parse.h"
#include "logger.h"

/*
 * Implementation Notes:
 *     meant to be called within parse_trial_section. Loops until the end lex is
 * found. Is used for each of the types of defintions which are possible: 1)
 * trial definitions, 2) block definitions, and 3) session definitions. To serve
 * these 3 purposes we need to initialize a lot of temporary data structures,
 * which may or may not be used depending on the current definition type being
 *     parsed.
 *
 *     Collects the required information for the given definition type. In order
 * to be compatible with "statements" within the definition consisting of either
 * single trial or block identifiers, we add a "shadow token" within the l_file,
 * which conceptually adds in the value of "1" for that trial or block
 * identifier.
 */
// static void parse_def(std::vector<lexed_token>::iterator &ltp,
//                       lexed_file &l_file, parsed_sess_file &s_file,
//                       std::string def_type, std::string def_label) {
//   std::pair<std::string, std::string> curr_pair = {};
//   std::map<std::string, std::string> curr_trial = {};
//   std::vector<std::pair<std::string, std::string>> curr_block = {};
//   lexeme prev_lex = NONE;
//   while (ltp->lex != END_MARKER) {
//     switch (ltp->lex) {
//     case VAR_IDENTIFIER:
//       if (def_type == "trial") {
//         curr_pair.first = ltp->raw_token;
//       } else {
//         /* disgusting hack: inserts a 'shadow token'
//          * so that if the prev identifier didnt include
//          * a value immediately proceeding it, a value
//          * is inserted into the vector to act as if there
//          * was one.
//          */
//         if (curr_pair.first != "") {
//           lexed_token shadow_token = {VAR_VALUE, "1"};
//           ltp = l_file.tokens.insert(ltp, shadow_token);
//           ltp--;
//         } else
//           curr_pair.first = ltp->raw_token;
//       }
//       break;
//     case VAR_VALUE:
//       if (prev_lex != VAR_IDENTIFIER && prev_lex != NEW_LINE) {
//         // TODO: report error
//       } else {
//         curr_pair.second = ltp->raw_token;
//         if (def_type == "trial") {
//           curr_trial[curr_pair.first] = curr_pair.second;
//         } else if (def_type == "block") {
//           curr_block.push_back(curr_pair);
//         } else if (def_type == "session") {
//           // TODO: should check if curr_pair.first is either a block or a
//           trial
//           // identifier (at a later stage of processing)
//           s_file.parsed_trial_info.session.push_back(curr_pair);
//         }
//         curr_pair = {};
//       }
//       break;
//     case SINGLE_COMMENT:
//       while (ltp->lex != NEW_LINE)
//         ltp++;
//       break;
//     }
//     prev_lex = ltp->lex;
//     ltp++;
//     // ensures a single identifier in a def block is added to the right
//     // structure
//     if (ltp->lex == END_MARKER && def_type != "trial" &&
//         curr_pair.first != "" && curr_pair.second == "") {
//       curr_pair.second = "1";
//       if (def_type == "block")
//         curr_block.push_back(curr_pair);
//       else if (def_type == "session")
//         s_file.parsed_trial_info.session.push_back(curr_pair);
//     }
//   }
//   if (def_type == "trial")
//     s_file.parsed_trial_info.trial_map[def_label] = curr_trial;
//   else if (def_type == "block")
//     s_file.parsed_trial_info.block_map[def_label] = curr_block;
// }

/*
 * Implementation Notes:
 *     meant for parsing var sections in parsed_session files. loops until the
 * end lex {END_MARKER, "end"} is found. Within the loop current variable
 * attributes are collected and added to the (temporary) current section. Once
 * the end marker is reached, the current section is added to the session file's
 * parsed_var_sections map.
 */
// static void parse_var_section(std::vector<lexed_token>::iterator &ltp,
//                               lexed_file &l_file, parsed_sess_file &s_file,
//                               std::string region_type) {
//   parsed_var_section curr_section = {};
//   while (ltp->lex != END_MARKER) {
//     auto next_ltp = std::next(ltp, 1);
//     if (ltp->lex == VAR_IDENTIFIER && next_ltp->lex == VAR_VALUE) {
//       curr_section.param_map[ltp->raw_token] = next_ltp->raw_token;
//       ltp += 1;
//     } else if (ltp->lex == SINGLE_COMMENT) {
//       while (ltp->lex != NEW_LINE)
//         ltp++;
//     }
//     ltp++;
//   }
//   s_file.parsed_var_sections[region_type] = curr_section;
// }

/*
 * Implementation Notes:
 *     loops over lexes until END_MARKER is found. checks whether a 3-lex
 * sequence definition header is found, and if so, calls parse_def. Ignores
 * single comments by passing over the lexes until the new line lex is found.
 */
// static void parse_trial_section(std::vector<lexed_token>::iterator &ltp,
//                                 lexed_file &l_file, parsed_sess_file &s_file)
//                                 {
//   while (ltp->lex != END_MARKER) // parse this section
//   {
//     if (ltp->lex == DEF) {
//       auto next_ltp = std::next(ltp, 1);
//       auto second_next_ltp = std::next(ltp, 2);
//       if (next_ltp->lex == DEF_TYPE && second_next_ltp->lex ==
//       VAR_IDENTIFIER) {
//         ltp += 4;
//         parse_def(ltp, l_file, s_file, next_ltp->raw_token,
//                   second_next_ltp->raw_token);
//       } else {
//       } // TODO: report error
//     } else if (ltp->lex == SINGLE_COMMENT) {
//       while (ltp->lex != NEW_LINE)
//         ltp++;
//     }
//     ltp++;
//   }
// }

/*
 * Implementation Notes:
 *     a region is defined as a code block in a .sess file which begins with
 * "begin" and ends with "end." This recursive function is run on regions in a
 * parsed sess file: it checks the region type (there are two recognized region
 * types) and calls parse_var_section or parse_trial_section, otherwise loops
 *     over lexes until it encounters a valid sequence of begin marker, region,
 * and region type.
 *
 */
// static void parse_region(std::vector<lexed_token>::iterator &ltp,
//                          lexed_file &l_file, parsed_sess_file &s_file,
//                          std::string region_type) {
//   if (region_type == "mf_input" || region_type == "activity" ||
//       region_type == "trial_spec") {
//     parse_var_section(ltp, l_file, s_file, region_type);
//   } else if (region_type == "trial_def") {
//     parse_trial_section(ltp, l_file, s_file);
//   } else {
//     while (ltp->lex != END_MARKER) {
//       auto next_lt = std::next(ltp, 1);
//       auto second_next_lt = std::next(ltp, 2);
//       if (ltp->lex == BEGIN_MARKER && next_lt->lex == REGION &&
//           second_next_lt->lex == REGION_TYPE) {
//         ltp += 4;
//         parse_region(ltp, l_file, s_file, second_next_lt->raw_token);
//       } else if (ltp->lex == SINGLE_COMMENT) {
//         while (ltp->lex != NEW_LINE)
//           ltp++;
//       }
//       ltp++;
//     }
//   }
// }

/*
 * Implementation Notes:
 *     the main function for parsing lexes within l_file, assuming l_file
 * represents a lexed session file. First loops over lexes until it encounters
 * the following sequence of tokens: "begin filetype run" TODO: change filetype
 * to session. Then loops over lexes until trial section or variable section
 * 3-lex sequence headers are found. If either is found, enters the appropriate
 *     parsing subroutine.
 *
 */
// void parse_lexed_sess_file(lexed_file &l_file, parsed_sess_file &s_file) {
//   // NOTE: ltp stands for [l]exed [t]oken [p]ointer, not long-term
//   potentiation! std::vector<lexed_token>::iterator ltp =
//   l_file.tokens.begin();
//   // parse the first region, ie the filetype region
//   while (ltp->lex != BEGIN_MARKER) {
//     if (ltp->lex == SINGLE_COMMENT) {
//       while (ltp->lex != NEW_LINE)
//         ltp++;
//     } else if (ltp->lex != NEW_LINE) {
//       LOG_FATAL("Unidentified token. Exiting...");
//       exit(1);
//     }
//     ltp++;
//   }
//   auto next_lt = std::next(ltp, 1);
//   auto second_next_lt = std::next(ltp, 2);
//   if (next_lt->lex == REGION) {
//     if (next_lt->raw_token != "filetype") {
//       LOG_FATAL(
//           "First interpretable line does not specify filetype. Exiting...");
//       exit(2);
//     } else if (second_next_lt->raw_token != "run") {
//       LOG_FATAL("'%s' does not indicate a session file. Exiting...",
//                 second_next_lt->raw_token.c_str());
//       exit(3);
//     } else {
//       ltp += 4;
//       parse_region(ltp, l_file, s_file, second_next_lt->raw_token);
//     }
//   } else {
//     LOG_FATAL("Unidentified token after '%s'. Exiting...",
//               ltp->raw_token.c_str());
//     exit(4);
//   }
// }

void cp_parsed_sess_file(parsed_sess_file &from_s_file,
                         parsed_sess_file &to_s_file) {
  to_s_file.parsed_trial_info.trial_map =
      from_s_file.parsed_trial_info.trial_map;
  to_s_file.parsed_trial_info.block_map =
      from_s_file.parsed_trial_info.block_map;
  to_s_file.parsed_trial_info.session = from_s_file.parsed_trial_info.session;
  to_s_file.parsed_var_sections = from_s_file.parsed_var_sections;
}

/*
 * Implementation Notes:
 *     allocates memory for arrays in td. Note that it is expected that
 *     num_trials would be determined from the .sess file via
 * calculate_num_trials.
 *
 */
void allocate_trials_data(trials_data &td, uint32_t num_trials) {
  td.trial_names = (std::string *)calloc(num_trials, sizeof(std::string));
  td.use_css = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
  td.cs_onsets = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
  td.cs_lens = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
  td.cs_percents = (float *)calloc(num_trials, sizeof(float));
  td.use_uss = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
  td.us_onsets = (uint32_t *)calloc(num_trials, sizeof(uint32_t));
}

/*
 * Implementation Notes:
 *     the main recursive function for determining the ordering of trial names
 * from .sess file. Loops through every element of the input map and either adds
 *     the name of the current trial at the next empty trial name in
 * td.trial_names, or checks whether the element is an already defined block. If
 * the current element is within pt_section.block_map, it recurses and checks
 * elements within the block for a previous trial definition.
 *
 */
static void initialize_trial_names_helper(
    trials_data &td, parsed_trial_section &pt_section,
    std::vector<std::pair<std::string, std::string>> &in_vec) {
  for (auto vec_pair : in_vec) {
    if (pt_section.trial_map.find(vec_pair.first) !=
        pt_section.trial_map.end()) {
      auto first_empty =
          std::find(td.trial_names, td.trial_names + td.num_trials, "");
      auto initial_empty = first_empty;
      if (first_empty != td.trial_names + td.num_trials) {
        uint32_t this_num_trials = std::stoi(vec_pair.second);
        while (first_empty != initial_empty + this_num_trials) {
          /* mad sus */
          *first_empty = vec_pair.first;
          first_empty++;
        }
      }
    } else {
      for (uint32_t i = 0; i < std::stoi(vec_pair.second); i++) {
        if (pt_section.block_map.find(vec_pair.first) !=
            pt_section.block_map.end()) {
          initialize_trial_names_helper(td, pt_section,
                                        pt_section.block_map[vec_pair.first]);
        }
      }
    }
  }
}

/*
 * Implementation Notes:
 *     main entry point for initializing td array elements from pt_section.
 *     The trial names need to be initialized first, as they are used in a
 * look-up table to obtain the rest of the relevant data for that trial.
 *
 */
void initialize_trials_data(trials_data &td, parsed_trial_section &pt_section) {
  initialize_trial_names_helper(td, pt_section, pt_section.session);
  for (uint32_t i = 0; i < td.num_trials; i++) {
    td.use_css[i] =
        std::stoi(pt_section.trial_map[td.trial_names[i]]["use_cs"]);
    td.cs_onsets[i] =
        std::stoi(pt_section.trial_map[td.trial_names[i]]["cs_onset"]);
    td.cs_lens[i] =
        std::stoi(pt_section.trial_map[td.trial_names[i]]["cs_len"]);
    td.cs_percents[i] =
        std::stof(pt_section.trial_map[td.trial_names[i]]["cs_percent"]);
    td.use_uss[i] =
        std::stoi(pt_section.trial_map[td.trial_names[i]]["use_us"]);
    td.us_onsets[i] =
        std::stoi(pt_section.trial_map[td.trial_names[i]]["us_onset"]);
  }
}

void delete_trials_data(trials_data &td) {
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
 *     the main recursive algorithm for determining the number of trials
 * specified from .sess file and encoded withing pt_section. Computes a sum over
 * the current iteration, and for each level of recursion, finishes by
 * multiplying the current sum by the previously obtained number of trials. Of
 * course, only recurses if vec_pair.first is a valid definition of a block.
 */
void calculate_num_trials_helper(
    parsed_trial_section &pt_section,
    std::vector<std::pair<std::string, std::string>> &in_vec,
    uint32_t &temp_num_trials) {
  uint32_t temp_sum = 0;
  for (auto vec_pair : in_vec) {
    uint32_t temp_count = std::stoi(vec_pair.second);
    if (pt_section.block_map.find(vec_pair.first) !=
        pt_section.block_map.end()) {
      calculate_num_trials_helper(
          pt_section, pt_section.block_map[vec_pair.first], temp_count);
    }
    temp_sum += temp_count;
  }
  temp_num_trials *= temp_sum;
}

/*
 * Implementation Notes:
 *     the entry point for recursively calculating the number of trials from
 * pt_section. Because sums computed at each recursive layer are multiplied by
 * and assigned to the previous temp_num_trials, we need to "seed" this process
 * with 1.
 *
 */
uint32_t calculate_num_trials(parsed_trial_section &pt_section) {
  uint32_t temp_num_trials = 1;
  calculate_num_trials_helper(pt_section, pt_section.session, temp_num_trials);
  return temp_num_trials;
}

/*
 * Implementation Notes:
 *     Computes the number of trials and allocates/initializes td given the
 * computed number of trials.
 *
 */
void translate_parsed_trials(parsed_sess_file &s_file, trials_data &td) {
  td.num_trials = calculate_num_trials(s_file.parsed_trial_info);
  // no error checking on returned value, uh oh!
  allocate_trials_data(td, td.num_trials);
  initialize_trials_data(td, s_file.parsed_trial_info);
}

std::string parsed_sess_file_to_str(parsed_sess_file &s_file) {
  std::stringstream sess_file_buf;
  sess_file_buf << "{\n";
  // variable sections
  for (auto var_sec : s_file.parsed_var_sections) {
    sess_file_buf << "\t{\n";
    for (auto pair : var_sec.second.param_map) {
      sess_file_buf << "\t\t'" << pair.first << "', '";
      sess_file_buf << pair.second << "'\n";
    }
    sess_file_buf << "\t},\n";
  }
  // trial definitions
  sess_file_buf << "\t{\n";
  for (auto pair : s_file.parsed_trial_info.trial_map) {
    sess_file_buf << "\t\t'" << pair.first << "': {\n";
    for (auto vars : pair.second) {
      sess_file_buf << "\t\t\t{'" << vars.first << "', '";
      sess_file_buf << vars.second << "'},\n";
    }
    sess_file_buf << "\t\t},\n";
  }
  sess_file_buf << "\t},\n";

  // block definitions
  sess_file_buf << "\t{\n";
  for (auto block : s_file.parsed_trial_info.block_map) {
    sess_file_buf << "\t\t'" << block.first << "': {\n";
    for (auto t_pair : block.second) {
      sess_file_buf << "\t\t\t{'" << t_pair.first << "', '";
      sess_file_buf << t_pair.second << "'},\n";
    }
    sess_file_buf << "\t\t},\n";
  }
  sess_file_buf << "\t},\n";

  // session definitions
  sess_file_buf << "\t{\n";
  for (auto bt_pair : s_file.parsed_trial_info.session) {
    sess_file_buf << "\t\t{'" << bt_pair.first << "', '";
    sess_file_buf << bt_pair.second << "'},\n";
  }
  sess_file_buf << "\t}\n";
  sess_file_buf << "}\n";

  return sess_file_buf.str();
}

std::ostream &operator<<(std::ostream &os, parsed_sess_file &s_file) {
  return os << parsed_sess_file_to_str(s_file);
}
