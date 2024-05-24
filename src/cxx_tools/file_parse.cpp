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

static uint32_t calculate_num_trials(json &s_file);
static void translate_trial_names(trials_data &td, json &s_file);
static void translate_trial_props(trials_data &td, json &s_file);

/*
 * Implementation Notes:
 *     allocates memory for arrays in td. Note that it is expected that
 *     num_trials would be determined from the session file via
 *     calculate_num_trials
 *
 */
void allocate_trials_data(trials_data &td, std::string s_file_name) {
  std::ifstream s_file_buf(s_file_name);
  json s_file = json::parse(s_file_buf);
  s_file_buf.close();
  td.num_trials = calculate_num_trials(s_file);
  td.trial_names = (std::string *)calloc(td.num_trials, sizeof(std::string));
  td.use_css = (uint32_t *)calloc(td.num_trials, sizeof(uint32_t));
  td.cs_onsets = (uint32_t *)calloc(td.num_trials, sizeof(uint32_t));
  td.cs_lens = (uint32_t *)calloc(td.num_trials, sizeof(uint32_t));
  td.cs_percents = (float *)calloc(td.num_trials, sizeof(float));
  td.use_uss = (uint32_t *)calloc(td.num_trials, sizeof(uint32_t));
  td.us_onsets = (uint32_t *)calloc(td.num_trials, sizeof(uint32_t));
}

/*
 * Implementation Notes:
 *     translates data from the json file s_file_name into trials data struct td
 *
 */
void translate_trials(std::string s_file_name, trials_data &td) {
  std::ifstream s_file_buf(s_file_name);
  json s_file = json::parse(s_file_buf);
  translate_trial_names(td, s_file);
  translate_trial_props(td, s_file);
  s_file_buf.close();
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
 *    computes the number of trials for a session iteratively
 */
static uint32_t calculate_num_trials(json &s_file) {
  uint32_t num_trials = 0;
  auto session = s_file.at("session");
  auto blocks = s_file.at("blocks");
  auto trials = s_file.at("trials");
  for (auto elem : session) {
    for (auto &[sess_item_k, sess_item_v] : elem.items()) {
      if (blocks.contains(sess_item_k)) {
        uint32_t block_num_trials = 0;
        for (auto trial : blocks.at(sess_item_k)) {
          for (auto &[trial_name, trial_num] : trial.items()) {
            if (trials.contains(trial_name))
              block_num_trials += trial_num.template get<uint32_t>();
          }
        }
        num_trials += (sess_item_v.template get<uint32_t>() * block_num_trials);
      } else if (trials.contains(sess_item_k)) {
        num_trials += sess_item_v.template get<uint32_t>();
      }
    }
  }
  return num_trials;
}

/*
 * Implementation Notes:
 *     writes the correct trial name at the correct location in td.trial_names
 *     iteratively
 */
static void translate_trial_names(trials_data &td, json &s_file) {
  auto session = s_file.at("session");
  auto blocks = s_file.at("blocks");
  auto trials = s_file.at("trials");
  size_t trial_ctr = 0;
  for (auto elem : session) {
    for (auto &[sess_item_k, sess_item_v] : elem.items()) {
      if (blocks.contains(sess_item_k)) {
        for (size_t i = 0; i < sess_item_v.template get<size_t>(); i++) {
          for (auto trial : blocks.at(sess_item_k)) {
            for (auto &[trial_name, trial_num] : trial.items()) {
              if (trials.contains(trial_name)) {
                for (size_t j = 0; j < trial_num.template get<uint32_t>();
                     j++) {
                  td.trial_names[trial_ctr] = trial_name;
                  trial_ctr++;
                }
              }
            }
          }
        }
      } else if (trials.contains(sess_item_k)) {
        for (size_t i = 0; i < sess_item_v.template get<uint32_t>(); i++) {
          td.trial_names[trial_ctr] = sess_item_k;
          trial_ctr++;
        }
      }
    }
  }
}

/*
 * Implementation Notes:
 *    translates the trial properties in json s_file into each element of
 *    each array in td
 */
static void translate_trial_props(trials_data &td, json &s_file) {
  auto trials = s_file.at("trials");
  for (size_t i = 0; i < td.num_trials; i++) {
    td.use_css[i] =
        trials.at(td.trial_names[i]).at("use_cs").template get<uint32_t>();
    td.cs_onsets[i] =
        trials.at(td.trial_names[i]).at("cs_onset").template get<uint32_t>();
    td.cs_lens[i] =
        trials.at(td.trial_names[i]).at("cs_len").template get<uint32_t>();
    td.cs_percents[i] =
        trials.at(td.trial_names[i]).at("cs_percent").template get<float>();
    td.use_uss[i] =
        trials.at(td.trial_names[i]).at("use_us").template get<uint32_t>();
    td.us_onsets[i] =
        trials.at(td.trial_names[i]).at("us_onset").template get<uint32_t>();
  }
}

