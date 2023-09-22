/*
 * connectivityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "connectivityparams.h"

bool con_params_populated = false;

int mf_x = 0;
int mf_y = 0;
int num_mf = 0;
int gl_x = 0;
int gl_y = 0;
int num_gl = 0;
int gr_x = 0;
int gr_y = 0;
int num_gr = 0;
int go_x = 0;
int go_y = 0;
int num_go = 0;
int ubc_x = 0;
int ubc_y = 0;
int num_ubc = 0;
int num_bc = 0;
int num_sc = 0;
int num_pc = 0;
int num_nc = 0;
int num_io = 0;
int span_mf_to_gl_x = 0;
int span_mf_to_gl_y = 0;
int num_p_mf_to_gl = 0;
int max_num_p_mf_from_mf_to_gl = 0;
int initial_mf_output = 0;
int max_mf_to_gl_attempts = 0;
int span_gl_to_gr_x = 0;
int span_gl_to_gr_y = 0;
int num_p_gl_to_gr = 0;
int low_num_p_gl_from_gl_to_gr = 0;
int max_num_p_gr_from_gl_to_gr = 0;
int max_num_p_gl_from_gl_to_gr = 0;
int low_gl_to_gr_attempts = 0;
int max_gl_to_gr_attempts = 0;
int span_pf_to_go_x = 0;
int span_pf_to_go_y = 0;
int num_p_pf_to_go = 0;
int max_num_p_go_from_gr_to_go = 0;
int max_num_p_gr_from_gr_to_go = 0;
int max_pf_to_go_input = 0;
int max_pf_to_go_attempts = 0;
int span_aa_to_go_x = 0;
int span_aa_to_go_y = 0;
int num_p_aa_to_go = 0;
int max_aa_to_go_input = 0;
int max_aa_to_go_attempts = 0;
int span_go_to_go_x = 0;
int span_go_to_go_y = 0;
int num_p_go_to_go = 0;
int num_con_go_to_go = 0;
int go_go_recip_cons = 0;
int reduce_base_recip_go_go = 0;
int max_go_to_go_attempts = 0;
int span_go_to_go_gj_x = 0;
int span_go_to_go_gj_y = 0;
int num_p_go_to_go_gj = 0;
int span_go_to_gl_x = 0;
int span_go_to_gl_y = 0;
int num_p_go_to_gl = 0;
int max_num_p_gl_from_go_to_gl = 0;
int max_num_p_go_from_go_to_gl = 0;
int max_go_to_gl_attempts = 0;
int span_gl_to_go_x = 0;
int span_gl_to_go_y = 0;
int num_p_gl_to_go = 0;
int low_num_p_gl_from_gl_to_go = 0;
int max_num_p_gl_from_gl_to_go = 0;
int max_num_p_go_from_gl_to_go = 0;
int initial_go_input = 0;
int low_gl_to_go_attempts = 0;
int max_gl_to_go_attempts = 0;
int max_num_p_go_from_go_to_gr = 0;
int max_num_p_gr_from_go_to_gr = 0;
int max_num_p_gr_from_mf_to_gr = 0;
int max_num_p_mf_from_mf_to_gr = 0;
int max_num_p_go_from_mf_to_go = 0;
int max_num_p_mf_from_mf_to_go = 0;
int gr_pf_vel_in_gr_x_per_t_step = 0;
int gr_af_delay_in_t_step = 0;
int num_p_bc_from_bc_to_pc = 0;
int num_p_pc_from_bc_to_pc = 0;
int num_p_bc_from_gr_to_bc = 0;
int num_p_bc_from_gr_to_bc_p2 = 0;
int num_p_pc_from_pc_to_bc = 0;
int num_p_bc_from_pc_to_bc = 0;
int num_p_sc_from_sc_to_pc = 0;
int num_p_pc_from_sc_to_pc = 0;
int num_p_sc_from_gr_to_sc = 0;
int num_p_sc_from_gr_to_sc_p2 = 0;
int num_p_pc_from_pc_to_nc = 0;
int num_p_nc_from_pc_to_nc = 0;
int num_p_pc_from_gr_to_pc = 0;
int num_p_pc_from_gr_to_pc_p2 = 0;
int num_p_mf_from_mf_to_nc = 0;
int num_p_nc_from_mf_to_nc = 0;
int num_p_nc_from_nc_to_io = 0;
int num_p_io_from_nc_to_io = 0;
int num_p_io_from_io_to_pc = 0;
int num_p_io_in_io_to_io = 0;
int num_p_io_out_io_to_io = 0;

float msPerTimeStep = 0.0;
float numPopHistBinsPC = 0.0;
float ampl_go_to_go = 0.0;
float std_dev_go_to_go = 0.0;
float p_recip_go_go = 0.0;
float p_recip_lower_base_go_go = 0.0;
float ampl_go_to_gl = 0.0;
float std_dev_go_to_gl_ml = 0.0;
float std_dev_go_to_gl_s = 0.0;

float eLeakGO = 0.0;
float threshRestGO = 0.0;
float eLeakGR = 0.0;
float threshRestGR = 0.0;

float eLeakSC = 0.0;
float threshRestSC = 0.0;
float eLeakBC = 0.0;
float threshRestBC = 0.0;
float eLeakPC = 0.0;
float threshRestPC = 0.0;
float initSynWofGRtoPC = 0.0;
float eLeakIO = 0.0;
float threshRestIO = 0.0;
float eLeakNC = 0.0;
float threshRestNC = 0.0;
float initSynWofMFtoNC = 0.0;

void populate_con_params(parsed_build_file &p_file) {
  /* int con params */
  mf_x = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["mf_x"].value);
  mf_y = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["mf_y"].value);
  num_mf = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_mf"].value);
  gl_x = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["gl_x"].value);
  gl_y = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["gl_y"].value);
  num_gl = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_gl"].value);
  gr_x = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["gr_x"].value);
  gr_y = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["gr_y"].value);
  num_gr = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_gr"].value);
  go_x = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["go_x"].value);
  go_y = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["go_y"].value);
  num_go = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_go"].value);
  ubc_x = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["ubc_x"].value);
  ubc_y = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["ubc_y"].value);
  num_ubc = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_ubc"].value);
  num_bc = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_bc"].value);
  num_sc = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_sc"].value);
  num_pc = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_pc"].value);
  num_nc = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_nc"].value);
  num_io = std::stoi(
      p_file.parsed_var_sections["connectivity"].param_map["num_io"].value);
  span_mf_to_gl_x = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_mf_to_gl_x"]
                                  .value);
  span_mf_to_gl_y = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_mf_to_gl_y"]
                                  .value);
  num_p_mf_to_gl = std::stoi(p_file.parsed_var_sections["connectivity"]
                                 .param_map["num_p_mf_to_gl"]
                                 .value);
  max_num_p_mf_from_mf_to_gl =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_mf_from_mf_to_gl"]
                    .value);
  initial_mf_output = std::stoi(p_file.parsed_var_sections["connectivity"]
                                    .param_map["initial_mf_output"]
                                    .value);
  max_mf_to_gl_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["max_mf_to_gl_attempts"]
                                        .value);
  span_gl_to_gr_x = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_gl_to_gr_x"]
                                  .value);
  span_gl_to_gr_y = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_gl_to_gr_y"]
                                  .value);
  num_p_gl_to_gr = std::stoi(p_file.parsed_var_sections["connectivity"]
                                 .param_map["num_p_gl_to_gr"]
                                 .value);
  low_num_p_gl_from_gl_to_gr =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["low_num_p_gl_from_gl_to_gr"]
                    .value);
  max_num_p_gr_from_gl_to_gr =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_gr_from_gl_to_gr"]
                    .value);
  max_num_p_gl_from_gl_to_gr =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_gl_from_gl_to_gr"]
                    .value);
  low_gl_to_gr_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["low_gl_to_gr_attempts"]
                                        .value);
  max_gl_to_gr_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["max_gl_to_gr_attempts"]
                                        .value);
  span_pf_to_go_x = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_pf_to_go_x"]
                                  .value);
  span_pf_to_go_y = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_pf_to_go_y"]
                                  .value);
  num_p_pf_to_go = std::stoi(p_file.parsed_var_sections["connectivity"]
                                 .param_map["num_p_pf_to_go"]
                                 .value);
  max_num_p_go_from_gr_to_go =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_go_from_gr_to_go"]
                    .value);
  max_num_p_gr_from_gr_to_go =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_gr_from_gr_to_go"]
                    .value);
  max_pf_to_go_input = std::stoi(p_file.parsed_var_sections["connectivity"]
                                     .param_map["max_pf_to_go_input"]
                                     .value);
  max_pf_to_go_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["max_pf_to_go_attempts"]
                                        .value);
  span_aa_to_go_x = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_aa_to_go_x"]
                                  .value);
  span_aa_to_go_y = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_aa_to_go_y"]
                                  .value);
  num_p_aa_to_go = std::stoi(p_file.parsed_var_sections["connectivity"]
                                 .param_map["num_p_aa_to_go"]
                                 .value);
  max_aa_to_go_input = std::stoi(p_file.parsed_var_sections["connectivity"]
                                     .param_map["max_aa_to_go_input"]
                                     .value);
  max_aa_to_go_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["max_aa_to_go_attempts"]
                                        .value);
  span_go_to_go_x = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_go_to_go_x"]
                                  .value);
  span_go_to_go_y = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_go_to_go_y"]
                                  .value);
  num_p_go_to_go = std::stoi(p_file.parsed_var_sections["connectivity"]
                                 .param_map["num_p_go_to_go"]
                                 .value);
  num_con_go_to_go = std::stoi(p_file.parsed_var_sections["connectivity"]
                                   .param_map["num_con_go_to_go"]
                                   .value);
  go_go_recip_cons = std::stoi(p_file.parsed_var_sections["connectivity"]
                                   .param_map["go_go_recip_cons"]
                                   .value);
  reduce_base_recip_go_go = std::stoi(p_file.parsed_var_sections["connectivity"]
                                          .param_map["reduce_base_recip_go_go"]
                                          .value);
  max_go_to_go_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["max_go_to_go_attempts"]
                                        .value);
  span_go_to_go_gj_x = std::stoi(p_file.parsed_var_sections["connectivity"]
                                     .param_map["span_go_to_go_gj_x"]
                                     .value);
  span_go_to_go_gj_y = std::stoi(p_file.parsed_var_sections["connectivity"]
                                     .param_map["span_go_to_go_gj_y"]
                                     .value);
  num_p_go_to_go_gj = std::stoi(p_file.parsed_var_sections["connectivity"]
                                    .param_map["num_p_go_to_go_gj"]
                                    .value);
  span_go_to_gl_x = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_go_to_gl_x"]
                                  .value);
  span_go_to_gl_y = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_go_to_gl_y"]
                                  .value);
  num_p_go_to_gl = std::stoi(p_file.parsed_var_sections["connectivity"]
                                 .param_map["num_p_go_to_gl"]
                                 .value);
  max_num_p_gl_from_go_to_gl =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_gl_from_go_to_gl"]
                    .value);
  max_num_p_go_from_go_to_gl =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_go_from_go_to_gl"]
                    .value);
  max_go_to_gl_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["max_go_to_gl_attempts"]
                                        .value);
  span_gl_to_go_x = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_gl_to_go_x"]
                                  .value);
  span_gl_to_go_y = std::stoi(p_file.parsed_var_sections["connectivity"]
                                  .param_map["span_gl_to_go_y"]
                                  .value);
  num_p_gl_to_go = std::stoi(p_file.parsed_var_sections["connectivity"]
                                 .param_map["num_p_gl_to_go"]
                                 .value);
  low_num_p_gl_from_gl_to_go =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["low_num_p_gl_from_gl_to_go"]
                    .value);
  max_num_p_gl_from_gl_to_go =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_gl_from_gl_to_go"]
                    .value);
  max_num_p_go_from_gl_to_go =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_go_from_gl_to_go"]
                    .value);
  initial_go_input = std::stoi(p_file.parsed_var_sections["connectivity"]
                                   .param_map["initial_go_input"]
                                   .value);
  low_gl_to_go_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["low_gl_to_go_attempts"]
                                        .value);
  max_gl_to_go_attempts = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["max_gl_to_go_attempts"]
                                        .value);
  max_num_p_go_from_go_to_gr =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_go_from_go_to_gr"]
                    .value);
  max_num_p_gr_from_go_to_gr =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_gr_from_go_to_gr"]
                    .value);
  max_num_p_gr_from_mf_to_gr =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_gr_from_mf_to_gr"]
                    .value);
  max_num_p_mf_from_mf_to_gr =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_mf_from_mf_to_gr"]
                    .value);
  max_num_p_go_from_mf_to_go =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_go_from_mf_to_go"]
                    .value);
  max_num_p_mf_from_mf_to_go =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["max_num_p_mf_from_mf_to_go"]
                    .value);
  gr_pf_vel_in_gr_x_per_t_step =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["gr_pf_vel_in_gr_x_per_t_step"]
                    .value);
  gr_af_delay_in_t_step = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["gr_af_delay_in_t_step"]
                                        .value);
  num_p_bc_from_bc_to_pc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_bc_from_bc_to_pc"]
                                         .value);
  num_p_pc_from_bc_to_pc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_pc_from_bc_to_pc"]
                                         .value);
  num_p_bc_from_gr_to_bc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_bc_from_gr_to_bc"]
                                         .value);
  num_p_bc_from_gr_to_bc_p2 =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["num_p_bc_from_gr_to_bc_p2"]
                    .value);
  num_p_pc_from_pc_to_bc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_pc_from_pc_to_bc"]
                                         .value);
  num_p_bc_from_pc_to_bc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_bc_from_pc_to_bc"]
                                         .value);
  num_p_sc_from_sc_to_pc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_sc_from_sc_to_pc"]
                                         .value);
  num_p_pc_from_sc_to_pc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_pc_from_sc_to_pc"]
                                         .value);
  num_p_sc_from_gr_to_sc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_sc_from_gr_to_sc"]
                                         .value);
  num_p_sc_from_gr_to_sc_p2 =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["num_p_sc_from_gr_to_sc_p2"]
                    .value);
  num_p_pc_from_pc_to_nc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_pc_from_pc_to_nc"]
                                         .value);
  num_p_nc_from_pc_to_nc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_nc_from_pc_to_nc"]
                                         .value);
  num_p_pc_from_gr_to_pc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_pc_from_gr_to_pc"]
                                         .value);
  num_p_pc_from_gr_to_pc_p2 =
      std::stoi(p_file.parsed_var_sections["connectivity"]
                    .param_map["num_p_pc_from_gr_to_pc_p2"]
                    .value);
  num_p_mf_from_mf_to_nc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_mf_from_mf_to_nc"]
                                         .value);
  num_p_nc_from_mf_to_nc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_nc_from_mf_to_nc"]
                                         .value);
  num_p_nc_from_nc_to_io = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_nc_from_nc_to_io"]
                                         .value);
  num_p_io_from_nc_to_io = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_io_from_nc_to_io"]
                                         .value);
  num_p_io_from_io_to_pc = std::stoi(p_file.parsed_var_sections["connectivity"]
                                         .param_map["num_p_io_from_io_to_pc"]
                                         .value);
  num_p_io_in_io_to_io = std::stoi(p_file.parsed_var_sections["connectivity"]
                                       .param_map["num_p_io_in_io_to_io"]
                                       .value);
  num_p_io_out_io_to_io = std::stoi(p_file.parsed_var_sections["connectivity"]
                                        .param_map["num_p_io_out_io_to_io"]
                                        .value);

  /* float con params */
  msPerTimeStep = std::stof(p_file.parsed_var_sections["connectivity"]
                                .param_map["msPerTimeStep"]
                                .value);
  numPopHistBinsPC = std::stof(p_file.parsed_var_sections["connectivity"]
                                   .param_map["numPopHistBinsPC"]
                                   .value);
  ampl_go_to_go = std::stof(p_file.parsed_var_sections["connectivity"]
                                .param_map["ampl_go_to_go"]
                                .value);
  std_dev_go_to_go = std::stof(p_file.parsed_var_sections["connectivity"]
                                   .param_map["std_dev_go_to_go"]
                                   .value);
  p_recip_go_go = std::stof(p_file.parsed_var_sections["connectivity"]
                                .param_map["p_recip_go_go"]
                                .value);
  p_recip_lower_base_go_go =
      std::stof(p_file.parsed_var_sections["connectivity"]
                    .param_map["p_recip_lower_base_go_go"]
                    .value);
  ampl_go_to_gl = std::stof(p_file.parsed_var_sections["connectivity"]
                                .param_map["ampl_go_to_gl"]
                                .value);
  std_dev_go_to_gl_ml = std::stof(p_file.parsed_var_sections["connectivity"]
                                      .param_map["std_dev_go_to_gl_ml"]
                                      .value);
  std_dev_go_to_gl_s = std::stof(p_file.parsed_var_sections["connectivity"]
                                     .param_map["std_dev_go_to_gl_s"]
                                     .value);

  /* act params */
  eLeakGO = std::stof(
      p_file.parsed_var_sections["activity"].param_map["eLeakGO"].value);
  threshRestGO = std::stof(
      p_file.parsed_var_sections["activity"].param_map["threshRestGO"].value);
  eLeakGR = std::stof(
      p_file.parsed_var_sections["activity"].param_map["eLeakGR"].value);
  threshRestGR = std::stof(
      p_file.parsed_var_sections["activity"].param_map["threshRestGR"].value);

  eLeakSC = std::stof(
      p_file.parsed_var_sections["activity"].param_map["eLeakSC"].value);
  threshRestSC = std::stof(
      p_file.parsed_var_sections["activity"].param_map["threshRestSC"].value);
  eLeakBC = std::stof(
      p_file.parsed_var_sections["activity"].param_map["eLeakBC"].value);
  threshRestBC = std::stof(
      p_file.parsed_var_sections["activity"].param_map["threshRestBC"].value);
  eLeakPC = std::stof(
      p_file.parsed_var_sections["activity"].param_map["eLeakPC"].value);
  threshRestPC = std::stof(
      p_file.parsed_var_sections["activity"].param_map["threshRestPC"].value);
  initSynWofGRtoPC = std::stof(p_file.parsed_var_sections["activity"]
                                   .param_map["initSynWofGRtoPC"]
                                   .value);
  eLeakIO = std::stof(
      p_file.parsed_var_sections["activity"].param_map["eLeakIO"].value);
  threshRestIO = std::stof(
      p_file.parsed_var_sections["activity"].param_map["threshRestIO"].value);
  eLeakNC = std::stof(
      p_file.parsed_var_sections["activity"].param_map["eLeakNC"].value);
  threshRestNC = std::stof(
      p_file.parsed_var_sections["activity"].param_map["threshRestNC"].value);
  initSynWofMFtoNC = std::stof(p_file.parsed_var_sections["activity"]
                                   .param_map["initSynWofMFtoNC"]
                                   .value);

  con_params_populated = true;
}

void read_con_params(std::fstream &in_param_buf) {
  /* not checking whether these things are zeros or not... */
  in_param_buf.read((char *)&mf_x, sizeof(int));
  in_param_buf.read((char *)&mf_y, sizeof(int));
  in_param_buf.read((char *)&num_mf, sizeof(int));
  in_param_buf.read((char *)&gl_x, sizeof(int));
  in_param_buf.read((char *)&gl_y, sizeof(int));
  in_param_buf.read((char *)&num_gl, sizeof(int));
  in_param_buf.read((char *)&gr_x, sizeof(int));
  in_param_buf.read((char *)&gr_y, sizeof(int));
  in_param_buf.read((char *)&num_gr, sizeof(int));
  in_param_buf.read((char *)&go_x, sizeof(int));
  in_param_buf.read((char *)&go_y, sizeof(int));
  in_param_buf.read((char *)&num_go, sizeof(int));
  in_param_buf.read((char *)&ubc_x, sizeof(int));
  in_param_buf.read((char *)&ubc_y, sizeof(int));
  in_param_buf.read((char *)&num_ubc, sizeof(int));
  in_param_buf.read((char *)&num_bc, sizeof(int));
  in_param_buf.read((char *)&num_sc, sizeof(int));
  in_param_buf.read((char *)&num_pc, sizeof(int));
  in_param_buf.read((char *)&num_nc, sizeof(int));
  in_param_buf.read((char *)&num_io, sizeof(int));
  in_param_buf.read((char *)&span_mf_to_gl_x, sizeof(int));
  in_param_buf.read((char *)&span_mf_to_gl_y, sizeof(int));
  in_param_buf.read((char *)&num_p_mf_to_gl, sizeof(int));
  in_param_buf.read((char *)&max_num_p_mf_from_mf_to_gl, sizeof(int));
  in_param_buf.read((char *)&initial_mf_output, sizeof(int));
  in_param_buf.read((char *)&max_mf_to_gl_attempts, sizeof(int));
  in_param_buf.read((char *)&span_gl_to_gr_x, sizeof(int));
  in_param_buf.read((char *)&span_gl_to_gr_y, sizeof(int));
  in_param_buf.read((char *)&num_p_gl_to_gr, sizeof(int));
  in_param_buf.read((char *)&low_num_p_gl_from_gl_to_gr, sizeof(int));
  in_param_buf.read((char *)&max_num_p_gr_from_gl_to_gr, sizeof(int));
  in_param_buf.read((char *)&max_num_p_gl_from_gl_to_gr, sizeof(int));
  in_param_buf.read((char *)&low_gl_to_gr_attempts, sizeof(int));
  in_param_buf.read((char *)&max_gl_to_gr_attempts, sizeof(int));
  in_param_buf.read((char *)&span_pf_to_go_x, sizeof(int));
  in_param_buf.read((char *)&span_pf_to_go_y, sizeof(int));
  in_param_buf.read((char *)&num_p_pf_to_go, sizeof(int));
  in_param_buf.read((char *)&max_num_p_go_from_gr_to_go, sizeof(int));
  in_param_buf.read((char *)&max_num_p_gr_from_gr_to_go, sizeof(int));
  in_param_buf.read((char *)&max_pf_to_go_input, sizeof(int));
  in_param_buf.read((char *)&max_pf_to_go_attempts, sizeof(int));
  in_param_buf.read((char *)&span_aa_to_go_x, sizeof(int));
  in_param_buf.read((char *)&span_aa_to_go_y, sizeof(int));
  in_param_buf.read((char *)&num_p_aa_to_go, sizeof(int));
  in_param_buf.read((char *)&max_aa_to_go_input, sizeof(int));
  in_param_buf.read((char *)&max_aa_to_go_attempts, sizeof(int));
  in_param_buf.read((char *)&span_go_to_go_x, sizeof(int));
  in_param_buf.read((char *)&span_go_to_go_y, sizeof(int));
  in_param_buf.read((char *)&num_p_go_to_go, sizeof(int));
  in_param_buf.read((char *)&num_con_go_to_go, sizeof(int));
  in_param_buf.read((char *)&go_go_recip_cons, sizeof(int));
  in_param_buf.read((char *)&reduce_base_recip_go_go, sizeof(int));
  in_param_buf.read((char *)&max_go_to_go_attempts, sizeof(int));
  in_param_buf.read((char *)&span_go_to_go_gj_x, sizeof(int));
  in_param_buf.read((char *)&span_go_to_go_gj_y, sizeof(int));
  in_param_buf.read((char *)&num_p_go_to_go_gj, sizeof(int));
  in_param_buf.read((char *)&span_go_to_gl_x, sizeof(int));
  in_param_buf.read((char *)&span_go_to_gl_y, sizeof(int));
  in_param_buf.read((char *)&num_p_go_to_gl, sizeof(int));
  in_param_buf.read((char *)&max_num_p_gl_from_go_to_gl, sizeof(int));
  in_param_buf.read((char *)&max_num_p_go_from_go_to_gl, sizeof(int));
  in_param_buf.read((char *)&max_go_to_gl_attempts, sizeof(int));
  in_param_buf.read((char *)&span_gl_to_go_x, sizeof(int));
  in_param_buf.read((char *)&span_gl_to_go_y, sizeof(int));
  in_param_buf.read((char *)&num_p_gl_to_go, sizeof(int));
  in_param_buf.read((char *)&low_num_p_gl_from_gl_to_go, sizeof(int));
  in_param_buf.read((char *)&max_num_p_gl_from_gl_to_go, sizeof(int));
  in_param_buf.read((char *)&max_num_p_go_from_gl_to_go, sizeof(int));
  in_param_buf.read((char *)&initial_go_input, sizeof(int));
  in_param_buf.read((char *)&low_gl_to_go_attempts, sizeof(int));
  in_param_buf.read((char *)&max_gl_to_go_attempts, sizeof(int));
  in_param_buf.read((char *)&max_num_p_go_from_go_to_gr, sizeof(int));
  in_param_buf.read((char *)&max_num_p_gr_from_go_to_gr, sizeof(int));
  in_param_buf.read((char *)&max_num_p_gr_from_mf_to_gr, sizeof(int));
  in_param_buf.read((char *)&max_num_p_mf_from_mf_to_gr, sizeof(int));
  in_param_buf.read((char *)&max_num_p_go_from_mf_to_go, sizeof(int));
  in_param_buf.read((char *)&max_num_p_mf_from_mf_to_go, sizeof(int));
  in_param_buf.read((char *)&gr_pf_vel_in_gr_x_per_t_step, sizeof(int));
  in_param_buf.read((char *)&gr_af_delay_in_t_step, sizeof(int));
  in_param_buf.read((char *)&num_p_bc_from_bc_to_pc, sizeof(int));
  in_param_buf.read((char *)&num_p_pc_from_bc_to_pc, sizeof(int));
  in_param_buf.read((char *)&num_p_bc_from_gr_to_bc, sizeof(int));
  in_param_buf.read((char *)&num_p_bc_from_gr_to_bc_p2, sizeof(int));
  in_param_buf.read((char *)&num_p_pc_from_pc_to_bc, sizeof(int));
  in_param_buf.read((char *)&num_p_bc_from_pc_to_bc, sizeof(int));
  in_param_buf.read((char *)&num_p_sc_from_sc_to_pc, sizeof(int));
  in_param_buf.read((char *)&num_p_pc_from_sc_to_pc, sizeof(int));
  in_param_buf.read((char *)&num_p_sc_from_gr_to_sc, sizeof(int));
  in_param_buf.read((char *)&num_p_sc_from_gr_to_sc_p2, sizeof(int));
  in_param_buf.read((char *)&num_p_pc_from_pc_to_nc, sizeof(int));
  in_param_buf.read((char *)&num_p_nc_from_pc_to_nc, sizeof(int));
  in_param_buf.read((char *)&num_p_pc_from_gr_to_pc, sizeof(int));
  in_param_buf.read((char *)&num_p_pc_from_gr_to_pc_p2, sizeof(int));
  in_param_buf.read((char *)&num_p_mf_from_mf_to_nc, sizeof(int));
  in_param_buf.read((char *)&num_p_nc_from_mf_to_nc, sizeof(int));
  in_param_buf.read((char *)&num_p_nc_from_nc_to_io, sizeof(int));
  in_param_buf.read((char *)&num_p_io_from_nc_to_io, sizeof(int));
  in_param_buf.read((char *)&num_p_io_from_io_to_pc, sizeof(int));
  in_param_buf.read((char *)&num_p_io_in_io_to_io, sizeof(int));
  in_param_buf.read((char *)&num_p_io_out_io_to_io, sizeof(int));

  in_param_buf.read((char *)&msPerTimeStep, sizeof(float));
  in_param_buf.read((char *)&numPopHistBinsPC, sizeof(float));
  in_param_buf.read((char *)&ampl_go_to_go, sizeof(float));
  in_param_buf.read((char *)&std_dev_go_to_go, sizeof(float));
  in_param_buf.read((char *)&p_recip_go_go, sizeof(float));
  in_param_buf.read((char *)&p_recip_lower_base_go_go, sizeof(float));
  in_param_buf.read((char *)&ampl_go_to_gl, sizeof(float));
  in_param_buf.read((char *)&std_dev_go_to_gl_ml, sizeof(float));
  in_param_buf.read((char *)&std_dev_go_to_gl_s, sizeof(float));

  /* act params */
  in_param_buf.read((char *)&eLeakGO, sizeof(float));
  in_param_buf.read((char *)&threshRestGO, sizeof(float));
  in_param_buf.read((char *)&eLeakGR, sizeof(float));
  in_param_buf.read((char *)&threshRestGR, sizeof(float));

  in_param_buf.read((char *)&eLeakSC, sizeof(float));
  in_param_buf.read((char *)&threshRestSC, sizeof(float));
  in_param_buf.read((char *)&eLeakBC, sizeof(float));
  in_param_buf.read((char *)&threshRestBC, sizeof(float));
  in_param_buf.read((char *)&eLeakPC, sizeof(float));
  in_param_buf.read((char *)&threshRestPC, sizeof(float));
  in_param_buf.read((char *)&initSynWofGRtoPC, sizeof(float));
  in_param_buf.read((char *)&eLeakIO, sizeof(float));
  in_param_buf.read((char *)&threshRestIO, sizeof(float));
  in_param_buf.read((char *)&eLeakNC, sizeof(float));
  in_param_buf.read((char *)&threshRestNC, sizeof(float));
  in_param_buf.read((char *)&initSynWofMFtoNC, sizeof(float));

  con_params_populated = true;
}

void write_con_params(std::fstream &out_param_buf) {
  /* not checking whether these things are zeros or not... */
  out_param_buf.write((char *)&mf_x, sizeof(int));
  out_param_buf.write((char *)&mf_y, sizeof(int));
  out_param_buf.write((char *)&num_mf, sizeof(int));
  out_param_buf.write((char *)&gl_x, sizeof(int));
  out_param_buf.write((char *)&gl_y, sizeof(int));
  out_param_buf.write((char *)&num_gl, sizeof(int));
  out_param_buf.write((char *)&gr_x, sizeof(int));
  out_param_buf.write((char *)&gr_y, sizeof(int));
  out_param_buf.write((char *)&num_gr, sizeof(int));
  out_param_buf.write((char *)&go_x, sizeof(int));
  out_param_buf.write((char *)&go_y, sizeof(int));
  out_param_buf.write((char *)&num_go, sizeof(int));
  out_param_buf.write((char *)&ubc_x, sizeof(int));
  out_param_buf.write((char *)&ubc_y, sizeof(int));
  out_param_buf.write((char *)&num_ubc, sizeof(int));
  out_param_buf.write((char *)&num_bc, sizeof(int));
  out_param_buf.write((char *)&num_sc, sizeof(int));
  out_param_buf.write((char *)&num_pc, sizeof(int));
  out_param_buf.write((char *)&num_nc, sizeof(int));
  out_param_buf.write((char *)&num_io, sizeof(int));
  out_param_buf.write((char *)&span_mf_to_gl_x, sizeof(int));
  out_param_buf.write((char *)&span_mf_to_gl_y, sizeof(int));
  out_param_buf.write((char *)&num_p_mf_to_gl, sizeof(int));
  out_param_buf.write((char *)&max_num_p_mf_from_mf_to_gl, sizeof(int));
  out_param_buf.write((char *)&initial_mf_output, sizeof(int));
  out_param_buf.write((char *)&max_mf_to_gl_attempts, sizeof(int));
  out_param_buf.write((char *)&span_gl_to_gr_x, sizeof(int));
  out_param_buf.write((char *)&span_gl_to_gr_y, sizeof(int));
  out_param_buf.write((char *)&num_p_gl_to_gr, sizeof(int));
  out_param_buf.write((char *)&low_num_p_gl_from_gl_to_gr, sizeof(int));
  out_param_buf.write((char *)&max_num_p_gr_from_gl_to_gr, sizeof(int));
  out_param_buf.write((char *)&max_num_p_gl_from_gl_to_gr, sizeof(int));
  out_param_buf.write((char *)&low_gl_to_gr_attempts, sizeof(int));
  out_param_buf.write((char *)&max_gl_to_gr_attempts, sizeof(int));
  out_param_buf.write((char *)&span_pf_to_go_x, sizeof(int));
  out_param_buf.write((char *)&span_pf_to_go_y, sizeof(int));
  out_param_buf.write((char *)&num_p_pf_to_go, sizeof(int));
  out_param_buf.write((char *)&max_num_p_go_from_gr_to_go, sizeof(int));
  out_param_buf.write((char *)&max_num_p_gr_from_gr_to_go, sizeof(int));
  out_param_buf.write((char *)&max_pf_to_go_input, sizeof(int));
  out_param_buf.write((char *)&max_pf_to_go_attempts, sizeof(int));
  out_param_buf.write((char *)&span_aa_to_go_x, sizeof(int));
  out_param_buf.write((char *)&span_aa_to_go_y, sizeof(int));
  out_param_buf.write((char *)&num_p_aa_to_go, sizeof(int));
  out_param_buf.write((char *)&max_aa_to_go_input, sizeof(int));
  out_param_buf.write((char *)&max_aa_to_go_attempts, sizeof(int));
  out_param_buf.write((char *)&span_go_to_go_x, sizeof(int));
  out_param_buf.write((char *)&span_go_to_go_y, sizeof(int));
  out_param_buf.write((char *)&num_p_go_to_go, sizeof(int));
  out_param_buf.write((char *)&num_con_go_to_go, sizeof(int));
  out_param_buf.write((char *)&go_go_recip_cons, sizeof(int));
  out_param_buf.write((char *)&reduce_base_recip_go_go, sizeof(int));
  out_param_buf.write((char *)&max_go_to_go_attempts, sizeof(int));
  out_param_buf.write((char *)&span_go_to_go_gj_x, sizeof(int));
  out_param_buf.write((char *)&span_go_to_go_gj_y, sizeof(int));
  out_param_buf.write((char *)&num_p_go_to_go_gj, sizeof(int));
  out_param_buf.write((char *)&span_go_to_gl_x, sizeof(int));
  out_param_buf.write((char *)&span_go_to_gl_y, sizeof(int));
  out_param_buf.write((char *)&num_p_go_to_gl, sizeof(int));
  out_param_buf.write((char *)&max_num_p_gl_from_go_to_gl, sizeof(int));
  out_param_buf.write((char *)&max_num_p_go_from_go_to_gl, sizeof(int));
  out_param_buf.write((char *)&max_go_to_gl_attempts, sizeof(int));
  out_param_buf.write((char *)&span_gl_to_go_x, sizeof(int));
  out_param_buf.write((char *)&span_gl_to_go_y, sizeof(int));
  out_param_buf.write((char *)&num_p_gl_to_go, sizeof(int));
  out_param_buf.write((char *)&low_num_p_gl_from_gl_to_go, sizeof(int));
  out_param_buf.write((char *)&max_num_p_gl_from_gl_to_go, sizeof(int));
  out_param_buf.write((char *)&max_num_p_go_from_gl_to_go, sizeof(int));
  out_param_buf.write((char *)&initial_go_input, sizeof(int));
  out_param_buf.write((char *)&low_gl_to_go_attempts, sizeof(int));
  out_param_buf.write((char *)&max_gl_to_go_attempts, sizeof(int));
  out_param_buf.write((char *)&max_num_p_go_from_go_to_gr, sizeof(int));
  out_param_buf.write((char *)&max_num_p_gr_from_go_to_gr, sizeof(int));
  out_param_buf.write((char *)&max_num_p_gr_from_mf_to_gr, sizeof(int));
  out_param_buf.write((char *)&max_num_p_mf_from_mf_to_gr, sizeof(int));
  out_param_buf.write((char *)&max_num_p_go_from_mf_to_go, sizeof(int));
  out_param_buf.write((char *)&max_num_p_mf_from_mf_to_go, sizeof(int));
  out_param_buf.write((char *)&gr_pf_vel_in_gr_x_per_t_step, sizeof(int));
  out_param_buf.write((char *)&gr_af_delay_in_t_step, sizeof(int));
  out_param_buf.write((char *)&num_p_bc_from_bc_to_pc, sizeof(int));
  out_param_buf.write((char *)&num_p_pc_from_bc_to_pc, sizeof(int));
  out_param_buf.write((char *)&num_p_bc_from_gr_to_bc, sizeof(int));
  out_param_buf.write((char *)&num_p_bc_from_gr_to_bc_p2, sizeof(int));
  out_param_buf.write((char *)&num_p_pc_from_pc_to_bc, sizeof(int));
  out_param_buf.write((char *)&num_p_bc_from_pc_to_bc, sizeof(int));
  out_param_buf.write((char *)&num_p_sc_from_sc_to_pc, sizeof(int));
  out_param_buf.write((char *)&num_p_pc_from_sc_to_pc, sizeof(int));
  out_param_buf.write((char *)&num_p_sc_from_gr_to_sc, sizeof(int));
  out_param_buf.write((char *)&num_p_sc_from_gr_to_sc_p2, sizeof(int));
  out_param_buf.write((char *)&num_p_pc_from_pc_to_nc, sizeof(int));
  out_param_buf.write((char *)&num_p_nc_from_pc_to_nc, sizeof(int));
  out_param_buf.write((char *)&num_p_pc_from_gr_to_pc, sizeof(int));
  out_param_buf.write((char *)&num_p_pc_from_gr_to_pc_p2, sizeof(int));
  out_param_buf.write((char *)&num_p_mf_from_mf_to_nc, sizeof(int));
  out_param_buf.write((char *)&num_p_nc_from_mf_to_nc, sizeof(int));
  out_param_buf.write((char *)&num_p_nc_from_nc_to_io, sizeof(int));
  out_param_buf.write((char *)&num_p_io_from_nc_to_io, sizeof(int));
  out_param_buf.write((char *)&num_p_io_from_io_to_pc, sizeof(int));
  out_param_buf.write((char *)&num_p_io_in_io_to_io, sizeof(int));
  out_param_buf.write((char *)&num_p_io_out_io_to_io, sizeof(int));

  out_param_buf.write((char *)&msPerTimeStep, sizeof(float));
  out_param_buf.write((char *)&numPopHistBinsPC, sizeof(float));
  out_param_buf.write((char *)&ampl_go_to_go, sizeof(float));
  out_param_buf.write((char *)&std_dev_go_to_go, sizeof(float));
  out_param_buf.write((char *)&p_recip_go_go, sizeof(float));
  out_param_buf.write((char *)&p_recip_lower_base_go_go, sizeof(float));
  out_param_buf.write((char *)&ampl_go_to_gl, sizeof(float));
  out_param_buf.write((char *)&std_dev_go_to_gl_ml, sizeof(float));
  out_param_buf.write((char *)&std_dev_go_to_gl_s, sizeof(float));

  out_param_buf.write((char *)&eLeakGO, sizeof(float));
  out_param_buf.write((char *)&threshRestGO, sizeof(float));
  out_param_buf.write((char *)&eLeakGR, sizeof(float));
  out_param_buf.write((char *)&threshRestGR, sizeof(float));

  out_param_buf.write((char *)&eLeakSC, sizeof(float));
  out_param_buf.write((char *)&threshRestSC, sizeof(float));
  out_param_buf.write((char *)&eLeakBC, sizeof(float));
  out_param_buf.write((char *)&threshRestBC, sizeof(float));
  out_param_buf.write((char *)&eLeakPC, sizeof(float));
  out_param_buf.write((char *)&threshRestPC, sizeof(float));
  out_param_buf.write((char *)&initSynWofGRtoPC, sizeof(float));
  out_param_buf.write((char *)&eLeakIO, sizeof(float));
  out_param_buf.write((char *)&threshRestIO, sizeof(float));
  out_param_buf.write((char *)&eLeakNC, sizeof(float));
  out_param_buf.write((char *)&threshRestNC, sizeof(float));
  out_param_buf.write((char *)&initSynWofMFtoNC, sizeof(float));
}
