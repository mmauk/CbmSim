/*
 * connectivityparams.h
 *
 *  Created on: Oct 15, 2012
 *      Author: varicella
 *
 *  Modified on: 06/09/2022
 *  	Contributor: Sean Gallogly
 */

#ifndef CONNECTIVITYPARAMS_H_
#define CONNECTIVITYPARAMS_H_

#include <iostream>
#include <fstream>
#include <string>
#include "file_parse.h"
#include "pstdint.h"

#define NUM_CON_PARAMS 107

extern bool con_params_populated;

extern int mf_x;
extern int mf_y;
extern int num_mf;
extern int gl_x;
extern int gl_y;
extern int num_gl;
extern int gr_x;
extern int gr_y;
extern int num_gr;
extern int go_x;
extern int go_y;
extern int num_go;
extern int ubc_x;
extern int ubc_y;
extern int num_ubc;
extern int num_bc;
extern int num_sc;
extern int num_pc;
extern int num_nc;
extern int num_io;
extern int span_mf_to_gl_x;
extern int span_mf_to_gl_y;
extern int num_p_mf_to_gl;
extern int max_num_p_mf_from_mf_to_gl;
extern int initial_mf_output;
extern int max_mf_to_gl_attempts;
extern int span_gl_to_gr_x;
extern int span_gl_to_gr_y;
extern int num_p_gl_to_gr;
extern int low_num_p_gl_from_gl_to_gr;
extern int max_num_p_gr_from_gl_to_gr;
extern int max_num_p_gl_from_gl_to_gr;
extern int low_gl_to_gr_attempts;
extern int max_gl_to_gr_attempts;
extern int span_pf_to_go_x;
extern int span_pf_to_go_y;
extern int num_p_pf_to_go;
extern int max_num_p_go_from_gr_to_go;
extern int max_num_p_gr_from_gr_to_go;
extern int max_pf_to_go_input;
extern int max_pf_to_go_attempts;
extern int span_aa_to_go_x;
extern int span_aa_to_go_y;
extern int num_p_aa_to_go;
extern int max_aa_to_go_input;
extern int max_aa_to_go_attempts;
extern int span_go_to_go_x;
extern int span_go_to_go_y;
extern int num_p_go_to_go;
extern int num_con_go_to_go;
extern int go_go_recip_cons;
extern int reduce_base_recip_go_go;
extern int max_go_to_go_attempts;
extern int span_go_to_go_gj_x;
extern int span_go_to_go_gj_y;
extern int num_p_go_to_go_gj;
extern int span_go_to_gl_x;
extern int span_go_to_gl_y;
extern int num_p_go_to_gl;
extern int max_num_p_gl_from_go_to_gl;
extern int max_num_p_go_from_go_to_gl;
extern int max_go_to_gl_attempts;
extern int span_gl_to_go_x;
extern int span_gl_to_go_y;
extern int num_p_gl_to_go;
extern int low_num_p_gl_from_gl_to_go;
extern int max_num_p_gl_from_gl_to_go;
extern int max_num_p_go_from_gl_to_go;
extern int initial_go_input;
extern int low_gl_to_go_attempts;
extern int max_gl_to_go_attempts;
extern int max_num_p_go_from_go_to_gr;
extern int max_num_p_gr_from_go_to_gr;
extern int max_num_p_gr_from_mf_to_gr;
extern int max_num_p_mf_from_mf_to_gr;
extern int max_num_p_go_from_mf_to_go;
extern int max_num_p_mf_from_mf_to_go;
extern int gr_pf_vel_in_gr_x_per_t_step;
extern int gr_af_delay_in_t_step;
extern int num_p_bc_from_bc_to_pc;
extern int num_p_pc_from_bc_to_pc;
extern int num_p_bc_from_gr_to_bc;
extern int num_p_bc_from_gr_to_bc_p2;
extern int num_p_pc_from_pc_to_bc;
extern int num_p_bc_from_pc_to_bc;
extern int num_p_sc_from_sc_to_pc;
extern int num_p_pc_from_sc_to_pc;
extern int num_p_sc_from_gr_to_sc;
extern int num_p_sc_from_gr_to_sc_p2;
extern int num_p_pc_from_pc_to_nc;
extern int num_p_nc_from_pc_to_nc;
extern int num_p_pc_from_gr_to_pc;
extern int num_p_pc_from_gr_to_pc_p2;
extern int num_p_mf_from_mf_to_nc;
extern int num_p_nc_from_mf_to_nc;
extern int num_p_nc_from_nc_to_io;
extern int num_p_io_from_nc_to_io;
extern int num_p_io_from_io_to_pc;
extern int num_p_io_in_io_to_io;
extern int num_p_io_out_io_to_io;

extern float msPerTimeStep;
extern float ampl_go_to_go;
extern float std_dev_go_to_go;
extern float p_recip_go_go;
extern float p_recip_lower_base_go_go;
extern float ampl_go_to_gl;
extern float std_dev_go_to_gl_ml;
extern float std_dev_go_to_gl_s;

extern float eLeakGO;
extern float threshRestGO;
extern float eLeakGR; 
extern float threshRestGR;

extern float eLeakSC;
extern float threshRestSC;
extern float eLeakBC;
extern float threshRestBC;
extern float eLeakPC;
extern float threshRestPC;
extern float initSynWofGRtoPC;
extern float eLeakIO;
extern float threshRestIO;
extern float eLeakNC;
extern float threshRestNC;
extern float initSynWofMFtoNC;

void populate_con_params(parsed_build_file &p_file);
void read_con_params(std::fstream &in_param_buf);
void write_con_params(std::fstream &out_param_buf);

#endif /* CONNECTIVITYPARAMS_H_ */

