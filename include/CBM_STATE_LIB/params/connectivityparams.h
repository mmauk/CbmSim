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
#include <unordered_map>
#include "fileIO/build_file.h"
#include "stdDefinitions/pstdint.h"

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

extern float ampl_go_to_go;
extern float std_dev_go_to_go;
extern float p_recip_go_go;
extern float p_recip_lower_base_go_go;
extern float ampl_go_to_gl;
extern float std_dev_go_to_gl_ml;
extern float std_dev_go_to_gl_s;

void populate_con_params(parsed_build_file &p_file);
void read_con_params(std::fstream &in_param_buf);
void write_con_params(std::fstream &out_param_buf);

/* ================================= OLD PARAMS ============================== */

class ConnectivityParams
{
	public:
		ConnectivityParams();
		ConnectivityParams(parsed_build_file &p_file);
		ConnectivityParams(std::fstream &sim_file_buf);
		void readParams(std::fstream &inParamBuf);
		void writeParams(std::fstream &outParamBuf);
		std::string toString();
		
		friend std::ostream &operator<<(std::ostream &os, ConnectivityParams &cp);

		std::unordered_map<std::string, int> int_params;
		std::unordered_map<std::string, float> float_params;
};

std::ostream &operator<<(std::ostream &os, ConnectivityParams &cp);

/* ================================= OLD OLD PARAMS ============================== */

// cell numbers
const int MF_X = 128;
const int MF_Y = 32;
const int NUM_MF = 4096;

const int GL_X = 512;
const int GL_Y = 128;
const int NUM_GL = 65536;

const int GR_X = 2048;
const int GR_Y = 512;
const int NUM_GR = 1048576;

const int GO_X = 128;
const int GO_Y = 32;
const int NUM_GO = 4096; 

const int UBC_X = 64;
const int UBC_Y = 16;
const int NUM_UBC = 1024;

// mzone cell numbers
const int NUM_BC = 128;
const int NUM_SC = 512;
const int NUM_PC = 32;
const int NUM_NC = 8;
const int NUM_IO = 4;

// mf -> gl (uses Joe's Con alg (06/01/2022))
const int SPAN_MF_TO_GL_X = 512; 
const int SPAN_MF_TO_GL_Y = 128;
const int NUM_P_MF_TO_GL = 66177;

// before was 40 as a conservative size (06/07/2022)
const int MAX_NUM_P_MF_FROM_MF_TO_GL = 16;

// NOTE: not a maxnump as alg is adaptive
const int INITIAL_MF_OUTPUT = 14;
const int MAX_MF_TO_GL_ATTEMPTS = 4;

// gl -> gr (uses connectCommon (06/01/2022))
const int SPAN_GL_TO_GR_X = 4;
const int SPAN_GL_TO_GR_Y = 4;
const int NUM_P_GL_TO_GR = 25;

const int LOW_NUM_P_GL_FROM_GL_TO_GR = 80;
const int MAX_NUM_P_GR_FROM_GL_TO_GR = 5;
const int MAX_NUM_P_GL_FROM_GL_TO_GR = 200;

const int LOW_GL_TO_GR_ATTEMPTS = 20000;
const int MAX_GL_TO_GR_ATTEMPTS = 50000;

// gr -> go (both pf and aa use Joe's Con Alg (06/01/2022))
const int SPAN_PF_TO_GO_X = 2048;
const int SPAN_PF_TO_GO_Y = 150;
const int NUM_P_PF_TO_GO = 309399;

const int MAX_NUM_P_GO_FROM_GR_TO_GO = 5000;
const int MAX_NUM_P_GR_FROM_GR_TO_GO = 50;

// one value we play around with
const int MAX_PF_TO_GO_INPUT = 3750;
const int MAX_PF_TO_GO_ATTEMPTS = 5;

const int SPAN_AA_TO_GO_X = 150; /* was 201 before tf? */
const int SPAN_AA_TO_GO_Y = 150; /* same w this one */
const int NUM_P_AA_TO_GO = 22801;

// one value we play around with
const int MAX_AA_TO_GO_INPUT = 1250;
const int MAX_AA_TO_GO_ATTEMPTS = 15;

// go <-> go (uses Joe's Con Alg (06/01/2022))
const int SPAN_GO_TO_GO_X = 10;	
const int SPAN_GO_TO_GO_Y = 10;	
const int NUM_P_GO_TO_GO = 121;

const int NUM_CON_GO_TO_GO = 12;

const float AMPL_GO_TO_GO = 0.35f;
const float STD_DEV_GO_TO_GO = 1.95f;
const float P_RECIP_GO_GO = 1.0f;
const float P_RECIP_LOWER_BASE_GO_GO = 0.0f;
const bool GO_GO_RECIP_CONS = true;
const bool REDUCE_BASE_RECIP_GO_GO = false;

const int MAX_GO_TO_GO_ATTEMPTS = 200;

// go <-> go gap junctions
const int SPAN_GO_TO_GO_GJ_X = 8;	
const int SPAN_GO_TO_GO_GJ_Y = 8;	
const int NUM_P_GO_TO_GO_GJ = 81;

// go -> gl (uses Joe's Con Alg (06/01/2022))
const int SPAN_GO_TO_GL_X = 56;
const int SPAN_GO_TO_GL_Y = 56;
const int NUM_P_GO_TO_GL = 3249;

const int MAX_NUM_P_GL_FROM_GO_TO_GL = 1;
const int MAX_NUM_P_GO_FROM_GO_TO_GL = 64;

const float AMPL_GO_TO_GL = 0.01f;
const float STD_DEV_GO_TO_GL_ML = 100.0f;
const float STD_DEV_GO_TO_GL_S = 100.0f;

const int MAX_GO_TO_GL_ATTEMPTS = 100;

// gl -> go (uses connectCommon (06/01/2022))
// NOTE: span values taken from spanGODecDenOnGL*
const int SPAN_GL_TO_GO_X = 12;
const int SPAN_GL_TO_GO_Y = 12;
const int NUM_P_GL_TO_GO = 169;

const int LOW_NUM_P_GL_FROM_GL_TO_GO = 1;
const int MAX_NUM_P_GL_FROM_GL_TO_GO = 1;
const int MAX_NUM_P_GO_FROM_GL_TO_GO = 16;

const int INITIAL_GO_INPUT = 1;

const int LOW_GL_TO_GO_ATTEMPTS = 20000;
const int MAX_GL_TO_GO_ATTEMPTS = 50000;

// TRANSLATIONS
// go -> gr
const int MAX_NUM_P_GO_FROM_GO_TO_GR = 12800; /* might not use */
const int MAX_NUM_P_GR_FROM_GO_TO_GR = 3;

// mf -> gr
const int MAX_NUM_P_GR_FROM_MF_TO_GR = 5;	
const int MAX_NUM_P_MF_FROM_MF_TO_GR = 4000; /* 20 * MAX_NUM_P_GL_FROM_GL_TO_GR */

// mf -> go
const int MAX_NUM_P_GO_FROM_MF_TO_GO = 16; /* was 40 before (06/07/2022) */
const int MAX_NUM_P_MF_FROM_MF_TO_GO = 20; /* 20 * MAX_NUM_P_GL_FROM_GL_TO_GO */


// delay mask vars
const ct_uint32_t GR_PF_VEL_IN_GR_X_PER_T_STEP = 147;
const ct_uint32_t GR_AF_DELAY_IN_T_STEP = 1;

// MZONE VARS
// bc -> pc
const int NUM_P_BC_FROM_BC_TO_PC = 4;
const int NUM_P_PC_FROM_BC_TO_PC = 16;

// gr -> bc
const int NUM_P_BC_FROM_GR_TO_BC = 8192; /* NUM_GR / NUM_BC */
const int NUM_P_BC_FROM_GR_TO_BC_P2 = 13; /* NUM_GR_P2 - (NUM_PC_P2 + 2) */ 

// pc -> bc
const int NUM_P_PC_FROM_PC_TO_BC = 16;
const int NUM_P_BC_FROM_PC_TO_BC = 4;

// sc -> pc
const int NUM_P_SC_FROM_SC_TO_PC = 1;
const int NUM_P_PC_FROM_SC_TO_PC = 16;

// gr -> sc
const int NUM_P_SC_FROM_GR_TO_SC = 2048; /* NUM_GR / NUM_SC */
const int NUM_P_SC_FROM_GR_TO_SC_P2 = 11; 

// pc -> nc
const int NUM_P_PC_FROM_PC_TO_NC = 3;
const int NUM_P_NC_FROM_PC_TO_NC = 12; /* via connectivity relation */

// gr -> pc
const int NUM_P_PC_FROM_GR_TO_PC = 32768; /* NUM_GR / NUM_PC */
const int NUM_P_PC_FROM_GR_TO_PC_P2 = 15;

// mf -> nc
const int NUM_P_MF_FROM_MF_TO_NC = 1;
const int NUM_P_NC_FROM_MF_TO_NC = 512; /* NUM_MF / NUM_NC */

// nc -> io
const int NUM_P_NC_FROM_NC_TO_IO = 4; /* NUM_IO */
const int NUM_P_IO_FROM_NC_TO_IO = 8;

// io -> pc
const int NUM_P_IO_FROM_IO_TO_PC = 8; /* NUM_PC / NUM_IO */
//const int NUM_P_PC_FROM_IO_TO_PC;

// io <-> io
const int NUM_P_IO_IN_IO_TO_IO = 3; /* NUM_IO - 1 */
const int NUM_P_IO_OUT_IO_TO_IO = 3; /* NUM_IO - 1 */

#endif /* CONNECTIVITYPARAMS_H_ */

