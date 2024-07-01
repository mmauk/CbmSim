/*
 * connectivityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "connectivityparams.h"

// cell numbers
int mf_x = 128;
int mf_y = 32;
int num_mf = 4096;

int gl_x = 512;
int gl_y = 128;
int num_gl = 65536;

int gr_x = 2048;
int gr_y = 512;
int num_gr = 1048576;

int go_x = 128;
int go_y = 32;
int num_go = 4096;

int ubc_x = 64;
int ubc_y = 16;
int num_ubc = 1024;

// mzone numbers
int num_bc = 128;
int num_sc = 512;
int num_compart = 4096;
int num_pc = 32;
int num_nc = 8;
int num_io = 4;

// mf -> gl (uses Joe's Con alg (06/01/2022))
int span_mf_to_gl_x = 512;
int span_mf_to_gl_y = 128;
int num_p_mf_to_gl = 66177;

// int before = was; int 40 = as; int a = conservative; size (06/07/2022)
int max_num_p_mf_from_mf_to_gl = 16;

// NOTE: not a maxnump as alg is adaptive
int initial_mf_output = 14;
int max_mf_to_gl_attempts = 4;

// gl -> gr (int uses = connectCommon; (06/01/2022))
int span_gl_to_gr_x = 4;
int span_gl_to_gr_y = 4;
int num_p_gl_to_gr = 25;

int low_num_p_gl_from_gl_to_gr = 80;
int max_num_p_gr_from_gl_to_gr = 5;
int max_num_p_gl_from_gl_to_gr = 200;

int low_gl_to_gr_attempts = 20000;
int max_gl_to_gr_attempts = 50000;

// gr -> go (both pf and aa use Joe's Con Alg (06/01/2022))
int span_pf_to_go_x = 2048;
int span_pf_to_go_y = 150;
int num_p_pf_to_go = 309399;

// (was 5000) should be 4000 08/09/2022
int max_num_p_go_from_gr_to_go = 4000;
// should be 22 08/09/2022 (change to 22, no gr activity???)
int max_num_p_gr_from_gr_to_go = 50;

// one value we play around with
int max_pf_to_go_input = 3000; // (was 3750) should be 3000 08/09/2022
int max_pf_to_go_attempts = 5;

int span_aa_to_go_x = 150;
int span_aa_to_go_y = 150;
int num_p_aa_to_go = 22801;

// one value we play around with
int max_aa_to_go_input = 1000; // (was 1250) should be 1000 08/08/2022
int max_aa_to_go_attempts = 15;

// go <-> go (uses Joe's Con Alg (06/01/2022))
int span_go_to_go_x = 10;
int span_go_to_go_y = 10;
int num_p_go_to_go = 121;

int num_con_go_to_go = 12;

int go_go_recip_cons = 1;        // will be interpreted as boolean
int reduce_base_recip_go_go = 0; // will be interpreted as boolean

int max_go_to_go_attempts = 200;

// go <-> go gap junctions
int span_go_to_go_gj_x = 8;
int span_go_to_go_gj_y = 8;
int num_p_go_to_go_gj = 81;

// go -> gl (uses Joe's Con Alg (06/01/2022))
int span_go_to_gl_x = 56;
int span_go_to_gl_y = 56;
int num_p_go_to_gl = 3249;

int max_num_p_gl_from_go_to_gl = 1;
int max_num_p_go_from_go_to_gl = 64;

int max_go_to_gl_attempts = 100;

// gl -> go (uses connectCommon (06/01/2022))
// NOTE: span values taken from spanGODecDenOnGL
int span_gl_to_go_x = 12;
int span_gl_to_go_y = 12;
int num_p_gl_to_go = 169;

int low_num_p_gl_from_gl_to_go = 1;
int max_num_p_gl_from_gl_to_go = 1;
int max_num_p_go_from_gl_to_go = 16;
int initial_go_input = 1;
int low_gl_to_go_attempts = 20000;
int max_gl_to_go_attempts = 50000;

// TRANSLATIONS
// go -> gr
int max_num_p_go_from_go_to_gr = 12800;
int max_num_p_gr_from_go_to_gr = 3;

// mf -> gr
int max_num_p_gr_from_mf_to_gr = 5;
int max_num_p_mf_from_mf_to_gr = 4000;

// mf -> go
int max_num_p_go_from_mf_to_go = 16;
int max_num_p_mf_from_mf_to_go = 20;

// delay mask vars
int gr_pf_vel_in_gr_x_per_t_step = 147;
int gr_af_delay_in_t_step = 1;

// MZONE VARS
// bc -> pc
int num_p_bc_from_bc_to_pc = 4;
int num_p_pc_from_bc_to_pc = 16;

// gr -> bc
int num_p_bc_from_gr_to_bc = 8192;
int num_p_bc_from_gr_to_bc_p2 = 13;

// pc -> bc
int num_p_pc_from_pc_to_bc = 16;
int num_p_bc_from_pc_to_bc = 4;

// sc -> pc
int num_p_sc_from_sc_to_pc = 1;
int num_p_pc_from_sc_to_pc = 16;

// sc -> compart
int num_p_sc_from_sc_to_compart = 256;
int num_p_compart_from_sc_to_compart = 32;

// compart -> pc
int min_num_p_pc_from_compart_to_pc = 100;
int max_num_p_pc_from_compart_to_pc = 200;

// gr -> sc
int num_p_sc_from_gr_to_sc = 2048;
int num_p_sc_from_gr_to_sc_p2 = 11;

// pc -> nc
int num_p_pc_from_pc_to_nc = 3;
int num_p_nc_from_pc_to_nc = 12;

// gr -> pc
int num_p_pc_from_gr_to_pc = 32768;
int num_p_pc_from_gr_to_pc_p2 = 15;

// mf -> nc
int num_p_mf_from_mf_to_nc = 1;
int num_p_nc_from_mf_to_nc = 512;

// nc -> io
int num_p_nc_from_nc_to_io = 4;
int num_p_io_from_nc_to_io = 8;

// io -> pc
int num_p_io_from_io_to_pc = 8;

// io -> io
int num_p_io_in_io_to_io = 3;
int num_p_io_out_io_to_io = 3;

float msPerTimeStep = 1.0;
float numPopHistBinsPC = 8.0;
float ampl_go_to_go = 0.35;
float std_dev_go_to_go = 1.95;
float p_recip_go_go = 1.0;
float p_recip_lower_base_go_go = 0.0;
float ampl_go_to_gl = 0.01;
float std_dev_go_to_gl_ml = 100.0; // medial lateral
float std_dev_go_to_gl_s = 100.0;  // sagittal

// innet
float eLeakGO = -70.0;
float threshRestGO = -34.0;
float eLeakGR = -65.0;
float threshRestGR = -40.0;

// mzone
float eLeakSC = -60.0;
float threshRestSC = -50.0;

// NEW
float eLeakCompart = -60.0;
float compartThresh = -10.0;

float eLeakBC = -70.0;
float threshRestBC = -65.0;
float eLeakPC = -60.0;
float threshRestPC = -60.62;
float initSynWofGRtoPC = 0.5;
float eLeakIO = -60.0;
float threshRestIO = -57.4;
float eLeakNC = -65.0;
float threshRestNC = -72.0;
float initSynWofMFtoNC = 0.00085;

