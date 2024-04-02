/*
 * templatepoissoncells.h
 *
 *  Created on: Dec 13, 2012
 *      Author: consciousness
 */

#ifndef TEMPLATE_POISSONREGENCELLS_H_
#define TEMPLATE_POISSONREGENCELLS_H_

#include <algorithm> // for random_shuffle
#include <cstdint>
#include <cstdlib> // for srand and rand, sorry Wen
#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <random>

#include "activityparams.h"
#include "connectivityparams.h"
#include "kernels.h"
#include "mzone.h"
#include "sfmt.h"

class TemplatePoissonCells {
public:
  TemplatePoissonCells();
  TemplatePoissonCells(uint32_t num_gpus, uint32_t gpuIndStart, std::fstream &psth_file_buf,
                       cudaStream_t **streams);
  ~TemplatePoissonCells();

  void calcGRPoissActivity(size_t ts, cudaStream_t **streams, uint8_t streamN);
  const uint8_t *getGRAPs();
  const float **getGRFRs();
  uint32_t **get_ap_buf_gr_gpu();
  uint64_t **get_ap_hist_gr_gpu();

private:
  void init_fr_from_file(std::fstream &input_file_buf);
  void init_templates_from_psth_file(std::fstream &input_psth_file_buf);
  void initGRCUDA();
  void initCURAND(cudaStream_t **streams);

  std::normal_distribution<float> *normDist;
  std::mt19937 *noiseRandGen;
  CRandomSFMT0 *randSeedGen;
  CRandomSFMT0 **randGens;
  curandStateMRG32k3a **mrg32k3aRNGs; // device randGens

  uint32_t gpuIndStart = 0;
  uint64_t numGPUs; // = 4;

  float **grActRandNums;

  unsigned int nThreads;
  uint32_t num_in_template_ts;  // num ts collected from CbmSim as psth
  uint32_t num_out_template_ts; // num ts use to produce poisson activity here
  uint32_t num_trials;
  uint32_t isi;        // obviously must match too
  uint32_t pre_cs_ms;  // must match what we ran in cbmsim -> gives correct
                       // alignment to cs
  uint32_t post_cs_ms; // this is arbitrary
  uint32_t remaining_in_trial; // every ts that is not pre, post cs and isi

  uint64_t numGROldPerGPU;
  uint64_t numGRPerGPU;
  uint32_t calcGRActNumBlocks;
  uint32_t calcGRActNumGRPerB;

  uint64_t numGRPerRandBatch;
  uint64_t updateGRRandNumBlocks;
  uint64_t updateGRRandNumGRPerB;

  float threshBase;
  float threshMax;
  float threshIncTau;
  float threshInc;
  float sPerTS;
  uint32_t expansion_factor;
  size_t num_gr_old;

  float **gr_fr;
  float **gr_templates_h;
  // i made this var because is mildly faster :-) come back in 46 days pls and
  // thank
  float **gr_templates_t_h;
  float **gr_templates_t_d;
  size_t *gr_templates_t_pitch;

  float *threshs_h;
  uint8_t *aps_h;
  uint32_t *aps_buf_h;
  uint64_t *aps_hist_h;

  float **threshs_d;
  uint8_t **aps_d;
  uint32_t **aps_buf_d;

  uint64_t **aps_hist_d;
};

#endif /* TEMPLATE_POISSONREGENCELLS_H_ */
