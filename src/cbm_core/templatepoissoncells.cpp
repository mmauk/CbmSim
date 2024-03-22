/*
 * templatepoissonregencells.cpp
 *
 *  Created on: Dec 13, 2012
 *      Author: consciousness
 *  Modified on: Jul 22, 2015
 *  	Author: evandelord
 */
#include <cstring>
#include <omp.h>

#include "array_util.h"
#include "dynamic2darray.h"
#include "logger.h"
#include "templatepoissoncells.h"

TemplatePoissonCells::TemplatePoissonCells(uint32_t num_gpus,
                                           std::fstream &psth_file_buf,
                                           cudaStream_t **streams) {
  // make this time(NULL) or something for random runs
  randSeedGen = new CRandomSFMT0(0);

  numGPUs = num_gpus;
  nThreads = 1;
  // bad: should read this in from file so you do not have
  // to change in two places for every isi.
  num_in_template_ts = 5000;
  num_trials = 720 * 28; // 20000;
  isi = 500;             // obviously must match too
  pre_cs_ms =
      2000; // must match what we ran in cbmsim -> gives correct alignment to cs
  post_cs_ms = 400; // this is arbitrary
  remaining_in_trial = num_in_template_ts - pre_cs_ms - isi - post_cs_ms;
  num_out_template_ts =
      isi + post_cs_ms + 2; // 1 for pre-cs background, 1 for after post_cs_ms

  randGens = new CRandomSFMT0 *[nThreads];

  for (unsigned int i = 0; i < nThreads; i++) {
    randGens[i] = new CRandomSFMT0(randSeedGen->IRandom(0, INT_MAX));
  }

  // threshes need to be scaled
  threshBase = 0;
  threshMax = 1;
  threshIncTau = 5; // hard code in the decay time constant
  threshInc =
      1 - exp(-1.0 / threshIncTau); // for now, hard code in time-bin size
  sPerTS = msPerTimeStep / 1000.0;

  expansion_factor =
      1; // by what factor are we expanding the granule cell number?
  num_gr_old = num_gr / expansion_factor;

  threshs_h = (float *)calloc(num_gr, sizeof(float));
  aps_h = (uint8_t *)calloc(num_gr, sizeof(uint8_t));
  aps_buf_h = (uint32_t *)calloc(num_gr, sizeof(uint32_t));
  aps_hist_h = (uint64_t *)calloc(num_gr, sizeof(uint64_t));

  for (size_t i = 0; i < num_gr; i++)
    threshs_h[i] = threshMax;

  gr_templates_h = allocate2DArray<float>(
      num_gr_old, num_out_template_ts); // for now, only have firing rates from
                                        // trials of num_template_ts
  init_templates_from_psth_file(psth_file_buf);

  initGRCUDA();
  initCURAND(streams);
}

TemplatePoissonCells::~TemplatePoissonCells() {
  delete randSeedGen;
  for (uint32_t i = 0; i < nThreads; i++) {
    delete randGens[i];
  }
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaFree(mrg32k3aRNGs[i]);
    cudaFree(grActRandNums[i]);
    cudaDeviceSynchronize();
  }
  delete[] mrg32k3aRNGs;
  delete[] grActRandNums;

  delete[] randGens;

  free(threshs_h);
  free(aps_h);
  free(aps_buf_h);
  free(aps_hist_h);
  delete2DArray(gr_templates_h);
  delete2DArray(gr_templates_t_h);

  for (uint32_t i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaFree(gr_templates_t_d[i]);
    cudaFree(threshs_d[i]);
    cudaFree(aps_d[i]);
    cudaFree(aps_buf_d[i]);
    cudaFree(aps_hist_d[i]);
    cudaDeviceSynchronize();
  }
  free(gr_templates_t_d);
  free(gr_templates_t_pitch);

  free(threshs_d);
  free(aps_d);
  free(aps_buf_d);
  free(aps_hist_d);
}

// Soon to be deprecated: was for loading in pre-computed smoothed-fr
// void TemplatePoissonCells::init_fr_from_file(std::fstream &input_file_buf)
//{
//	input_file_buf.read((char *)gr_fr[0], num_gr * num_template_ts *
// sizeof(float));
//}

void TemplatePoissonCells::init_templates_from_psth_file(
    std::fstream &input_psth_file_buf) {
  enum { ZERO_FIRERS, LOW_FIRERS, HIGH_FIRERS };
  uint32_t num_fire_categories[3] = {0};
  // expect input data comes from num_trials trials, num_template_ts ts a piece.
  const float THRESH_FR = 10.0; // in Hz
  const uint32_t THRESH_COUNT =
      (THRESH_FR / 1000.0) * num_in_template_ts * num_trials;
  uint32_t **input_psths =
      allocate2DArray<uint32_t>(num_in_template_ts, num_gr_old);
  uint8_t *firing_categories = (uint8_t *)calloc(num_gr_old, sizeof(uint8_t));
  uint32_t *cell_spike_sums = (uint32_t *)calloc(num_gr_old, sizeof(uint32_t));
  LOG_INFO("loading cells from file.");
  input_psth_file_buf.read((char *)input_psths[0],
                           num_in_template_ts * num_gr_old * sizeof(uint32_t));
  LOG_INFO("finished load.");
  // LOG_INFO("transposing...");
  //  uint32_t **input_psths_t =
  //      transpose2DArray(input_psths, num_in_template_ts, num_gr_old);
  // LOG_INFO("finished transposing.");

  // determine firing rate categories
  LOG_INFO("determining firing categories...");
  for (size_t i = 0; i < num_in_template_ts; i++) {
    for (size_t j = 0; j < num_gr_old; j++) {
      cell_spike_sums[j] +=
          input_psths[i][j]; // add in the current time-steps accumulated spikes
    }
  }

  for (size_t i = 0; i < num_gr_old; i++) {
    if (cell_spike_sums[i] > THRESH_COUNT) {
      firing_categories[i] = HIGH_FIRERS;
      num_fire_categories[HIGH_FIRERS]++;
    } else if (cell_spike_sums[i] < THRESH_COUNT && cell_spike_sums[i] > 0) {
      firing_categories[i] = LOW_FIRERS;
      num_fire_categories[LOW_FIRERS]++;
    } else {
      firing_categories[i] = ZERO_FIRERS;
      num_fire_categories[ZERO_FIRERS]++;
    }
  }
  LOG_INFO("finished determining firing categories.");
  LOG_INFO("num high: %u", num_fire_categories[HIGH_FIRERS]);
  LOG_INFO("num low: %u", num_fire_categories[LOW_FIRERS]);
  LOG_INFO("num zero: %u", num_fire_categories[ZERO_FIRERS]);

  // create template pdfs
  LOG_INFO("creating templates...");
  for (size_t i = 0; i < num_gr_old; i++) {
    // simplest model: every gr gets its mean fr
    float mean_rate = ((float)cell_spike_sums[i] * 1000.0) /
                      (num_in_template_ts * num_trials);
    for (size_t j = 0; j < num_out_template_ts; j++) {
      gr_templates_h[i][j] = mean_rate; // just set every bin to same avg fr
    }
    // //NOTE: no need to include zero_firers as calloc sets their values to
    // zero
    // if (firing_categories[i] == LOW_FIRERS)
    //{
    //	float mean_rate = ((float)cell_spike_sums[i] * 1000.0) /
    //(num_in_template_ts * num_trials); 	for (size_t j = 0; j <
    // num_out_template_ts; j++)
    //	{
    //		gr_templates_h[i][j] = mean_rate; // just set every bin to same
    // avg fr
    //	}
    //}
    // else // high firers --> this is where we apply the transformation
    //{
    //	uint32_t ts_out = 0;
    //	uint32_t pre_cs_sum = 0;
    //	uint32_t post_cs_sum = 0;
    //	for (size_t j = 0; j < num_in_template_ts; j++)
    //	{
    //		if (j < pre_cs_ms) {
    //			pre_cs_sum += input_psths_t[i][j];
    //		} else if (j == pre_cs_ms) {
    //			gr_templates_h[i][ts_out] = ((float)pre_cs_sum * 1000.0)
    /// (pre_cs_ms * num_trials); 			ts_out++;
    //		}

    //		if (j >= pre_cs_ms && j < pre_cs_ms + isi + post_cs_ms) {
    //			// here multiply by 1000 to convert to seconds, then
    // divide by num trials 			gr_templates_h[i][ts_out] =
    //((float)input_psths_t[i][j] * 1000.0) / (float)num_trials;
    // ts_out++; 		} else if (j >= pre_cs_ms + isi + post_cs_ms) {
    // post_cs_sum += input_psths_t[i][j];
    //		}
    //	}
    //	gr_templates_h[i][ts_out] = ((float)post_cs_sum * 1000.0) /
    //(remaining_in_trial * num_trials);
    //}
  }
  LOG_INFO("finished creating templates...");

  LOG_INFO("transposing templates...");
  gr_templates_t_h =
      transpose2DArray(gr_templates_h, num_gr_old, num_out_template_ts);
  LOG_INFO("finished transposing templates.");

  delete2DArray(input_psths);
  // delete2DArray(input_psths_t);
  free(firing_categories);
  free(cell_spike_sums);
}

void TemplatePoissonCells::initGRCUDA() {
  numGROldPerGPU = num_gr_old / numGPUs;
  numGRPerGPU = num_gr / numGPUs;

  calcGRActNumGRPerB = 512; // 1024 ie max num threads per block
  calcGRActNumBlocks = numGRPerGPU / calcGRActNumGRPerB; // gives 2 ^ 17

  gr_templates_t_d = (float **)calloc(numGPUs, sizeof(float *));
  gr_templates_t_pitch = (size_t *)calloc(numGPUs, sizeof(size_t));

  threshs_d = (float **)calloc(numGPUs, sizeof(float *));
  aps_d = (uint8_t **)calloc(numGPUs, sizeof(uint8_t *));
  aps_buf_d = (uint32_t **)calloc(numGPUs, sizeof(uint32_t *));
  aps_hist_d = (uint64_t **)calloc(numGPUs, sizeof(uint64_t *));

  LOG_INFO("Allocating device memory...");
  for (uint32_t i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMallocPitch(&gr_templates_t_d[i], &gr_templates_t_pitch[i],
                    numGROldPerGPU * sizeof(float), num_out_template_ts);
    LOG_INFO("gr templates malloc pitch error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&threshs_d[i], numGRPerGPU * sizeof(float));
    LOG_INFO("threshs malloc error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&aps_d[i], numGRPerGPU * sizeof(uint8_t));
    LOG_INFO("aps malloc error: %s", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&aps_buf_d[i], numGRPerGPU * sizeof(uint32_t));
    LOG_INFO("aps buf malloc error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&aps_hist_d[i], numGRPerGPU * sizeof(uint64_t));
    LOG_INFO("aps hist malloc error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
  }
  LOG_INFO("alloc final error: %s", cudaGetErrorString(cudaGetLastError()));
  LOG_INFO("Finished allocating device memory.");

  LOG_INFO("Copying host to device memory...");

  uint64_t grOldCpySize = numGROldPerGPU;
  uint64_t grCpySize = numGRPerGPU;
  uint64_t grOldCpyStartInd;
  uint64_t grCpyStartInd;
  for (uint32_t i = 0; i < numGPUs; i++) {
    grOldCpyStartInd = i * numGROldPerGPU;
    grCpyStartInd = i * numGRPerGPU;
    cudaSetDevice(i + gpuIndStart);

    for (size_t j = 0; j < num_out_template_ts; j++) {
      cudaMemcpy((char *)gr_templates_t_d[i] + j * gr_templates_t_pitch[i],
                 &gr_templates_t_h[j][grOldCpyStartInd],
                 grOldCpySize * sizeof(float), cudaMemcpyHostToDevice);
    }
    LOG_INFO("gr templates memcpy2D error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(threshs_d[i], &threshs_h[grCpyStartInd],
               grCpySize * sizeof(float), cudaMemcpyHostToDevice);
    LOG_INFO("threshs memcpy error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(aps_d[i], &aps_h[grCpyStartInd], grCpySize * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    LOG_INFO("aps memcpy error: %s", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(aps_buf_d[i], &aps_buf_h[grCpyStartInd],
               grCpySize * sizeof(uint32_t), cudaMemcpyHostToDevice);
    LOG_INFO("aps buf memcpy error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(aps_hist_d[i], &aps_hist_h[grCpyStartInd],
               grCpySize * sizeof(uint64_t), cudaMemcpyHostToDevice);
    LOG_INFO("aps hist memcpy error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
  }
  LOG_INFO("memcpy final error: %s", cudaGetErrorString(cudaGetLastError()));
  LOG_INFO("Finished copying host to device memory.");
}

void TemplatePoissonCells::initCURAND(cudaStream_t **streams) {
  // set up rng
  LOG_INFO("Initializing curand state...");

  CRandomSFMT cudaRNGSeedGen(time(0));

  int32_t curandInitSeed = cudaRNGSeedGen.IRandom(0, INT_MAX);

  mrg32k3aRNGs = new curandStateMRG32k3a *[numGPUs];
  grActRandNums = new float *[numGPUs];

  // we're just playing with some numbers rn
  numGRPerRandBatch = 1048576;
  updateGRRandNumGRPerB = 512;
  updateGRRandNumBlocks = numGRPerRandBatch / updateGRRandNumGRPerB;
  dim3 updateGRRandGridDim(updateGRRandNumBlocks);
  dim3 updateGRRandBlockDim(updateGRRandNumGRPerB);

  for (uint8_t i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMalloc(&mrg32k3aRNGs[i], updateGRRandNumGRPerB * updateGRRandNumBlocks *
                                     sizeof(curandStateMRG32k3a));
    LOG_INFO("rng malloc error: %s", cudaGetErrorString(cudaGetLastError()));
    // adding i to init seed so that we get different nums across gpus
    callCurandSetupKernel<curandStateMRG32k3a, dim3, dim3>(
        streams[i][i], mrg32k3aRNGs[i],
        (curandInitSeed + (uint32_t)i) % INT_MAX, updateGRRandGridDim,
        updateGRRandBlockDim);
    LOG_INFO("curand setup error: %s", cudaGetErrorString(cudaGetLastError()));
    cudaMalloc(&grActRandNums[i], numGRPerGPU * sizeof(float));
    LOG_INFO("rand num malloc error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaMemset(grActRandNums[i], 0, numGRPerGPU * sizeof(float));
    LOG_INFO("rand num memset error: %s",
             cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
  }
  LOG_INFO("Finished initializing curand state.");
  LOG_INFO("Last error: %s", cudaGetErrorString(cudaGetLastError()));
}

void TemplatePoissonCells::calcGRPoissActivity(size_t ts,
                                               cudaStream_t **streams,
                                               uint8_t streamN) {
  uint32_t apBufGRHistMask = (1 << (int)tsPerHistBinGR) - 1;

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    // generate the random numbers in batches
    for (int j = 0; j < numGRPerGPU; j += numGRPerRandBatch) {
      callCurandGenerateUniformKernel<curandStateMRG32k3a>(
          streams[i][3], mrg32k3aRNGs[i], updateGRRandNumBlocks,
          updateGRRandNumGRPerB, grActRandNums[i], j);
    }

    callGRPoissActKernel(
        streams[i][streamN], calcGRActNumBlocks, calcGRActNumGRPerB,
        threshs_d[i], aps_d[i], aps_buf_d[i], aps_hist_d[i], grActRandNums[i],
        gr_templates_t_d[i], gr_templates_t_pitch[i], numGROldPerGPU, ts,
        sPerTS, pre_cs_ms, isi, num_out_template_ts, apBufGRHistMask,
        threshBase, threshMax, threshInc);
  }
}

const uint8_t *TemplatePoissonCells::getGRAPs() {
  for (uint32_t i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMemcpy(&aps_h[i * numGRPerGPU], aps_d[i], numGRPerGPU * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);
  }
  return (const uint8_t *)aps_h;
}

const float **TemplatePoissonCells::getGRFRs() { return (const float **)gr_fr; }
uint32_t **TemplatePoissonCells::get_ap_buf_gr_gpu() { return aps_buf_d; }
uint64_t **TemplatePoissonCells::get_ap_hist_gr_gpu() { return aps_hist_d; }
