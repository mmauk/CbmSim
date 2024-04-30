#ifndef _RED_NUCLEUS_H
#define _RED_NUCLEUS_H
#endif /* _RED_NUCLEUS_H */

#include <cstdint>
#include <stdlib.h>
#include <string.h>

#include "connectivityparams.h"
#include "file_parse.h"
#include "logger.h"

class RedNucleus {
public:
  RedNucleus(uint32_t num_inputs);
  ~RedNucleus();

  float calc_step(const uint8_t *in_aps);
  void calc_crs_from(const uint8_t **nc_rasters, float **crs,
                     uint32_t num_trials, uint32_t bun_viz_ts_per_trial,
                     uint32_t bun_viz_ms_pre_cs, uint32_t sim_pre_cs,
                     uint32_t sim_ts_per_trial);
  void reset();

private:
  float g_leak;
  float g_e_tau;
  float g_e_decay;
  float e_leak;
  float v_m;
  float thresh;
  uint32_t num_inputs;
  float *g_e;
};

RedNucleus::RedNucleus(uint32_t num_inputs) {
  g_leak = 0.025 / (6.0 - msPerTimeStep);
  g_e_tau = 15.0;
  g_e_decay = exp(-msPerTimeStep / g_e_tau);
  e_leak = 0.0;
  v_m = e_leak;
  thresh = 0.02;
  this->num_inputs = num_inputs;
  g_e = (float *)calloc(num_inputs, sizeof(float));
}

RedNucleus::~RedNucleus() { free(g_e); }

float RedNucleus::calc_step(const uint8_t *in_aps) {
  float g_e_tot = 0;
  for (uint32_t i = 0; i < num_inputs; i++) {
    g_e[i] *= g_e_decay;
    g_e[i] += in_aps[i] * 0.012;
    g_e_tot += g_e[i];
  }
  g_e_tot -= 0.05;
  g_e_tot = (g_e_tot < 0.0) ? 0.0 : g_e_tot;
  g_e_tot = 5.0 * pow(g_e_tot, 3);
  g_e_tot = (g_e_tot < thresh) ? 0.0 : g_e_tot;
  v_m += -g_leak * v_m + g_e_tot * (80 - v_m);
  return v_m;
}

void RedNucleus::reset() {
  e_leak = 0.0;
  v_m = e_leak;
  memset((char *)g_e, 0, num_inputs * sizeof(float));
}

void RedNucleus::calc_crs_from(const uint8_t **nc_rasters, float **crs,
                               uint32_t num_trials,
                               uint32_t bun_viz_ts_per_trial,
                               uint32_t bun_viz_ms_pre_cs, uint32_t sim_pre_cs,
                               uint32_t sim_ts_per_trial) {

  uint32_t ts_start = sim_pre_cs - bun_viz_ms_pre_cs;
  for (size_t i = 0; i < num_trials; i++) {
    for (size_t j = ts_start; j < sim_ts_per_trial; j++) {
      crs[i][j - ts_start] = calc_step(nc_rasters[i * sim_ts_per_trial + j]);
    }
    reset();
  }
  norm_2d_array_ip<float>(crs, num_trials, bun_viz_ts_per_trial);
  mult_2d_array_by_val_ip<float>(crs, num_trials, bun_viz_ts_per_trial, 6.0);
}
