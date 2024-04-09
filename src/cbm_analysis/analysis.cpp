#include <algorithm> /* std::min_element */
#include <numeric>   /* std::accumulate */

#include "analysis.h"

void Analysis::calculate_pc_crs(bool pc_crs_initialized, uint32_t ms_pre_cs,
                                uint32_t ms_measure, trials_data &td,
                                float **pc_crs, float **pc_smooth_frs) {
  if (!pc_crs_initialized) {
    LOG_FATAL("Attempting to compute crs from uninitialized pc rasters.");
    exit(1);
  }
  uint32_t base_interval_low = 0;
  // todo: compute smoothed firing rates from rasters
  uint32_t base_interval_high = ms_pre_cs;
  for (uint32_t trial = 0; trial < td.num_trials; trial++) {
    float response_onset =
        0.8 * std::accumulate(pc_crs[trial] + base_interval_low,
                              pc_crs[trial] + base_interval_high, 0.0);
    float amp_est =
        response_onset -
        *std::min_element(pc_crs[trial] + ms_pre_cs,
                          pc_crs[trial] + ms_pre_cs + td.cs_lens[trial]);
  }
}
