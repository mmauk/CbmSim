#ifndef _ANALYSIS_H
#define _ANALYSIS_H
#endif /* _ANALYSIS_H */

#include <cstdint>

#include "file_parse.h"
#include "logger.h"

namespace Analysis {
/**
 *  @brief compute crs for all trials
 */
void calculate_pc_crs(bool pc_crs_initialized, uint32_t ms_pre_cs,
                      uint32_t ms_measure, trials_data &td, float **pc_crs,
                      float **pc_smooth_frs);
} // namespace Analysis
