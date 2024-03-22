/*
 * cbmsimcore.h
 *
 *  Created on: Dec 14, 2011
 *      Author: consciousness
 *
 * This file pulls together innet and mzone classes and performs the
 * heavy-lifting of calculating cell spiking activity every time step
 */

#ifndef CBMSIMCORE_H_
#define CBMSIMCORE_H_

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <string>
#include <time.h>
#include <vector>

#include "cbmstate.h"
#include "innet.h"
#include "mzone.h"
#include "sfmt.h"
#include "templatepoissoncells.h"

enum plasticity { OFF, GRADED, BINARY, CASCADE };

class CBMSimCore {
public:
  CBMSimCore();
  CBMSimCore(CBMState *state, std::string in_gr_psth_filename,
             int gpuIndStart = -1, int numGPUP2 = -1);
  ~CBMSimCore();

  void calcActivityGRPoiss(enum plasticity pf_pc_plast, uint32_t ts);
  void calcActivity(float spillFrac, enum plasticity pf_pc_plast,
                    enum plasticity mf_nc_plast, bool use_weight_mask = false);
  void updateMFInput(const uint8_t *mfIn);
  void setTrueMFs(bool *isCollateralMF);
  void updateGRStim(int startGRStim, int numGRStim);
  void updateErrDrive(unsigned int zoneN, float errDriveRelative);

  void writeToState(bool use_gr_act_from_poiss = false);
  void writeState(std::fstream &outfile, bool use_gr_act_from_poiss = false);

  TemplatePoissonCells *getPoissGrs();
  InNet *getInputNet();
  MZone **getMZoneList();

protected:
  void initCUDAStreams();
  void initAuxVars();

  void syncCUDA(std::string title);

  TemplatePoissonCells *grs = NULL;
  CBMState *simState = NULL;

  uint32_t numZones;

  InNet *inputNet = NULL;
  MZone **zones = NULL;

  cudaStream_t **streams;
  int gpuIndStart;
  int numGPUs;

private:
  bool isGRStim = false;
  int numGRStim = 0;
  int startGRStim = 0;

  uint32_t curTime;

  void construct(CBMState *state, int *mzoneRSeed, int gpuIndStart,
                 int numGPUP2, std::string in_gr_psth_filename);
};

#endif /* CBMSIMCORE_H_ */
