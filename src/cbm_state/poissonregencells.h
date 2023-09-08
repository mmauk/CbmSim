/*
 * poissonregencells.h
 *
 *  Created on: Dec 13, 2012
 *      Author: consciousness
 */

#ifndef POISSONREGENCELLS_H_
#define POISSONREGENCELLS_H_

#include <algorithm> // for random_shuffle
#include <cstdlib>   // for srand and rand, sorry Wen
#include <iostream>
#include <limits.h>
#include <math.h>
#include <random>

#include "activityparams.h"
#include "connectivityparams.h"
#include "mzone.h"
#include "sfmt.h"
#include <cstdint>

class PoissonRegenCells {
public:
  PoissonRegenCells();
  PoissonRegenCells(int randSeed, float threshDecayTau, unsigned int numZones,
                    float sigma = 0);
  ~PoissonRegenCells();

  const uint8_t *calcPoissActivity(const float *freqencies, MZone **mZoneList,
                                   int ispikei = 18);
  bool *calcTrueMFs(const float *freqencies);
  const uint8_t *getAPs();

private:
  void prepCollaterals(int rSeed);

  std::normal_distribution<float> *normDist;
  std::mt19937 *noiseRandGen;
  CRandomSFMT0 *randSeedGen;
  CRandomSFMT0 **randGens;

  unsigned int nThreads;

  unsigned int numZones;
  float sPerTS;
  float sigma;

  uint8_t *aps;
  uint32_t *dnCellIndex;
  uint32_t *mZoneIndex;
  int spikeTimer = 0;
  bool *isTrueMF;
};

#endif /* POISSONREGENCELLS_H_ */
