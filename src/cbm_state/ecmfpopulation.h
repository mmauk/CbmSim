/*
 * ecmfpopulation.h
 *
 *  Created on: Jul 11, 2014
 *      Author: consciousness
 */

#ifndef ECMFPOPULATION_H_
#define ECMFPOPULATION_H_

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "activityparams.h"
#include "connectivityparams.h"
#include "mzone.h"
#include "randomc.h"
#include "sfmt.h"

enum mf_type { BKGD, TONIC_CS_A, TONIC_CS_B, PHASIC_CS };

class ECMFPopulation {
public:
  ECMFPopulation(int numMF, int randSeed, float fracCSTMF, float fracCSPMF,
                 float fracCtxtMF, float fracCollNC, float bgFreqMin,
                 float csBGFreqMin, float ctxtFreqMin, float csTFreqMin,
                 float csPFreqMin, float bgFreqMax, float csBGFreqMax,
                 float ctxtFreqMax, float csTFreqMax, float csPFreqMax,
                 bool turnOffColls, float fracImportCells, bool secondCS,
                 float fracOverlap, uint32_t numZones, float noiseSigma = 0);

  ~ECMFPopulation();

  void writeToFile(std::fstream &outfile);
  void writeMFLabels(std::string labelFileName);

  float *getBGFreq();
  float *getTonicCSAFreq();
  float *getTonicCSBFreq();
  float *getPhasicCSFreq();

  bool *getTonicCSAIds();
  bool *getTonicCSBIds();
  bool *getPhasicCSIds();
  bool *getContextIds();
  bool *getCollateralIds();

  const uint8_t *calcPoissActivity(enum mf_type type, MZone **mZoneList,
                                   int ispikei = 18);
  const uint8_t *getAPs();

private:
  /* frequency pop functions */
  void setMFs(int numTypeMF, int numMF, CRandomSFMT0 &randGen, bool *isAny,
              bool *isType);
  void setMFsOverlap(int numTypeMF, int numMF, CRandomSFMT0 &randGen,
                     bool *isAny, bool *isTypeA, bool *isTypeB,
                     float fracOverlap);

  /* poisson gen functions */
  void prepCollaterals(int rSeed);
  /* frequency population variables */
  uint32_t numMF;

  float *mfFreqBG;
  float *mfFreqInCSTonicA;
  float *mfFreqInCSTonicB;
  float *mfFreqInCSPhasic;

  bool *isCSTonicA;
  bool *isCSTonicB;
  bool *isCSPhasic;
  bool *isContext;
  bool *isCollateral;
  bool *isImport;
  bool *isAny;
  bool turnOffColls;

  /* poisson spike generator vars */
  uint32_t nThreads = 1;   // hard-coded
  uint32_t spikeTimer = 0; // initialize within class def
  uint32_t numZones;

  float sPerTS = msPerTimeStep / 1000;
  float noiseSigma;

  CRandomSFMT0 *randSeedGen; // the uber seed
  CRandomSFMT0 **randGens;   // diff randgens per thread, if use openmp
  std::normal_distribution<float> *normDist; // for noise
  std::mt19937 *noiseRandGen;                // for noise

  uint8_t *aps;

  uint32_t *dnCellIndex;
  uint32_t *mZoneIndex;

  bool *isTrueMF;
};

#endif /* ECMFPOPULATION_H_ */
