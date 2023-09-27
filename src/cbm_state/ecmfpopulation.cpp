/*
 * ecmfpopulation.cpp
 *
 *  Created on: Jul 13, 2014
 *      Author: consciousness
 */

#include <random>

#include "ecmfpopulation.h"
#include "logger.h"

ECMFPopulation::ECMFPopulation(
    int numMF, int randSeed, float fracCSTMF, float fracCSPMF, float fracCtxtMF,
    float fracCollNC, float bgFreqMin, float csBGFreqMin, float ctxtFreqMin,
    float csTFreqMin, float csPFreqMin, float bgFreqMax, float csBGFreqMax,
    float ctxtFreqMax, float csTFreqMax, float csPFreqMax, bool turnOffColls,
    float fracImportMF, bool secondCS, float fracOverlap, uint32_t numZones,
    float noiseSigma) {

  /* initialize mf frequency population variables */
  CRandomSFMT0 randGen(randSeed);

  int numCSTMF;
  int numCSTMFA;
  int numCSTMFB;
  int numCSPMF;
  int numCtxtMF;
  int numCollNC;
  int numImportMF;

  this->turnOffColls = turnOffColls;
  this->numMF = numMF;

  mfFreqBG = new float[numMF];
  mfFreqInCSPhasic = new float[numMF];
  mfFreqInCSTonicA = new float[numMF];
  mfFreqInCSTonicB = new float[numMF];

  isCSTonicA = new bool[numMF];
  isCSTonicB = new bool[numMF];
  isCSPhasic = new bool[numMF];
  isContext = new bool[numMF];
  isCollateral = new bool[numMF];
  isImport = new bool[numMF];
  isAny = new bool[numMF];

  for (int i = 0; i < numMF; i++) {
    mfFreqBG[i] = randGen.Random() * (bgFreqMax - bgFreqMin) + bgFreqMin;

    mfFreqInCSTonicA[i] = mfFreqBG[i];
    mfFreqInCSTonicB[i] = mfFreqBG[i];
    mfFreqInCSPhasic[i] = mfFreqBG[i];

    isCSTonicA[i] = false;
    isCSTonicB[i] = false;
    isCSPhasic[i] = false;
    isContext[i] = false;
    isCollateral[i] = false;
    isImport[i] = false;
    isAny[i] = false;
  }

  numCSTMF = fracCSTMF * numMF;
  numCSTMFA = numCSTMF;

  if (secondCS)
    numCSTMFB = numCSTMF;
  else
    numCSTMFB = 0;

  numCSPMF = fracCSPMF * numMF;
  numCtxtMF = fracCtxtMF * numMF;
  numCollNC = fracCollNC * numMF;
  numImportMF = fracImportMF * numMF;

  // Set up Mossy Fibers. Order is important for Rand num generation.
  setMFs(numCollNC, numMF, randGen, isAny, isCollateral);
  setMFs(numImportMF, numMF, randGen, isAny, isImport);

  // Order below is important for competing stimulus experiments.
  // Pick MFs for Tonic A
  setMFs(numCSTMFA, numMF, randGen, isAny, isCSTonicA);

  // Pick remaining MFs that would have been in full Tonic B so that
  // random seed leaves off at same place
  setMFs(numCSTMFA, numMF, randGen, isAny, isAny);

  // Pick Tonic B MFs
  //  Ensure remaining MFs that would have been in full Tonic A are not picked
  setMFs(numCSTMFB, numMF, randGen, isAny, isAny);

  int phasicSum = 0;
  int contextSum = 0;
  int collateralSum = 0;
  int tonicASum = 0;
  int tonicBSum = 0;
  int importSum = 0;
  int anySum = 0;

  for (int i = 0; i < numMF; i++) {
    phasicSum += isCSPhasic[i];
    contextSum += isContext[i];
    collateralSum += isCollateral[i];
    tonicASum += isCSTonicA[i];
    tonicBSum += isCSTonicB[i];
    importSum += isImport[i];
    anySum += isAny[i]; /* wouldn't this be the whole population? */
  }

  for (int i = 0; i < numMF; i++) {
    if (isContext[i]) {
      mfFreqBG[i] =
          randGen.Random() * (ctxtFreqMax - ctxtFreqMin) + ctxtFreqMin;

      mfFreqInCSTonicA[i] = mfFreqBG[i];
      mfFreqInCSTonicB[i] = mfFreqBG[i];
      mfFreqInCSPhasic[i] = mfFreqBG[i];

      randGen.Random();
    } else if (isCSPhasic[i]) {
      mfFreqBG[i] =
          randGen.Random() * (csBGFreqMax - csBGFreqMin) + csBGFreqMin;

      mfFreqInCSTonicA[i] = mfFreqBG[i];
      mfFreqInCSTonicB[i] = mfFreqBG[i];

      mfFreqInCSPhasic[i] =
          randGen.Random() * (csPFreqMax - csPFreqMin) + csPFreqMin;
    } else if (isCollateral[i] &&
               !turnOffColls) /* confusing: is an afferent to the dcn. why treat
                                 collaterals as recurrent dcn->mf feedback? */
    {
      mfFreqBG[i] = -1;
      mfFreqInCSTonicA[i] = -1;
      mfFreqInCSTonicB[i] = -1;
      mfFreqInCSPhasic[i] = -1;
      // NOTE: that we were choosing random twice before.
      // any reason why?
      randGen.Random();
    } else if (isImport[i]) {
      mfFreqBG[i] =
          randGen.Random() * (csBGFreqMax - csBGFreqMin) + csBGFreqMin;
      mfFreqInCSTonicA[i] = -2;
      mfFreqInCSTonicB[i] = -2;
      mfFreqInCSPhasic[i] = -2;

      randGen.Random();
    } else if (isCSTonicA[i]) {
      mfFreqBG[i] =
          randGen.Random() * (csBGFreqMax - csBGFreqMin) + csBGFreqMin;

      mfFreqInCSTonicA[i] =
          randGen.Random() * (csTFreqMax - csTFreqMin) + csTFreqMin;
      mfFreqInCSPhasic[i] = mfFreqInCSTonicA[i];
    } else if (isCSTonicB[i]) {
      mfFreqBG[i] =
          randGen.Random() * (csBGFreqMax - csBGFreqMin) + csBGFreqMin;

      mfFreqInCSTonicB[i] =
          randGen.Random() * (csTFreqMax - csTFreqMin) + csTFreqMin;
      mfFreqInCSPhasic[i] = mfFreqInCSTonicB[i];
    } else {
      // NOTE: again, two random generations.
      // Note sure why.
      randGen.Random();
    }
  }

  /* initializing poisson gen vars */

  this->numZones = numZones;
  this->noiseSigma = noiseSigma;

  randSeedGen = new CRandomSFMT0(randSeed);
  randGens = new CRandomSFMT0 *[nThreads];

  for (uint32_t i = 0; i < nThreads; i++) {
    randGens[i] = new CRandomSFMT0(randSeedGen->IRandom(0, INT_MAX));
  }

  normDist = new std::normal_distribution<float>(0, this->noiseSigma);
  noiseRandGen = new std::mt19937(randSeed);

  aps = (uint8_t *)calloc(num_mf, sizeof(uint8_t));

  dnCellIndex = (uint32_t *)calloc(num_mf, sizeof(uint32_t));
  mZoneIndex = (uint32_t *)calloc(num_mf, sizeof(uint32_t));

  isTrueMF = (bool *)calloc(num_mf, sizeof(bool));
  memset(isTrueMF, true, num_mf * sizeof(bool));
  
  prepCollaterals(randSeedGen->IRandom(0, INT_MAX));
}

ECMFPopulation::~ECMFPopulation() {
  delete[] mfFreqBG;
  delete[] mfFreqInCSPhasic;
  delete[] mfFreqInCSTonicA;
  delete[] mfFreqInCSTonicB;

  delete[] isCSTonicA;
  delete[] isCSTonicB;
  delete[] isCSPhasic;
  delete[] isContext;
  delete[] isCollateral;
  delete[] isImport;
  delete[] isAny;

  delete randSeedGen;
  delete noiseRandGen;
  for (uint32_t i = 0; i < nThreads; i++) {
    delete randGens[i];
  }

  delete[] randGens;
  delete normDist;
  free(aps);
  free(isTrueMF);
  free(dnCellIndex);
  free(mZoneIndex);
}

/* public methods except constructor and destructor */
void ECMFPopulation::writeToFile(std::fstream &outfile) {
  outfile.write((char *)&numMF, sizeof(numMF));
  outfile.write((char *)mfFreqBG, numMF * sizeof(float));
  outfile.write((char *)mfFreqInCSTonicA, numMF * sizeof(float));
  outfile.write((char *)mfFreqInCSTonicB, numMF * sizeof(float));
  outfile.write((char *)mfFreqInCSPhasic, numMF * sizeof(float));
}

void ECMFPopulation::writeMFLabels(std::string labelFileName) {
  LOG_DEBUG("Writing MF labels...");
  std::fstream mflabels(labelFileName.c_str(), std::fstream::out);

  for (int i = 0; i < numMF; i++) {
    if (isContext[i]) {
      mflabels << "con ";
    } else if (isCSPhasic[i]) {
      mflabels << "pha ";
    } else if (isCollateral[i] && !turnOffColls) {
      mflabels << "col ";
    } else if (isImport[i]) {
      mflabels << "imp ";
    } else if (isCSTonicA[i] || isCSTonicB[i]) {
      mflabels << "ton ";
    } else {
      mflabels << "bac ";
    }
  }
  mflabels.close();
  LOG_DEBUG("MF labels written.");
}

float *ECMFPopulation::getBGFreq() { return mfFreqBG; }

float *ECMFPopulation::getTonicCSAFreq() { return mfFreqInCSTonicA; }

float *ECMFPopulation::getTonicCSBFreq() { return mfFreqInCSTonicB; }

float *ECMFPopulation::getPhasicCSFreq() { return mfFreqInCSPhasic; }

bool *ECMFPopulation::getTonicCSAIds() { return isCSTonicA; }

bool *ECMFPopulation::getTonicCSBIds() { return isCSTonicB; }

bool *ECMFPopulation::getPhasicCSIds() { return isCSPhasic; }

bool *ECMFPopulation::getContextIds() { return isContext; }

bool *ECMFPopulation::getCollateralIds() { return isCollateral; }

const uint8_t *ECMFPopulation::calcPoissActivity(enum mf_type type,
                                                 MZone **mZoneList,
                                                 int ispikei) {
  float *frequencies = NULL;
  int countColls = 0;
  const uint8_t *holdNCs;
  float noise;
  spikeTimer++;
  switch (type) {
  case BKGD:
    frequencies = mfFreqBG;
    break;
  case TONIC_CS_A:
    frequencies = mfFreqInCSTonicA;
    break;
  case TONIC_CS_B:
    frequencies = mfFreqInCSTonicB;
    break;
  case PHASIC_CS:
    frequencies = mfFreqInCSPhasic;
    break;
  default:
    break;
  }

  for (uint32_t i = 0; i < num_mf; i++) {
    if (frequencies[i] == -1) /* dcn mfs (or collaterals this makes no sense) */
    {
      holdNCs = mZoneList[mZoneIndex[countColls]]->exportAPNC();
      aps[i] = holdNCs[dnCellIndex[countColls]];
      countColls++;
    }
    // below is calculated w isi. why not do so for cs too?
    else if (frequencies[i] == -2)
      aps[i] =
          (spikeTimer == ispikei); /* background or import, whatever that is */
    else                           /* cs */
    {
      int tid = 0;
      if (noiseSigma == 0)
        noise = 0.0;
      else
        noise = (*normDist)((*noiseRandGen));

      aps[i] = (randGens[tid]->Random() < (frequencies[i] + noise) * sPerTS);
    }
  }
  if (spikeTimer == ispikei)
    spikeTimer = 0;
  return (const uint8_t *)aps;
}

const uint8_t *ECMFPopulation::getAPs() { return (const uint8_t *)aps; }

/* private methods */
void ECMFPopulation::setMFs(int numTypeMF, int numMF, CRandomSFMT0 &randGen,
                            bool *isAny, bool *isType) {
  for (int i = 0; i < numTypeMF; i++) {
    while (true) {
      int mfInd = randGen.IRandom(0, numMF - 1);

      if (!isAny[mfInd]) {
        isAny[mfInd] = true;
        isType[mfInd] = true;
        break;
      }
    }
  }
}

void ECMFPopulation::setMFsOverlap(int numTypeMF, int numMF,
                                   CRandomSFMT0 &randGen, bool *isAny,
                                   bool *isTypeA, bool *isTypeB,
                                   float fracOverlap) {

  // Get population sizes
  int numOverlapMF = numTypeMF * fracOverlap;
  int numIndependentMF = numTypeMF - numOverlapMF;
  LOG_DEBUG("NumOverlap: %d", numOverlapMF);

  // Select overlaping population
  int counter = 0;

  for (int i = 0; i < numMF; i++) {
    if (isTypeA[i] && counter < numOverlapMF) {
      isTypeB[i] = true;
      isAny[i] = true;
      counter++;
    }
  }
  //
  // Select non-overlaping population
  for (int i = 0; i < numIndependentMF; i++) {
    while (true) {
      int mfInd = randGen.IRandom(0, numMF - 1);

      if (isAny[mfInd])
        continue;

      isAny[mfInd] = true;
      isTypeB[mfInd] = true;
      break;
    }
  }
}

void ECMFPopulation::prepCollaterals(int rSeed) {
  uint32_t repeats = num_mf / (numZones * num_nc) + 1;
  uint32_t *tempNCs = new uint32_t[repeats * numZones * num_nc];
  uint32_t *tempMZs = new uint32_t[repeats * numZones * num_nc];

  for (uint32_t i = 0; i < repeats; i++) {
    for (uint32_t j = 0; j < numZones; j++) {
      for (uint32_t k = 0; k < num_nc; k++) {
        tempNCs[k + num_nc * j + num_nc * numZones * i] = k;
        tempMZs[k + num_nc * j + num_nc * numZones * i] = j;
      }
    }
  }
  std::srand(rSeed);
  std::random_shuffle(tempNCs, tempNCs + repeats * numZones * num_nc);
  std::srand(rSeed);
  std::random_shuffle(tempMZs, tempMZs + repeats * numZones * num_nc);
  std::copy(tempNCs, tempNCs + num_mf, dnCellIndex);
  std::copy(tempMZs, tempMZs + num_mf, mZoneIndex);

  delete[] tempNCs;
  delete[] tempMZs;
}
