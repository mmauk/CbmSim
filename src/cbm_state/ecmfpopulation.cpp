/*
 * ecmfpopulation.cpp
 *
 *  Created on: Jul 13, 2014
 *      Author: consciousness
 */

#include <random>

#include "ecmfpopulation.h"
#include "logger.h"

ECMFPopulation::ECMFPopulation(int randSeed, float fracCS, float fracColl,
                               float bgFreqMin, float csFreqMin,
                               float bgFreqMax, float csFreqMax,
                               bool turnOnColls, uint32_t numZones,
                               float noiseSigma) {

  /* initialize mf frequency population variables */
  CRandomSFMT0 randGen(randSeed);

  int numCS = fracCS * num_mf;
  int numColl = fracColl * num_mf;

  mfFreqBG = new float[num_mf]();
  mfFreqCS = new float[num_mf]();
  mfThresh = new float[num_mf]();

  isCS = new bool[num_mf]();
  isColl = new bool[num_mf]();
  isAny = new bool[num_mf]();

  // Pick MFs for CS
  setMFs(numCS, num_mf, randGen, isAny, isCS);

  // Set the collaterals
  if (turnOnColls)
    setMFs(numColl, num_mf, randGen, isAny, isColl);


  int collSum = 0;
  int csSum = 0;
  int anySum = 0;

  for (int i = 0; i < num_mf; i++) {
    collSum += isColl[i];
    csSum += isCS[i];
    anySum += isAny[i];
  }

  // float adjustFactor = 1.2;
  for (int i = 0; i < num_mf; i++) {
    if (isColl[i]) {
      mfFreqBG[i] = -1;
      mfFreqCS[i] = -1;
    } else {
      mfFreqBG[i] = randGen.Random() * (bgFreqMax - bgFreqMin) + bgFreqMin;
      mfFreqBG[i] *= sPerTS * kappa;
      // mfFreqBG[i] -= (mfFreqBG[i] * (100.0 - mfFreqBG[i])) / 55000;
      if (isCS[i]) {
        mfFreqCS[i] = randGen.Random() * (csFreqMax - csFreqMin) + csFreqMin;
        mfFreqCS[i] *= sPerTS * kappa;
        // mfFreqCS[i] -= (mfFreqCS[i] * (100.0 - mfFreqCS[i])) / 55000;
      } else
        mfFreqCS[i] = mfFreqBG[i];
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
  apBufs = (uint32_t *)calloc(num_mf, sizeof(uint32_t));

  dnCellIndex = (uint32_t *)calloc(num_mf, sizeof(uint32_t));
  mZoneIndex = (uint32_t *)calloc(num_mf, sizeof(uint32_t));

  isTrueMF = (bool *)calloc(num_mf, sizeof(bool));
  memset(isTrueMF, true, num_mf * sizeof(bool));
  
  prepCollaterals(randSeedGen->IRandom(0, INT_MAX));
}

ECMFPopulation::~ECMFPopulation() {
  delete[] mfFreqBG;
  delete[] mfFreqCS;
  delete[] mfThresh;

  delete[] isCS;
  delete[] isColl;
  delete[] isAny;

  delete randSeedGen;
  delete noiseRandGen;
  for (uint32_t i = 0; i < nThreads; i++) {
    delete randGens[i];
  }

  delete[] randGens;
  delete normDist;
  free(aps);
  free(apBufs);
  free(isTrueMF);
  free(dnCellIndex);
  free(mZoneIndex);
}

/* public methods except constructor and destructor */
void ECMFPopulation::writeToFile(std::fstream &outfile) {
  outfile.write((char *)mfFreqBG, num_mf * sizeof(float));
  outfile.write((char *)mfFreqCS, num_mf * sizeof(float));
  outfile.write((char *)isCS, num_mf * sizeof(bool));
  outfile.write((char *)isColl, num_mf * sizeof(bool));
}

void ECMFPopulation::writeMFLabels(std::string labelFileName) {
  LOG_DEBUG("Writing MF labels...");
  std::fstream mflabels(labelFileName.c_str(), std::fstream::out);

  for (int i = 0; i < num_mf; i++) {
    if (isColl[i]) {
      mflabels << "col ";
    } else if (isCS) {
      mflabels << "ton ";
    } else {
      mflabels << "bac ";
    }
  }
  mflabels.close();
  LOG_DEBUG("MF labels written.");
}

const float *ECMFPopulation::getBGFreq() { return mfFreqBG; }

const float *ECMFPopulation::getCSFreq() { return mfFreqCS; }

const bool *ECMFPopulation::getCSIds() { return isCS; }

const bool *ECMFPopulation::getCollIds() { return isColl; }

void ECMFPopulation::calcPoissActivity(enum mf_type type, MZone **mZoneList) {
  float *frequencies = (type == CS) ? mfFreqCS : mfFreqBG;
  int countColls = 0;
  float noise = 0.0;
  int tid = 0;
  uint8_t reset_ap_buf = 0;
  for (size_t i = 0; i < num_mf; i++) {
    if (frequencies[i] == -1) {
      const uint8_t *holdNCs = mZoneList[mZoneIndex[countColls]]->exportAPNC();
      aps[i] = holdNCs[dnCellIndex[countColls++]];
    } else {
      if (noiseSigma != 0)
        noise = (*normDist)((*noiseRandGen));

      // frequencies is really the prob of firing (see constructor)
      aps[i] =
          randGens[tid]->Random() < frequencies[i] + noise; // - mfThresh[i];
      // mfThresh[i] = 2.0 * aps[i] * frequencies[i] +
      //               (1 - aps[i]) * mfThresh[i] * threshDecMF;
      // change apBuf only if we spike!
      apBufs[i] =
          aps[i] * ((apBufs[i] << 1) | aps[i]) + (1 - aps[i]) * apBufs[i];
      // only keep spikes if we reset apBuf
      reset_ap_buf = (apBufs[i] & num_spike_mask) == num_spike_mask;
      aps[i] = reset_ap_buf;
      // reset apBuf
      apBufs[i] = (1 - reset_ap_buf) * apBufs[i];
    }
  }
}

const uint8_t *ECMFPopulation::getAPs() { return (const uint8_t *)aps; }

/* private methods */
void ECMFPopulation::setMFs(int numTypeMF, int num_mf, CRandomSFMT0 &randGen,
                            bool *isAny, bool *isType) {
  for (int i = 0; i < numTypeMF; i++) {
    while (true) {
      int mfInd = randGen.IRandom(0, num_mf - 1);

      if (!isAny[mfInd]) {
        isAny[mfInd] = true;
        isType[mfInd] = true;
        break;
      }
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
