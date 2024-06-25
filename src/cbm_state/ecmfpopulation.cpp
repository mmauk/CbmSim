/*
 * ecmfpopulation.cpp
 *
 *  Created on: Jul 13, 2014
 *      Author: consciousness
 */

#include <random>

#include "ecmfpopulation.h"
#include "file_utility.h"
#include "logger.h"

ECMFPopulation::ECMFPopulation() {

  /* initialize mf frequency population variables */
  CRandomSFMT0 randGen(randSeed);
  LOG_DEBUG("Allocating mf memory...");
  allocateMemory();
  LOG_DEBUG("Finished allocating mf memory.");

  int numCS = fracCS * num_mf;
  int numColl = fracColl * num_mf;

  LOG_DEBUG("Setting CS Mfs...");
  // Pick MFs for CS
  setMFs(numCS, num_mf, randGen, isAny, isCS);
  LOG_DEBUG("Finished setting CS Mfs...");

  // Set the collaterals
  if (turnOnColls) {
    LOG_DEBUG("Setting Collateral Mfs...");
    setMFs(numColl, num_mf, randGen, isAny, isColl);
    LOG_DEBUG("Finished setting Collateral Mfs.");
  }
  LOG_DEBUG("Setting Mf frequencies...");
  for (int i = 0; i < num_mf; i++) {
    if (isColl[i]) {
      mfFreqBG[i] = -1;
      mfFreqCS[i] = -1;
    } else {
      mfFreqBG[i] = randGen.Random() * (bgFreqMax - bgFreqMin) + bgFreqMin;
      mfFreqBG[i] *= sPerTS * kappa;
      if (isCS[i]) {
        mfFreqCS[i] = randGen.Random() * (csFreqMax - csFreqMin) + csFreqMin;
        mfFreqCS[i] *= sPerTS * kappa;
      } else
        mfFreqCS[i] = mfFreqBG[i];
    }
  }
  LOG_DEBUG("Finished setting Mf frequencies.");

  LOG_DEBUG("Preparing Collaterals...");
  prepCollaterals(randSeedGen->IRandom(0, INT_MAX));
  LOG_DEBUG("Finished preparing Collaterals...");
}

ECMFPopulation::ECMFPopulation(std::fstream &infile) {
  LOG_DEBUG("Allocating mf memory...");
  allocateMemory();
  LOG_DEBUG("Finished allocating mf memory.");
  LOG_DEBUG("Loading mfs from file...");
  rawBytesRW((char *)mfFreqBG, num_mf * sizeof(float), true, infile);
  rawBytesRW((char *)mfFreqCS, num_mf * sizeof(float), true, infile);
  rawBytesRW((char *)isCS, num_mf * sizeof(bool), true, infile);
  rawBytesRW((char *)isColl, num_mf * sizeof(bool), true, infile);
  rawBytesRW((char *)dnCellIndex, num_mf * sizeof(uint32_t), true, infile);
  rawBytesRW((char *)mZoneIndex, num_mf * sizeof(uint32_t), true, infile);
  LOG_DEBUG("finished loading mfs from file.");
}

ECMFPopulation::~ECMFPopulation() {
  delete[] mfFreqBG;
  delete[] mfFreqCS;

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
  free(dnCellIndex);
  free(mZoneIndex);
}

/* public methods except constructor and destructor */
void ECMFPopulation::writeToFile(std::fstream &outfile) {
  rawBytesRW((char *)mfFreqBG, num_mf * sizeof(float), false, outfile);
  rawBytesRW((char *)mfFreqCS, num_mf * sizeof(float), false, outfile);
  rawBytesRW((char *)isCS, num_mf * sizeof(bool), false, outfile);
  rawBytesRW((char *)isColl, num_mf * sizeof(bool), false, outfile);
  rawBytesRW((char *)dnCellIndex, num_mf * sizeof(uint32_t), false, outfile);
  rawBytesRW((char *)mZoneIndex, num_mf * sizeof(uint32_t), false, outfile);
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
      aps[i] = randGens[tid]->Random() < frequencies[i] + noise;
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

void ECMFPopulation::allocateMemory() {
  mfFreqBG = new float[num_mf]();
  mfFreqCS = new float[num_mf]();

  isCS = new bool[num_mf]();
  isColl = new bool[num_mf]();
  isAny = new bool[num_mf]();

  /* initializing poisson gen vars */

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
}

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
