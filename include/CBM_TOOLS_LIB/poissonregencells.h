/*
 * poissonregencells.h
 *
 *  Created on: Dec 13, 2012
 *      Author: consciousness
 */

#ifndef POISSONREGENCELLS_H_
#define POISSONREGENCELLS_H_

#include <iostream>
#include <algorithm> // for random_shuffle
#include <cstdlib> // for srand and rand, sorry Wen
#include <random>
#include <math.h>
#include <limits.h>

#include <stdDefinitions/pstdint.h>
#include <randGenerators/sfmt.h>
#include <interface/mzoneinterface.h>

class PoissonRegenCells
{
public:
	PoissonRegenCells(unsigned int numCells, int randSeed, float threshDecayTau, float msPerTimeStep, 
		unsigned int numZones, unsigned int numNC, float sigma=0);
	~PoissonRegenCells();

	const ct_uint8_t* calcThreshActivity(const float *freqencies, MZoneInterface **mZoneList); 
	const ct_uint8_t* calcPoissActivity(const float *freqencies, MZoneInterface **mZoneList, int ispikei = 18); 
	bool* calcTrueMFs(const float *freqencies);
private:
	PoissonRegenCells();
	void prepCollaterals(int rSeed);

	std::normal_distribution<float> *normDist;
	std::mt19937 *noiseRandGen;
	CRandomSFMT0 *randSeedGen;
	CRandomSFMT0 **randGens;

	unsigned int nThreads;

	unsigned int nCells;
	unsigned int numZones;
	unsigned int numNC; 
	float msPerTS;
	float sPerTS;
	float threshDecay;
	float sigma;

	ct_uint8_t *aps;
	float *threshs;
	unsigned int *dnCellIndex;
	unsigned int *mZoneIndex;
	int spikeTimer = 0;

	bool *isTrueMF;

};


#endif /* POISSONREGENCELLS_H_ */
