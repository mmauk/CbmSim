/*
 * poissonregencells.cpp
 *
 *  Created on: Dec 13, 2012
 *      Author: consciousness
 *  Modified on: Jul 22, 2015
 *  	Author: evandelord
 */

#include "poissonregencells.h"

PoissonRegenCells::PoissonRegenCells(int randSeed, float threshDecayTau, unsigned int numZones, float sigma)
{
	randSeedGen = new CRandomSFMT0(randSeed);
	noiseRandGen = new std::mt19937(randSeed);

	nThreads=1;
	randGens=new CRandomSFMT0*[nThreads];

	for(unsigned int i=0; i<nThreads; i++)
	{
		randGens[i]=new CRandomSFMT0(randSeedGen->IRandom(0, INT_MAX));
	}

	this->sigma = sigma;
	this->numZones = numZones;
	sPerTS = msPerTimeStep / 1000;

	normDist = new std::normal_distribution<float>(0, this->sigma);

	aps         = (uint8_t *)calloc(num_mf, sizeof(uint8_t));
	isTrueMF    = (bool *)calloc(num_mf, sizeof(bool));
	dnCellIndex = (uint32_t *)calloc(num_mf, sizeof(uint32_t));
	mZoneIndex  = (uint32_t *)calloc(num_mf, sizeof(uint32_t));

	prepCollaterals(randSeedGen->IRandom(0, INT_MAX));
}

PoissonRegenCells::~PoissonRegenCells()
{
	delete randSeedGen;
	delete noiseRandGen;
	for(uint32_t i=0; i<nThreads; i++)
	{
		delete randGens[i];
	}

	delete[] randGens;
	delete normDist;
	free(aps);
	free(isTrueMF);
	free(dnCellIndex);
	free(mZoneIndex);
}

void PoissonRegenCells::prepCollaterals(int rSeed)
{
	uint32_t repeats = num_mf / (numZones * num_nc) + 1;
	uint32_t *tempNCs = new uint32_t[repeats * numZones * num_nc];
	uint32_t *tempMZs = new uint32_t[repeats * numZones * num_nc];

	for(uint32_t i = 0; i < repeats; i++)
	{
		for(uint32_t j = 0; j < numZones; j++)
		{
			for(uint32_t k = 0; k < num_nc; k++)
			{
				tempNCs[k + num_nc * j + num_nc * numZones * i] = k;
				tempMZs[k + num_nc * j + num_nc * numZones * i] = j;
			}
		}
	}
	std::srand(rSeed);
	std::random_shuffle(tempNCs, tempNCs + repeats*numZones*num_nc);
	std::srand(rSeed);
	std::random_shuffle(tempMZs, tempMZs + repeats*numZones*num_nc);
	std::copy(tempNCs, tempNCs + num_mf, dnCellIndex);
	std::copy(tempMZs, tempMZs + num_mf, mZoneIndex);

	delete[] tempNCs;
	delete[] tempMZs;
}

const uint8_t* PoissonRegenCells::calcPoissActivity(const float *frequencies, MZone **mZoneList, int ispikei)
{
	int countColls = 0;
	const uint8_t *holdNCs;
	float noise;
	spikeTimer++;
	for (uint32_t i = 0; i < num_mf; i++)
	{
		if (frequencies[i] == -1) /* dcn mfs (or collaterals this makes no sense) */
		{
			holdNCs = mZoneList[mZoneIndex[countColls]]->exportAPNC();
			aps[i] = holdNCs[dnCellIndex[countColls]];
			countColls++;
		}
		// below is calculated w isi. why not do so for cs too?
		else if (frequencies[i] == -2) aps[i] = (spikeTimer == ispikei); /* background or import, whatever that is */
		else /* cs */
		{
			int tid = 0;
			if (sigma == 0) noise = 0.0;
			else noise = (*normDist)((*noiseRandGen));
			
			aps[i] = (randGens[tid]->Random() < (frequencies[i] + noise) * sPerTS);
		}
	}
	if (spikeTimer == ispikei) spikeTimer = 0;
	return (const uint8_t *)aps;
}

/*
 * Note: should be called something like "calcCollateralMFs." also, 
 * depending on the sets of mfs, why calculate this every time step. waste of time.
 */
bool* PoissonRegenCells::calcTrueMFs(const float *frequencies)
{
	for (uint32_t i = 0; i < num_mf; i++)
	{
		if (frequencies[i] == -1)
		{
			isTrueMF[i] = false;
		}
	}
	return isTrueMF;
}

const uint8_t* PoissonRegenCells::getAPs()
{
	return (const uint8_t *)aps;
}


