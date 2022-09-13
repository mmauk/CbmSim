/*
 * poissonregencells.cpp
 *
 *  Created on: Dec 13, 2012
 *      Author: consciousness
 *  Modified on: Jul 22, 2015
 *  	Author: evandelord
 */

#include "poissonregencells.h"

PoissonRegenCells::PoissonRegenCells(unsigned int numCells, int randSeed, float threshDecayTau, float msPerTimeStep, 
	unsigned int numZones, unsigned int numNC, float sigma)
{
	randSeedGen = new CRandomSFMT0(randSeed);
	noiseRandGen = new std::mt19937(randSeed);

	nThreads=1;
	randGens=new CRandomSFMT0*[nThreads];

	for(unsigned int i=0; i<nThreads; i++)
	{
		randGens[i]=new CRandomSFMT0(randSeedGen->IRandom(0, INT_MAX));
	}

	nCells=numCells;
	this->sigma = sigma;
	this->numZones = numZones;
	this->numNC = numNC;
	msPerTS=msPerTimeStep;
	sPerTS=msPerTimeStep/1000;
	threshDecay=1-exp(-msPerTS/threshDecayTau);

	normDist = new std::normal_distribution<float>(0, this->sigma);

	aps=new ct_uint8_t[nCells];
	threshs=new float[nCells];

	isTrueMF= new bool[nCells];

	for(unsigned int i=0; i<nCells; i++)
	{
		aps[i]=0;
		threshs[i]=1;
		isTrueMF[i]=true;
	}

	dnCellIndex = new unsigned int[nCells];
	mZoneIndex = new unsigned int[nCells];
	
	prepCollaterals(randSeedGen->IRandom(0, INT_MAX));
}

PoissonRegenCells::~PoissonRegenCells()
{
	delete randSeedGen;
	delete noiseRandGen;
	for(unsigned int i=0; i<nThreads; i++)
	{
		delete randGens[i];
	}

	delete[] randGens;
	delete normDist;
	delete[] aps;
	delete[] threshs;
	delete[] isTrueMF;
	delete[] dnCellIndex;
	delete[] mZoneIndex;
}

void PoissonRegenCells::prepCollaterals(int rSeed)
{
	unsigned int repeats = nCells / (numZones * numNC) + 1;
	unsigned int *tempNCs = new unsigned int[repeats * numZones * numNC];
	unsigned int *tempMZs = new unsigned int[repeats * numZones * numNC];

	for(unsigned int i = 0; i < repeats; i++)
	{
		for(unsigned int j = 0; j < numZones; j++)
		{
			for(unsigned int k = 0; k < numNC; k++)
			{
				tempNCs[k + numNC * j + numNC * numZones * i] = k;
				tempMZs[k + numNC * j + numNC * numZones * i] = j;
			}
		}
	}
	std::srand(rSeed);
	std::random_shuffle(tempNCs, tempNCs + repeats*numZones*numNC);
	std::srand(rSeed);
	std::random_shuffle(tempMZs, tempMZs + repeats*numZones*numNC);
	std::copy(tempNCs, tempNCs + nCells, dnCellIndex);
	std::copy(tempMZs, tempMZs + nCells, mZoneIndex);

	delete[] tempNCs;
	delete[] tempMZs;
}

const ct_uint8_t* PoissonRegenCells::calcThreshActivity(const float *frequencies, MZone **mZoneList)
{
	int countColls = 0;
	const ct_uint8_t *holdNCs;
	for(unsigned int i=0; i<nCells; i++)
	{
		if(frequencies[i] == -1)
		{
			holdNCs = mZoneList[mZoneIndex[countColls]]->exportAPNC();
			aps[i] = holdNCs[dnCellIndex[countColls]];
			countColls++;
		}else
		{
			int tid = 0;

			threshs[i]=threshs[i]+(1-threshs[i])*threshDecay;
			aps[i]=randGens[tid]->Random()<((frequencies[i]*sPerTS)*threshs[i]);
			threshs[i]=(!aps[i])*threshs[i];
		}
	}

	return (const ct_uint8_t *)aps;
}

const ct_uint8_t* PoissonRegenCells::calcPoissActivity(const float *frequencies, MZone **mZoneList, int ispikei)
{
	int countColls = 0;
	const ct_uint8_t *holdNCs;
	float noise;
	spikeTimer++;
	for (unsigned int i = 0; i < nCells; i++)
	{
		if (frequencies[i] == -1)
		{
			holdNCs = mZoneList[mZoneIndex[countColls]]->exportAPNC();
			aps[i] = holdNCs[dnCellIndex[countColls]];
			countColls++;
		}
		else if (frequencies[i] == -2) aps[i] = (spikeTimer == ispikei);
		else
		{
			int tid = 0;
			if (sigma == 0) noise = 0.0;
			else noise = (*normDist)((*noiseRandGen));
			
			aps[i] = (randGens[tid]->Random() < (frequencies[i] + noise) * sPerTS);
		}
	}
	if (spikeTimer == ispikei) spikeTimer = 0;

	return (const ct_uint8_t *)aps;
}

bool* PoissonRegenCells::calcTrueMFs(const float *frequencies)
{
	for (unsigned int i = 0; i < nCells; i++)
	{
		if (frequencies[i] == -1)
		{
			isTrueMF[i] = false;
		}
	}
	return isTrueMF;
}

