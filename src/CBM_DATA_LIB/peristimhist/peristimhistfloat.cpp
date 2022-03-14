/*
 * peristimhistfloat.cpp
 *
 *  Created on: Apr 28, 2014
 *      Author: consciousness
 */

#include "peristimhist/peristimhistfloat.h"

// TODO: GET RID OF THIS FILE, MAKE PERISTIMHIST FUNCS INTO TEMPLATED FUNCS WTFFF

using namespace std;


PeriStimHistFloat::PeriStimHistFloat(int numSteps, int stepsPerBin, int numCells)
{
	this->nCells=numCells;
	numTS=numSteps;
	tsPerBin=stepsPerBin;
	numBins=ceil(numTS/((float)tsPerBin));

	curTS=0;
	curTrialN=0;

	setBase();
}

PeriStimHistFloat::PeriStimHistFloat(fstream &infile)
{
	infile>>nCells;
	infile>>numTS;
	infile>>tsPerBin;
	infile>>numBins;
	infile>>curTS;
	infile>>curTrialN;

	infile.seekg(1, ios::cur);

	setBase();

	infile.read((char *)cells[0], numBins*nCells*sizeof(float));
}

PeriStimHistFloat::~PeriStimHistFloat()
{
	delete2DArray<float>(cells);
}

void PeriStimHistFloat::writeData(fstream &outfile)
{
	outfile<<nCells<<" ";
	outfile<<numTS<<" ";
	outfile<<tsPerBin<<" ";
	outfile<<numBins<<" ";
	outfile<<curTS<<" ";
	outfile<<curTrialN<<endl;

	outfile.write((char *)cells[0], numBins*nCells*sizeof(float));
}

void PeriStimHistFloat::setBase()
{
	cells=allocate2DArray<float>(numBins, nCells);

	for(int i=0; i<numBins; i++)
	{
		for(int j=0; j<nCells; j++)
		{
			cells[i][j]=0;
		}
	}
}

void PeriStimHistFloat::update(const ct_uint8_t* condition)
{
	int curBinN;

	curBinN=curTS/tsPerBin;

	for(int i=0; i<nCells; i++)
	{
		cells[curBinN][i]+=(condition[i]>0);
	}

	curTS++;
	if(curTS>=numTS)
	{
		curTS=0;
		curTrialN++;
	}
}

void PeriStimHistFloat::update(const ct_uint32_t* counts)
{
	int curBinN;

	curBinN=curTS/tsPerBin;
	for(int i=0; i<nCells; i++)
	{
		cells[curBinN][i]+=counts[i];
	}

	curTS++;
	if(curTS>=numTS)
	{
		curTS=0;
		curTrialN++;
	}
}

void PeriStimHistFloat::update(const float *input)
{
	int curBinN;
	curBinN=curTS/tsPerBin;
	for(int i=0; i<nCells; i++)
	{
		cells[curBinN][i]+=input[i];
	}

	curTS++;
	if(curTS>=numTS)
	{
		curTS=0;
		curTrialN++;
	}
}

vector<float> PeriStimHistFloat::getCellPSH(int cellN)
{
	vector<float> psh;

	psh.resize(numBins);

	for(int i=0; i<numBins; i++)
	{
		psh[i]=cells[i][cellN];
	}

	return psh;
}

vector<vector<float> > PeriStimHistFloat::getPopPSHs()
{
	vector<vector<float> > popPSHs;

	popPSHs.resize(nCells);

	for(int i=0; i<nCells; i++)
	{
		popPSHs[i]=getCellPSH(i);
	}

	return popPSHs;
}

ct_uint32_t PeriStimHistFloat::getNumCells()
{
	return nCells;
}

ct_uint32_t PeriStimHistFloat::getNumBins()
{
	return numBins;
}

ct_uint32_t PeriStimHistFloat::getBinWidthInMS()
{
	return tsPerBin;
}

ct_uint32_t PeriStimHistFloat::getNumTrials()
{
	return curTrialN;
}


