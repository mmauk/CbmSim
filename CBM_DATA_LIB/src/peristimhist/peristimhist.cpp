/*
 * peristimhist.cpp
 *
 *  Created on: Jan 14, 2013
 *      Author: varicella
 */

#include "peristimhist/peristimhist.h"

using namespace std;


PeriStimHist::PeriStimHist(int numSteps, int stepsPerBin, int numCells)
{
	this->nCells=numCells;
	numTS=numSteps;
	tsPerBin=stepsPerBin;
	numBins=ceil(numTS/((float)tsPerBin));

	curTS=0;
	curTrialN=0;

	setBase();
}

PeriStimHist::PeriStimHist(fstream &infile)
{
	infile>>nCells;
	infile>>numTS;
	infile>>tsPerBin;
	infile>>numBins;
	infile>>curTS;
	infile>>curTrialN;

	infile.seekg(1, ios::cur);

	setBase();

	infile.read((char *)cells[0], numBins*nCells*sizeof(ct_uint32_t));
}

PeriStimHist::~PeriStimHist()
{
	delete2DArray<ct_uint32_t>(cells);
}

void PeriStimHist::writeData(fstream &outfile)
{
	outfile<<nCells<<" ";
	outfile<<numTS<<" ";
	outfile<<tsPerBin<<" ";
	outfile<<numBins<<" ";
	outfile<<curTS<<" ";
	outfile<<curTrialN<<endl;

	outfile.write((char *)cells[0], numBins*nCells*sizeof(ct_uint32_t));
}

void PeriStimHist::setBase()
{
	cells=allocate2DArray<ct_uint32_t>(numBins, nCells);

	for(int i=0; i<numBins; i++)
	{
		for(int j=0; j<nCells; j++)
		{
			cells[i][j]=0;
		}
	}
}

void PeriStimHist::update(const ct_uint8_t* condition)
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

void PeriStimHist::update(const ct_uint32_t* counts)
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

void PeriStimHist::writeAndEnd(int currentTrial, fstream &outfile)
{
	cout<<"Total Trials Stated:"<<currentTrial<<endl;
	outfile<<curTrialN<<endl;
	for(int i=0; i<numBins; i++)
	{
		outfile<<tsPerBin*i<<"\t";
	}
	outfile<<endl;
	for(int i=0; i<nCells; i++)
	{
		for(int j=0; j<numBins; j++)
		{
			outfile<<cells[j][i]<<"\t";
		}
		outfile<<endl;
	}
}

vector<unsigned int> PeriStimHist::getCellPSH(int cellN)
{
	vector<unsigned int> psh;

	psh.resize(numBins);

	for(int i=0; i<numBins; i++)
	{
		psh[i]=cells[i][cellN];
	}

	return psh;
}

vector<vector<ct_uint32_t> > PeriStimHist::getPopPSHs()
{
	vector<vector<ct_uint32_t> > popPSHs;

	popPSHs.resize(nCells);

	for(int i=0; i<nCells; i++)
	{
		popPSHs[i]=getCellPSH(i);
	}

	return popPSHs;
}

ct_uint32_t PeriStimHist::getNumCells()
{
	return nCells;
}

ct_uint32_t PeriStimHist::getNumBins()
{
	return numBins;
}

ct_uint32_t PeriStimHist::getBinWidthInMS()
{
	return tsPerBin;
}

ct_uint32_t PeriStimHist::getNumTrials()
{
	return curTrialN;
}
