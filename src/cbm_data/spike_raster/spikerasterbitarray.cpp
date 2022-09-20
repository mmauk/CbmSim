/*
 * spikerasterbitarray.cpp
 *
 *  Created on: Aug 10, 2012
 *      Author: consciousness
 */

#include "spikerasterbitarray.h"

using namespace std;

SpikeRasterBitArray::SpikeRasterBitArray(string cellType, unsigned int numCells,
		unsigned int numTrials, unsigned int msPerTrial, unsigned int msPerTimeStep)
		:ISpikeRaster(cellType, numCells, numTrials, msPerTrial)
{
	msPerTS=msPerTimeStep;
	curTrialN=0;
	curTS=0;
	curBufN=0;
	tsPerTrial=msPerTrial/msPerTS;
	bufsPerTrial=floor(tsPerTrial/32.0);
	allocateMemory();
}

SpikeRasterBitArray::SpikeRasterBitArray(fstream &infile):ISpikeRaster(infile)
{
	infile>>msPerTS;
	infile>>curTrialN;
	infile>>curTS;
	infile>>curBufN;
	infile>>tsPerTrial;
	infile>>bufsPerTrial;
	infile.seekg(1, ios::cur);

	allocateMemory();

	dataIO(true, infile);
}

SpikeRasterBitArray::~SpikeRasterBitArray()
{
	for(int i=0; i<rasterBitArr.size(); i++)
	{
		delete2DArray<ct_uint32_t>(rasterBitArr[i]);
	}
}

void SpikeRasterBitArray::writeData(fstream &outfile)
{
	ISpikeRaster::writeData(outfile);

	outfile<<msPerTS<<" ";
	outfile<<curTrialN<<" ";
	outfile<<curTS<<" ";
	outfile<<curBufN<<" ";
	outfile<<tsPerTrial<<" ";
	outfile<<bufsPerTrial<<endl;

	dataIO(false, outfile);
}

bool SpikeRasterBitArray::updateRaster(const ct_uint32_t *input)
{
	ct_uint32_t **curRaster;
	
	if (curTrialN >= nTrials) return false;

	curRaster=rasterBitArr[curTrialN];

	for(int i=0; i<nCells; i++)
	{
		curRaster[curBufN][i]=input[i];
	}

	curBufN++;
	curTS=curTS+32;
	if(curBufN>=bufsPerTrial)
	{
		curTrialN++;
		curBufN=0;
		curTS=0;
	}

	return true;
}

vector<vector<int>> SpikeRasterBitArray::getCellSpikeTimes(unsigned cellN, int offset)
{
	vector<vector<int>> spikeTimes;

	for(unsigned int i=0; i<curTrialN; i++)
	{
		spikeTimes.push_back(getCellSpikeTimes(cellN, i, offset));
	}

	return spikeTimes;
}

vector<int> SpikeRasterBitArray::getCellSpikeTimes(unsigned int cellN, unsigned int trialN, int offset)
{
	vector<int> spikeTimes;

	int timeCounter;

	ct_uint32_t **curRaster;

	curRaster=rasterBitArr[trialN];

	timeCounter=offset;

	for(int i=0; i<bufsPerTrial; i++)
	{
		ct_uint32_t tempBuf;

		tempBuf=curRaster[i][cellN];
		for(int j=0; j<32; j++)
		{
			if((tempBuf&0x80000000)>0)
			{
				spikeTimes.push_back(timeCounter);
			}
			timeCounter+=msPerTS;
			tempBuf=tempBuf<<1;
		}
	}

	return spikeTimes;
}

vector<bool> SpikeRasterBitArray::getPopSpikesAtTS(unsigned int timeStep, unsigned int trialN)
{
	unsigned int bufN;
	unsigned int bitN;
	ct_uint32_t bufMask;

	vector<bool> aps;

	aps.resize(nCells);

	bufN=timeStep/32;
	bitN=timeStep%32;

	if(bufN>=bufsPerTrial)
	{
		for(int i=0; i<nCells; i++)
		{
			aps[i]=false;
		}

		return aps;
	}

	bufMask=0x80000000>>bitN;

	for(int i=0; i<nCells; i++)
	{
		aps[i]=(rasterBitArr[trialN][bufN][i]&bufMask)>0;
	}

	return aps;

}


unsigned int SpikeRasterBitArray::getNumTrials()
{
	return curTrialN;
}

unsigned int SpikeRasterBitArray::getUpdateInterval()
{
	return 32;
}

void SpikeRasterBitArray::allocateMemory()
{
	rasterBitArr.resize(nTrials);

	for(int i=0; i<nTrials; i++)
	{
		rasterBitArr[i]=allocate2DArray<ct_uint32_t>(bufsPerTrial, nCells);
	}
}

void SpikeRasterBitArray::dataIO(bool read, fstream &file)
{
	for(int i=0; i<nTrials; i++)
	{
		rawBytesRW((char *)rasterBitArr[i][0],
				bufsPerTrial*nCells*sizeof(ct_uint32_t), read, file);
	}
}
