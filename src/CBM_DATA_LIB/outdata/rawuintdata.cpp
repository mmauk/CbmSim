/*
 * rawuintdata.cpp
 *
 *  Created on: May 12, 2013
 *      Author: consciousness
 */

#include "outdata/rawuintdata.h"

using namespace std;

RawUIntData::RawUIntData(string label, unsigned int numRows, unsigned int numTrials,
		unsigned int numPointsPerTrial)
{
	dataLabel=label;
	numDataRows=numRows;
	maxNumTrials=numTrials;
	this->numPointsPerTrial=numPointsPerTrial;

	curTrialN=0;
	curPoint=0;


	allocateMemory();
}

RawUIntData::RawUIntData(fstream &infile)
{
	infile>>dataLabel;
	infile>>numDataRows;
	infile>>maxNumTrials;
	infile>>numPointsPerTrial;
	infile>>curTrialN;
	infile>>curPoint;
	infile.seekg(1, ios::cur);

	dataIO(true, infile);
}

RawUIntData::~RawUIntData()
{
	for(int i=0; i<data.size(); i++)
	{
		delete2DArray<ct_uint32_t>(data[i]);
	}
}

void RawUIntData::writeData(fstream &outfile)
{
	outfile<<dataLabel<<" ";
	outfile<<numDataRows<<" ";
	outfile<<maxNumTrials<<" ";
	outfile<<numPointsPerTrial<<" ";
	outfile<<curTrialN<<" ";
	outfile<<curPoint<<endl;

	dataIO(false, outfile);
}

bool RawUIntData::updateData(const ct_uint32_t *input)
{
	ct_uint32_t **curTrial;

	if(curTrialN>=maxNumTrials)
	{
		return false;
	}

	curTrial=data[curTrialN];

	for(int i=0; i<numDataRows; i++)
	{
		curTrial[curPoint][i]=input[i];
	}

	curPoint++;
	if(curPoint>=numPointsPerTrial)
	{
		curTrialN++;
		curPoint=0;
	}

	return true;
}

unsigned int RawUIntData::getNumTrials()
{
	return maxNumTrials;
}

unsigned int RawUIntData::getNumDataRows()
{
	return numDataRows;
}

unsigned int RawUIntData::getNumPointsPerTrial()
{
	return numPointsPerTrial;
}

void RawUIntData::allocateMemory()
{
	data.resize(maxNumTrials);

	for(int i=0; i<maxNumTrials; i++)
	{
		data[i]=allocate2DArray<ct_uint32_t>(numPointsPerTrial, numDataRows);
	}
}

void RawUIntData::dataIO(bool read, fstream &file)
{
	for(int i=0; i<maxNumTrials; i++)
	{
		rawBytesRW((char *)data[i][0],
				numPointsPerTrial*numDataRows*sizeof(ct_uint32_t), read, file);
	}
}
