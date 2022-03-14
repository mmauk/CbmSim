/*
 * ispikeraster.cpp
 *
 *  Created on: Jan 23, 2013
 *      Author: consciousness
 */

#include "interfaces/ispikeraster.h"

using namespace std;

ISpikeRaster::ISpikeRaster(std::string cellType, unsigned int numCells,
		unsigned int numTrials, unsigned int msPerTrial)
{
	cellT=cellType;
	this->nCells=numCells;
	this->nTrials=numTrials;
	this->msPerTrial=msPerTrial;
}

ISpikeRaster::ISpikeRaster(fstream &infile)
{
	infile>>cellT;
	infile>>nCells;
	infile>>nTrials;
	infile>>msPerTrial;
	infile.seekg(1, ios::cur);
}

ISpikeRaster::~ISpikeRaster()
{

}

void ISpikeRaster::writeData(fstream &outfile)
{
	outfile<<cellT<<" ";
	outfile<<nCells<<" ";
	outfile<<nTrials<<" ";
	outfile<<msPerTrial<<" ";
}

string ISpikeRaster::getCellType()
{
	return cellT;
}
unsigned int ISpikeRaster::getNumCells()
{
	return nCells;
}
unsigned int ISpikeRaster::getNumTrials()
{
	return nTrials;
}
unsigned int ISpikeRaster::getMSPerTrial()
{
	return msPerTrial;
}

