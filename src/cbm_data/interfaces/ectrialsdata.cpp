/*
 * ectrialsdata.cpp
 *
 *  Created on: Jan 28, 2013
 *      Author: consciousness
 */

#include "interfaces/ectrialsdata.h"

using namespace std;

ECTrialsData::ECTrialsData(unsigned int msPreCS, unsigned int msCS,
		unsigned int msPostCS, unsigned int msPerTimeStep, unsigned int numTrials,
		map<string, PSHParams> periStimH,
		map<string, RasterParams> cellRasters,
		map<string, RawUIntParams> rawUInt,
		EyelidOutParams eyelidout)
{
	map<string, PSHParams>::iterator pshI;
	map<string, RasterParams>::iterator rasterI;
	map<string, RawUIntParams>::iterator rawUIntI;

	this->msPreCS=msPreCS;
	this->msCS=msCS;
	this->msPostCS=msPostCS;
	msTotal=msPreCS+msCS+msPostCS;

	msPerTS=msPerTimeStep;
	tsTotal=msTotal/msPerTS;

	this->numTrials=numTrials;


	eyelidOut=new EyelidOut(numTrials, msPerTS, eyelidout.numTimeStepSmooth, msTotal);

	for(rawUIntI=rawUInt.begin(); rawUIntI!=rawUInt.end(); rawUIntI++)
	{
		uintData[rawUIntI->first]=new RawUIntData(rawUIntI->first, rawUIntI->second.numRows, numTrials, msTotal/msPerTS);
	}

	for(pshI=periStimH.begin(); pshI!=periStimH.end(); pshI++)
	{
		pshs[pshI->first]=new PeriStimHistFloat(tsTotal, pshI->second.numTimeStepsPerBin, pshI->second.numCells);
	}

	for(rasterI=cellRasters.begin(); rasterI!=cellRasters.end(); rasterI++)
	{
		rasters[rasterI->first]=new SpikeRasterBitArray(rasterI->first,
				rasterI->second.numCells, numTrials, msTotal, msPerTS);
	}
}

ECTrialsData::ECTrialsData(fstream &infile)
{
	ct_uint32_t numPSHs;
	ct_uint32_t numRasters;
	ct_uint32_t numRawUInts;

	infile>>msPreCS;
	infile>>msCS;
	infile>>msPostCS;
	infile>>msTotal;
	infile>>msPerTS;
	infile>>tsTotal;
	infile>>numTrials;

	infile.seekg(1, ios::cur);

	eyelidOut=new EyelidOut(infile);

	infile>>numPSHs;

	for(int i=0; i<numPSHs; i++)
	{
		string cellT;

		infile>>cellT;
		infile.seekg(1, ios::cur);

		pshs[cellT]=new PeriStimHistFloat(infile);
	}

	infile>>numRasters;
	for(int i=0; i<numRasters; i++)
	{
		string CellT;

		infile>>CellT;
		infile.seekg(1, ios::cur);

		rasters[CellT]=new SpikeRasterBitArray(infile);
	}

	numRawUInts=0;
	infile>>numRawUInts;
	for(int i=0; i<numRawUInts; i++)
	{
		string label;
		infile>>label;
		infile.seekg(1, ios::cur);

		uintData[label]=new RawUIntData(infile);
	}
}

ECTrialsData::~ECTrialsData()
{
	map<string, PeriStimHistFloat*>::iterator pshI;
	map<string, SpikeRasterBitArray*>::iterator rasterI;
	map<string, RawUIntData*>::iterator uintDataI;

	for(pshI=pshs.begin(); pshI!=pshs.end(); pshI++)
	{
		delete pshI->second;
	}
	pshs.clear();

	for(rasterI=rasters.begin(); rasterI!=rasters.end(); rasterI++)
	{
		delete rasterI->second;
	}
	rasters.clear();

	for(uintDataI=uintData.begin(); uintDataI!=uintData.end(); uintDataI++)
	{
		delete uintDataI->second;
	}
	uintData.clear();

	delete eyelidOut;
}

void ECTrialsData::writeData(fstream &outfile)
{
	map<string, PeriStimHistFloat*>::iterator pshI;
	map<string, SpikeRasterBitArray*>::iterator rasterI;
	map<string, RawUIntData*>::iterator uintDataI;

	outfile<<msPreCS<<" ";
	outfile<<msCS<<" ";
	outfile<<msPostCS<<" ";
	outfile<<msTotal<<" ";
	outfile<<msPerTS<<" ";
	outfile<<tsTotal<<" ";
	outfile<<numTrials<<endl;

	eyelidOut->writeData(outfile);

	outfile<<pshs.size()<<endl;
	for(pshI=pshs.begin(); pshI!=pshs.end(); pshI++)
	{
		outfile<<pshI->first<<endl;
		pshs[pshI->first]->writeData(outfile);
	}
	outfile<<rasters.size()<<endl;
	for(rasterI=rasters.begin(); rasterI!=rasters.end(); rasterI++)
	{
		outfile<<rasterI->first<<endl;
		rasters[rasterI->first]->writeData(outfile);
	}
	outfile<<uintData.size()<<endl;
	for(uintDataI=uintData.begin(); uintDataI!=uintData.end(); uintDataI++)
	{
		outfile<<uintDataI->first<<endl;
		uintData[uintDataI->first]->writeData(outfile);
	}
}

void ECTrialsData::updatePSH(string cellT, const ct_uint8_t *aps)
{
	pshs[cellT]->update(aps);
}

void ECTrialsData::updatePSH(string cellT, const ct_uint32_t *counts)
{
	pshs[cellT]->update(counts);
}

void ECTrialsData::updatePSH(string cellT, const float *input)
{
	pshs[cellT]->update(input);
}

void ECTrialsData::updateRaster(string cellT, const ct_uint32_t *apBufs)
{
	rasters[cellT]->updateRaster(apBufs);
}

void ECTrialsData::updateRawUInt(string label, const ct_uint32_t *input)
{
	uintData[label]->updateData(input);
}

void ECTrialsData::updateEyelid(float data)
{
	eyelidOut->updateData(data);
}

unsigned int ECTrialsData::getTSPerRasterUpdate()
{
	map<string, SpikeRasterBitArray*>::iterator rasterI;
	if(rasters.empty())
	{
		return 0;
	}
	rasterI=rasters.begin();
	return rasterI->second->getUpdateInterval();
}

ISpikeRaster* ECTrialsData::getRaster(string cellT)
{
	return (ISpikeRaster *)rasters[cellT];
}

PeriStimHistFloat* ECTrialsData::getPSH(string cellT)
{
	return pshs[cellT];
}

RawUIntData* ECTrialsData::getUIntData(string label)
{
	return uintData[label];
}

EyelidOut* ECTrialsData::getEyelidData()
{
	return eyelidOut;
}

unsigned int ECTrialsData::getMSPreTrial()
{
	return msPreCS;
}

unsigned int ECTrialsData::getMSTrial()
{
	return msCS;
}

unsigned int ECTrialsData::getMSPostTrial()
{
	return msPostCS;
}

unsigned int ECTrialsData::getMSPerTimeStep()
{
	return msPerTS;
}

unsigned int ECTrialsData::getMSTotal()
{
	return msTotal;
}

unsigned int ECTrialsData::getTimeStepsTotal()
{
	return tsTotal;
}

unsigned int ECTrialsData::getMaxNumTrials()
{
	return numTrials;
}
