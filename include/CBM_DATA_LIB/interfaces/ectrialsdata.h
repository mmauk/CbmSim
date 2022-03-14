/*
 * ectrialsdata.h
 *
 *  Created on: Jan 23, 2013
 *      Author: consciousness
 */

#ifndef ECTRIALSDATA_H_
#define ECTRIALSDATA_H_

#include <fstream>
#include <iostream>
#include <string>
#include <map>

#include <stdDefinitions/pstdint.h>

#include "outdata/eyelidout.h"
#include "outdata/rawuintdata.h"
#include "spikeraster/spikerasterbitarray.h"
#include "interfaces/ispikeraster.h"
#include "peristimhist/peristimhist.h"
#include "peristimhist/peristimhistfloat.h"

struct PSHParams
{
	unsigned int numTimeStepsPerBin;
	unsigned int numCells;
};

struct RasterParams
{
	unsigned int numCells;
};

struct RawUIntParams
{
	unsigned int numRows;
};

struct EyelidOutParams
{
	unsigned int numTimeStepSmooth;
};

class ECTrialsData
{
public:
	ECTrialsData(unsigned int msPreCS, unsigned int msCS, unsigned int msPostCS,
			unsigned int msPerTimeStep, unsigned int numTrials,
			std::map<std::string, PSHParams> periStimH,
			std::map<std::string, RasterParams> cellRasters,
			std::map<std::string, RawUIntParams> rawUInt,
			EyelidOutParams eyelidout);

	ECTrialsData(std::fstream &infile);

	~ECTrialsData();

	void writeData(std::fstream &outfile);

	void updatePSH(std::string cellT, const ct_uint8_t *aps);
	void updatePSH(std::string cellT, const ct_uint32_t *counts);
	void updatePSH(std::string cellT, const float *input);
	void updateRaster(std::string cellT, const ct_uint32_t *apBufs);
	void updateRawUInt(std::string label, const ct_uint32_t *input);
	void updateEyelid(float data);

	unsigned int getTSPerRasterUpdate();

	ISpikeRaster* getRaster(std::string cellT);

	PeriStimHistFloat* getPSH(std::string cellT);

	RawUIntData* getUIntData(std::string label);

	EyelidOut* getEyelidData();

	unsigned int getMSPreTrial();
	unsigned int getMSTrial();
	unsigned int getMSPostTrial();
	unsigned int getMSPerTimeStep();
	unsigned int getMSTotal();
	unsigned int getTimeStepsTotal();
	unsigned int getMaxNumTrials();

private:
	ECTrialsData();

	ct_uint32_t msPreCS;
	ct_uint32_t msCS;
	ct_uint32_t msPostCS;
	ct_uint32_t msPerTS;
	ct_uint32_t msTotal;
	ct_uint32_t tsTotal;

	ct_uint32_t numTrials;

	std::map<std::string, PeriStimHistFloat*> pshs;
	std::map<std::string, SpikeRasterBitArray*> rasters;
	std::map<std::string, RawUIntData*> uintData;

	EyelidOut *eyelidOut;
};

#endif /* ECTRIALSDATA_H_ */
