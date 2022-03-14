/*
 * spikerasterbitarray.h
 *
 *  Created on: Aug 10, 2012
 *      Author: consciousness
 */

#ifndef SPIKERASTERBITARRAY_H_
#define SPIKERASTERBITARRAY_H_

#include "interfaces/ispikeraster.h"
#include <fstream>
#include <string>
#include <vector>

#include <math.h>
#include <stdDefinitions/pstdint.h>
#include <memoryMgmt/dynamic2darray.h>
#include <fileIO/rawbytesrw.h>

class SpikeRasterBitArray : public ISpikeRaster
{
public:
	SpikeRasterBitArray(std::string cellType, unsigned int numCells, unsigned int numTrials,
			unsigned int msPerTrial, unsigned int msPerTimeStep);
	SpikeRasterBitArray(std::fstream &infile);
	~SpikeRasterBitArray();

	virtual void writeData(std::fstream &outfile);
	virtual bool updateRaster(const ct_uint32_t *input);
	virtual std::vector<std::vector<int> > getCellSpikeTimes(unsigned int cellN, int offset);
	virtual std::vector<int> getCellSpikeTimes(unsigned int cellN, unsigned int trialN, int offset);
	virtual std::vector<bool> getPopSpikesAtTS(unsigned int timeStep, unsigned int trialN);

	virtual unsigned int getNumTrials();

	virtual unsigned int getUpdateInterval();
protected:
	ct_uint32_t msPerTS;

	std::vector<ct_uint32_t **> rasterBitArr;

	unsigned int curTrialN;
	unsigned int curTS;
	unsigned int curBufN;
	unsigned int tsPerTrial;
	unsigned int bufsPerTrial;

private:
	SpikeRasterBitArray();

	void allocateMemory();

	void dataIO(bool read, std::fstream &file);
};

#endif /* SPIKERASTERBITARRAY_H_ */
