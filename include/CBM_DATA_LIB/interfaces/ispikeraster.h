/*
 * ispikeraster.h
 *
 *  Created on: Jan 23, 2013
 *      Author: consciousness
 */

#ifndef ISPIKERASTER_H_
#define ISPIKERASTER_H_

#include <vector>
#include <fstream>
#include <string>
#include <stdDefinitions/pstdint.h>

class ISpikeRaster
{
public:
	ISpikeRaster(std::string cellType, unsigned int numCells, unsigned int numTrials,
			unsigned int msPerTrial);
	ISpikeRaster(std::fstream &infile);
	virtual ~ISpikeRaster();
	virtual void writeData(std::fstream &outfile);
	virtual bool updateRaster(const ct_uint32_t *inputBuf)=0;
	virtual std::vector<std::vector<int> > getCellSpikeTimes(unsigned int cellN, int offset)=0;
	virtual std::vector<int> getCellSpikeTimes(unsigned int cellN, unsigned int trialN, int offset)=0;
	virtual std::vector<bool> getPopSpikesAtTS(unsigned int timeStep, unsigned int trialN)=0;

	virtual unsigned int getUpdateInterval()=0;

	virtual std::string getCellType();
	virtual unsigned int getNumCells();
	virtual unsigned int getNumTrials();
	virtual unsigned int getMSPerTrial();

protected:
	std::string cellT;
	ct_uint32_t nCells;
	ct_uint32_t nTrials;
	ct_uint32_t msPerTrial;

private:
	ISpikeRaster();

};

#endif /* ISPIKERASTER_H_ */
