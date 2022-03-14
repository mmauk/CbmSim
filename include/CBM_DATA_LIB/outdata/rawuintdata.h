/*
 * rawuintdata.h
 *
 *  Created on: May 11, 2013
 *      Author: consciousness
 */

#ifndef RAWUINTDATA_H_
#define RAWUINTDATA_H_

#include <fstream>
#include <vector>

#include <stdDefinitions/pstdint.h>
#include <memoryMgmt/dynamic2darray.h>
#include <fileIO/rawbytesrw.h>

class RawUIntData
{
public:
	RawUIntData(std::string label, unsigned int numRows, unsigned int numTrials,
			unsigned int numPointsPerTrial);

	RawUIntData(std::fstream &infile);

	virtual ~RawUIntData();

	virtual void writeData(std::fstream &outfile);

	virtual bool updateData(const ct_uint32_t *input);

	virtual unsigned int getNumTrials();
	virtual unsigned int getNumDataRows();
	virtual unsigned int getNumPointsPerTrial();

protected:
	std::vector<ct_uint32_t **> data;

	std::string dataLabel;
	unsigned int numDataRows;
	unsigned int maxNumTrials;
	unsigned int numPointsPerTrial;
	unsigned int curTrialN;
	unsigned int curPoint;

private:
	RawUIntData();

	void allocateMemory();
	void dataIO(bool read, std::fstream &file);
};

#endif /* RAWUINTDATA_H_ */
