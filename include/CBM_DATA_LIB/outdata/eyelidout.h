/*
 * eyelidout.h
 *
 *  Created on: Aug 28, 2012
 *      Author: consciousness
 */

#ifndef EYELIDOUT_H_
#define EYELIDOUT_H_

#include <fstream>
#include <vector>

#include <stdDefinitions/pstdint.h>
#include <memoryMgmt/dynamic2darray.h>
#include <fileIO/rawbytesrw.h>

class EyelidOut
{
public:
	EyelidOut(unsigned int numTrials, unsigned int msPerTimeStep, unsigned int numTimeStepSmooth,
			unsigned int msPerTrial);
	EyelidOut(std::fstream &infile);

	~EyelidOut();

	void writeData(std::fstream &outfile);

	bool updateData(float data);

	ct_uint32_t getNumTrials();

	ct_uint32_t getNumPointsPerTrial();

	ct_uint32_t getMSPerPoint();

	std::vector<float> getEyelidData(unsigned int trialN);
	std::vector<std::vector<float> > getEyelidData(std::vector<unsigned int> trialNs);
	std::vector<std::vector<float> > getEyelidData(unsigned int startTrialN, unsigned int endTrialN);

private:
	EyelidOut();

	void allocateMemory();

	void dataIO(bool read, std::fstream &file);

	ct_uint32_t nTrials;
	ct_uint32_t msPerTS;
	ct_uint32_t numTSSmooth;
	ct_uint32_t msPerTrial;
	ct_uint32_t tsPerTrial;

	ct_uint32_t curTrialN;
	ct_uint32_t curTS;

	float **rawData;
};


#endif /* EYELIDOUT_H_ */
