/*
 * peristimhistfloat.h
 *
 *  Created on: Apr 28, 2014
 *      Author: consciousness
 */

#ifndef PERISTIMHISTFLOAT_H_
#define PERISTIMHISTFLOAT_H_

#include <iostream>
#include <fstream>
#include <vector>

#include <omp.h>

#include <math.h>

#include <stdDefinitions/pstdint.h>
#include <memoryMgmt/dynamic2darray.h>

class PeriStimHistFloat
{
public:
	PeriStimHistFloat(int numSteps, int stepsPerBin, int numCells);
	PeriStimHistFloat(std::fstream &infile);
	~PeriStimHistFloat();

	void update(int step, const ct_uint8_t* condition);
	void update(const ct_uint8_t* condition);
	void update(const ct_uint32_t* counts);
	void update(const float* input);

	void writeData(std::fstream &outfile);

	std::vector<float> getCellPSH(int cellN);
	std::vector<std::vector<float> > getPopPSHs();

	ct_uint32_t getNumCells();
	ct_uint32_t getNumBins();
	ct_uint32_t getBinWidthInMS();
	ct_uint32_t getNumTrials();
private:
	float** cells;

	ct_uint32_t nCells;
	ct_uint32_t numTS;
	ct_uint32_t tsPerBin;
	ct_uint32_t numBins;

	ct_int32_t curTS;
	ct_uint32_t curTrialN;

	void setBase();
};




#endif /* PERISTIMHISTFLOAT_H_ */
