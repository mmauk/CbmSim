/*
 * peristimhist.h
 *
 *  Created on: Jan 14, 2013
 *      Author: varicella
 */

#ifndef PERISTIMHIST_H_
#define PERISTIMHIST_H_

#include <iostream>
#include <fstream>
#include <vector>

#include <omp.h>

#include <math.h>

#include <stdDefinitions/pstdint.h>
#include <memoryMgmt/dynamic2darray.h>

class PeriStimHist
{
public:
	PeriStimHist(int numSteps, int stepsPerBin, int numCells);
	PeriStimHist(std::fstream &infile);
	~PeriStimHist();

	void update(int step, const ct_uint8_t* condition);
	void update(const ct_uint8_t* condition);
	void update(const ct_uint32_t* counts);
	void writeAndEnd(int currentTrial, std::fstream &outfile);

	void writeData(std::fstream &outfile);

	std::vector<ct_uint32_t> getCellPSH(int cellN);
	std::vector<std::vector<ct_uint32_t> > getPopPSHs();

	ct_uint32_t getNumCells();
	ct_uint32_t getNumBins();
	ct_uint32_t getBinWidthInMS();
	ct_uint32_t getNumTrials();
private:
	ct_uint32_t** cells;

	ct_uint32_t nCells;
	ct_uint32_t numTS;
	ct_uint32_t tsPerBin;
	ct_uint32_t numBins;

	ct_int32_t curTS;
	ct_uint32_t curTrialN;

	void setBase();
};


#endif /* PERISTIMHIST_H_ */
