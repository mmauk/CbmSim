/*
 * innetactivitystate.h
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 */

#ifndef INNETACTIVITYSTATE_H_
#define INNETACTIVITYSTATE_H_

#include <memory>
#include <fstream>

#include "file_utility.h"
#include <cstdint>
#include "connectivityparams.h"
#include "activityparams.h"

class InNetActivityState
{
public:
	InNetActivityState();
	InNetActivityState(std::fstream &infile);

	~InNetActivityState();

	void readState(std::fstream &infile);
	void writeState(std::fstream &outfile);
	void resetState();

	//mossy fiber
	std::unique_ptr<uint8_t[]> histMF{nullptr};
	std::unique_ptr<uint32_t[]> apBufMF{nullptr};

	//golgi cells
	std::unique_ptr<float[]> synWscalerGOtoGO{nullptr};
	std::unique_ptr<float[]> synWscalerGRtoGO{nullptr};
	std::unique_ptr<uint8_t[]> apGO{nullptr};
	std::unique_ptr<uint32_t[]> apBufGO{nullptr};
	std::unique_ptr<float[]> vGO{nullptr};
	std::unique_ptr<float[]> vCoupleGO{nullptr};
	std::unique_ptr<float[]> threshCurGO{nullptr};

	std::unique_ptr<uint32_t[]> inputMFGO{nullptr};
	std::unique_ptr<float[]> depAmpMFGO{nullptr};
	std::unique_ptr<float[]> gi_MFtoGO{nullptr};
	std::unique_ptr<float[]> gSum_MFGO{nullptr};
	std::unique_ptr<float[]> inputGOGO{nullptr};

	std::unique_ptr<float[]> gi_GOtoGO{nullptr};
	std::unique_ptr<float[]> depAmpGOGO{nullptr};
	std::unique_ptr<float[]> gSum_GOGO{nullptr};
	std::unique_ptr<float[]> depAmpGOGR{nullptr};
	std::unique_ptr<float[]> dynamicAmpGOGR{nullptr};

	//NOTE: removed NMDA UBC GO conductance 06/15/2022
	std::unique_ptr<float[]> gNMDAMFGO{nullptr};
	std::unique_ptr<float[]> gNMDAIncMFGO{nullptr};
	std::unique_ptr<float[]> gGRGO{nullptr};
	std::unique_ptr<float[]> gGRGO_NMDA{nullptr};

	//granule cells
	std::unique_ptr<float[]> depAmpMFGR{nullptr};
	std::unique_ptr<uint8_t[]> apGR{nullptr}; // <- pulled via getGPUData
	std::unique_ptr<uint32_t[]> apBufGR{nullptr};
	// NOTE: gMFGR was 2D array, now 1D for smart ptr
	// access using indices 0 <= i < NUM_GR, 0 <= j < MAX_NUM_P_GR_FROM_MF_TO_GR like so:
	// i * NUM_GR + j
	std::unique_ptr<float[]> gMFGR{nullptr};
	std::unique_ptr<float[]> gMFSumGR{nullptr};
	// NOTE: removed gMFDirectGR and gMFSpilloverGR as we only used to 
	// initialize GPU vars of :similar: name
	// also removed gGODirectGR and gGOSpillover
	std::unique_ptr<float[]> apMFtoGR{nullptr};
	// removed gNMDA, gNMDAIncGR, gLeakGR, depAmpMFtoGR, dynamicAmpGOtoGR as were
	// only used to initialize gpu vars

	// NOTE: gGOGR used to be 2D array with dims NUM_GR and MAX_NUM_P_GR_FROM_GO_TO_GR
	std::unique_ptr<float[]> gGOGR{nullptr};
	std::unique_ptr<float[]> gGOSumGR{nullptr};
	std::unique_ptr<float[]> threshGR{nullptr};
	std::unique_ptr<float[]> vGR{nullptr};
	std::unique_ptr<float[]> gKCaGR{nullptr};
	std::unique_ptr<uint64_t[]> historyGR{nullptr};

private:
	void stateRW(bool read, std::fstream &file);
	void allocateMemory();
	void initializeVals();
};

#endif /* INNETACTIVITYSTATE_H_ */

