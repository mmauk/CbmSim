#include <string>
#include "simulation.h"

//Simulation::Simulation(ActivityParams &ap)
//{
//	std::cout << "Initializing state..." << std::endl;	
//	state = new CBMState(ap, 1);
//	std::cout << "finished initializing state..." << std::endl;	
//
//	std::cout << "Initializing simulation..." << std::endl;
//	sim = new CBMSimCore(ap, state, gpuIndex, gpuP2);
//	std::cout << "Finished initializing simulation." << std::endl;
//}
//// TODO: check this!
//Simulation::~Simulation()
//{
//	delete state;
//	delete sim;
//}
//
//// CRITICAL: caller needs to free memory for MFFreq
//ECMFPopulation* Simulation::getMFFreq(float csMinRate, float csMaxRate)
//{
//	bool collaterals_off = false;
//	float fracImport     = 0.0;
//	bool secondCS        = true;
//	float fracOverlap    = 0.2;
//
//	MFFreq = new ECMFPopulation(NUM_MF, randseed, CStonicMFfrac,
//		CSphasicMFfrac, contextMFfrac, nucCollfrac, bgFreqMin,
//		csbgFreqMin, contextFreqMin, csMinRate, phasicFreqMin,
//		bgFreqMax, csbgFreqMax, contextFreqMax, csMaxRate,
//		phasicFreqMax, collaterals_off, fracImport, secondCS, fracOverlap); 
//
//	return MFFreq;
//}
//
//// CRITICAL: caller needs to free memory for MF
//PoissonRegenCells* Simulation::getMFs(float msPerTimeStep)
//{
//	MFs = new PoissonRegenCells(NUM_MF, randseed, threshDecayTau,
//		msPerTimeStep, numMZs, NUM_NC);
//
//	return MFs;
//}

