#include <string>
#include "setsim.h"

SetSim::SetSim(ActivityParams *ap)
{
    // printing should be handled in main, or we send to temp buffer,
	// which we run through stdout in main
	
	std::cout << "Initializing state..." << std::endl;	
	state = new CBMState(ap, 1);
	std::cout << "finished initializing state..." << std::endl;	

	std::cout << "Initializing simulation..." << std::endl;
	sim = new CBMSimCore(ap, state, gpuIndex, gpuP2);
	std::cout << "Finished initializing simulation." << std::endl;
};

SetSim::~SetSim(){};

CBMState* SetSim::getstate()
{
	return state;
}

CBMSimCore *SetSim::getsim()
{
	return sim;
}	

ECMFPopulation* SetSim::getMFFreq(float csMinRate, float csMaxRate)
{
	bool collaterals_off = false;
	float fracImport     = 0.0;
	bool secondCS        = true;
	float fracOverlap    = 0.2;

	MFFreq = new ECMFPopulation(NUM_MF, randseed, CStonicMFfrac,
		CSphasicMFfrac, contextMFfrac, nucCollfrac, bgFreqMin,
		csbgFreqMin, contextFreqMin, csMinRate, phasicFreqMin,
		bgFreqMax, csbgFreqMax, contextFreqMax, csMaxRate,
		phasicFreqMax, collaterals_off, fracImport, secondCS, fracOverlap); 

	return MFFreq;
}

PoissonRegenCells* SetSim::getMFs(ActivityParams *ap)
{
	MFs = new PoissonRegenCells(NUM_MF, randseed, threshDecayTau,
		ap->msPerTimeStep, numMZs, NUM_NC);

	return MFs;
}

