#include <string>
#include "setsim.h"

SetSim::SetSim(ConnectivityParams &conParams, ActivityParams *actParams, int goRecipParam, int simNum)
{
    // printing should be handled in main, or we send to temp buffer,
	// which we run through stdout in main
	
	state = new CBMState(actPF, conPF, 1, goRecipParam, simNum);
	
	std::cout << "finished opening file and initializing state..." << std::endl;	

	actPF.close();
	conPF.close();

	std::cout << "Initializing simulation..." << std::endl;
	
	sim = new CBMSimCore(state, gpuIndex, gpuP2);

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
	unsigned int numMF   = state->getConParamsInternal()->getNumMF(); 		
	bool collaterals_off = false;
	float fracImport     = 0.0;
	bool secondCS        = true;
	float fracOverlap    = 0.2;

	MFFreq = new ECMFPopulation(							
		numMF, randseed, 
		CStonicMFfrac, CSphasicMFfrac, contextMFfrac, nucCollfrac,
		bgFreqMin, csbgFreqMin, contextFreqMin, csMinRate, phasicFreqMin,
		bgFreqMax, csbgFreqMax, contextFreqMax, csMaxRate, phasicFreqMax,
		collaterals_off, fracImport, secondCS, fracOverlap); 

	return MFFreq;
}

PoissonRegenCells* SetSim::getMFs()
{
	unsigned int numMF  = state->getConParamsInternal()->getNumMF(); 		
	unsigned int numNC  = state->getConParamsInternal()->getNumNC();
	float msPerTimeStep = state->getActivityParams()->getMSPerTimeStep() * 1.0;

	MFs = new PoissonRegenCells(						 
		numMF, randseed, threshDecayTau, msPerTimeStep, numMZs, numNC);

	return MFs;
}

