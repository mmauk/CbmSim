#include <string>
#include "setsim.h"

// simple template function which acts as a string interpolator
// courtesy of wcochran: https://stackoverflow.com/questions/63121776/
// simplest-syntax-for-string-interpolation-in-c
template<typename... Args>
std::string Sprintf(const char *fmt, Args... args)
{
    
	const size_t n = snprintf(nullptr, 0, fmt, args...);
	std::vector<char> buf(n+1);
	snprintf(buf.data(), n+1, fmt, args...);
	return std::string(buf.data());
}

SetSim::SetSim(int fileNum, int goRecipParam, int simNum)
{

    // Not sure how we want our act files to be named	
	std::string inActFile = Sprintf("./data/actParams_binChoice2_%d.txt", fileNum+1);	
	std::string inConFile = Sprintf("./data/conParams_binChoice2_%d.txt", fileNum+1);	

	std::fstream actPF(inActFile.c_str());
	std::fstream conPF(inConFile.c_str());
    
    // printing should be handled in main, or we send to temp buffer,
	// which we run through stdout in main
	std::cout << "Opening file...(and initializing state)" << std::endl;
	
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
	unsigned int numMF   = state->getConnectivityParams()->getNumMF(); 		
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
	unsigned int numMF  = state->getConnectivityParams()->getNumMF(); 		
	unsigned int numNC  = state->getConnectivityParams()->getNumNC();
	float msPerTimeStep = state->getActivityParams()->getMSPerTimeStep() * 1.0;

	MFs = new PoissonRegenCells(						 
		numMF, randseed, threshDecayTau, msPerTimeStep, numMZs, numNC);

	return MFs;
}

