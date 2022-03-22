#include <string>
#include "setsim.h"


// simple template function which acts as a string interpolator
// courtesy of wcochran: https://stackoverflow.com/questions/63121776/simplest-syntax-for-string-interpolation-in-c
template<typename... Args>
string Sprintf(const char *fmt, Args... args) {
    
	const size_t n = snprintf(nullptr, 0, fmt, args...);
	vector<char> buf(n+1);
	snprintf(buf.data(), n+1, fmt, args...);
	return string(buf.data());
}

SetSim::SetSim(int fileNum, int goRecipParam, int simNum){

    // Not sure how we want our act files to be named	
    string inActFile = Sprintf("./resources/actParams_binChoice2_%d.txt", fileNum+1);	
    string inConFile = Sprintf("./resources/conParams_binChoice2_%d.txt", fileNum+1);	

	fstream actPF(inActFile.c_str());
	fstream conPF(inConFile.c_str());
    
    // printing should be handled in main, or we send to temp buffer, which we run through stdout in main
    cout << "Opening file...(and initializing state)" << endl;
	
	this->state = new CBMState(actPF, conPF, 1, goRecipParam, simNum);
	
	cout << "finished opening file...(and initializing state)" << endl;	
	
	actPF.close();
	conPF.close();
};


SetSim::~SetSim(){};


CBMState* SetSim::getstate(){
	return state;
}


CBMSimCore *SetSim::getsim(){
	
	sim = new CBMSimCore(state, gpuIndex, gpuP2);
	return sim;
}	


ECMFPopulation* SetSim::getMFFreq(float csMinRate, float csMaxRate){
	unsigned int numMF = state->getConnectivityParams()->getNumMF(); 		
	bool collaterals_off = false;
	float fracImport = 0.0;
	bool secondCS = true;
	float fracOverlap = 0.2;

	MFFreq = new ECMFPopulation(							
		numMF, randseed, 
		CStonicMFfrac, CSphasicMFfrac, contextMFfrac, nucCollfrac,
		bgFreqMin, csbgFreqMin, contextFreqMin, csMinRate, phasicFreqMin,
		bgFreqMax, csbgFreqMax, contextFreqMax, csMaxRate, phasicFreqMax,
		collaterals_off, fracImport, secondCS, fracOverlap); 

	return MFFreq;
}


PoissonRegenCells* SetSim::getMFs(){
	unsigned int numMF = state->getConnectivityParams()->getNumMF(); 		
	unsigned int numNC = state->getConnectivityParams()->getNumNC();
	float msPerTimeStep = state->getActivityParams()->getMSPerTimeStep() * 1.0;

	MFs = new PoissonRegenCells(						 
		numMF, randseed, threshDecayTau, msPerTimeStep, numMZs, numNC);

	return MFs;
}





