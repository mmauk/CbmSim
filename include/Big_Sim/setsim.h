#ifndef SETSIM_H_
#define SETSIM_H_

#include <iostream> 
#include <fstream>
#include <string>
#include "connectivityparams.h"
#include <interfaces/cbmstate.h>
#include <interfaces/iactivityparams.h>
#include <interface/cbmsimcore.h>
#include <ecmfpopulation.h>
#include <poissonregencells.h>
#include <eyelidintegrator.h>

class SetSim {

public:
	// yikes. for now, pass actParams as a pointer by value while we fix conParams (06/01/2022)
	SetSim(ConnectivityParams &conParams, ActivityParams *actParams, int goRecipParam, int simNum);
	~SetSim();

	// State	
	int numMZs      = 1;	
	int innetCRSeed = 1;
	int *mzoneCRSeed;
	int *mzoneARSeed;
	
	CBMState* state;
	CBMState* getstate();
	void delstate();	
	
	//Sim
	int gpuIndex = 0;
	int gpuP2    = 2;
	CBMSimCore* sim;
	CBMSimCore* getsim();

	// MF parameters
	float threshDecayTau = 4.0;	
	int randseed         = 3;

	float nucCollfrac = 0.02;

	// Basic Mossy Fibers
	float CStonicMFfrac = 0.05;
	float tonicFreqMin  = 100.0;
	float tonicFreqMax  = 110.0;

	float CSphasicMFfrac = 0.00;
	float phasicFreqMin  = 200.0;
	float phasicFreqMax  = 250.0;
	
	float contextMFfrac  = 0.0;
	float contextFreqMin = 20.0;
	float contextFreqMax = 50.0;

	float bgFreqMin   = 10.0;
	float csbgFreqMin = 10.0;
	float bgFreqMax   = 30.0;
	float csbgFreqMax = 30.0;
	
	
	ECMFPopulation* MFFreq;
	ECMFPopulation* getMFFreq(float csMineRate, float csMaxRate);

	PoissonRegenCells* MFs;
	PoissonRegenCells* getMFs();

	// Eyelid integrator 
	float gDecayTau = 11;
	float gIncrease = 0.012;
	float gShift = -0.1;
	float gLeakRaw = 0.1;
	float maxAmplitude = 100;
};

#endif

