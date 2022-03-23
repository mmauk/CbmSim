#include <string>
#include <time.h>
#include "control.h"

Control::Control(){};

Control::~Control(){};

void Control::runSimulationWithGRdata(int fileNum, int goRecipParam, int numTuningTrials,
		int numGrDetectionTrials, int numTrainingTrials, int simNum, int csSize, float csFracMFs,
		float goMin, float GOGR, float GRGO, float MFGO, float csMinRate, float csMaxRate,
		float gogoW, int inputStrength, int inputWeight_two, float spillFrac)
{
	std::cout << "fileNum: " << fileNum << std::endl;
	
	SetSim simulation(fileNum, goRecipParam, simNum);
	joestate = simulation.getstate();
	joesim = simulation.getsim();
	joeMFFreq = simulation.getMFFreq(csMinRate, csMaxRate);
	joeMFs = simulation.getMFs();	

	int numTotalTrials = numTuningTrials + numGrDetectionTrials + numTrainingTrials;  
	int preTrialNumber = numTuningTrials + numGrDetectionTrials;
	int collectionTrials = numTotalTrials;
	
	std::cout << "Done filling MF arrays" << std::endl;	

	int conv[8] = {5000, 4000, 3000, 2000, 1000, 500, 250, 125};

	// Allocate and Initialize PSTH and Raster arrays
	allPCRaster = allocate2DArray<ct_uint8_t>(numPC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allPCRaster[0], 0, numPC*(csSize+msPreCS+msPostCS)*(collectionTrials));
	
	allNCRaster = allocate2DArray<ct_uint8_t>(numNC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allNCRaster[0], 0, numNC*(csSize+msPreCS+msPostCS)*(collectionTrials));

	allSCRaster = allocate2DArray<ct_uint8_t>(numSC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allSCRaster[0], 0, numSC*(csSize+msPreCS+msPostCS)*(collectionTrials));

	allBCRaster = allocate2DArray<ct_uint8_t>(numBC, (csSize+msPreCS+msPostCS)*(collectionTrials));	
	arrayInitialize<ct_uint8_t>(allBCRaster[0], 0, numBC*(csSize+msPreCS+msPostCS)*(collectionTrials));
	
	std::cout << "PC arrays" << std::endl;	

	allGOPSTH = allocate2DArray<ct_uint8_t>(numGO, (csSize+msPreCS+msPostCS));	
	arrayInitialize<ct_uint8_t>(allGOPSTH[0], 0, numGO*(csSize+msPreCS+msPostCS));

	float medTrials;
	float *mTall    = new float[numTotalTrials];
	float *grgoGall = new float[numTotalTrials];
	float *mfgoGall = new float[numTotalTrials];
	
	clock_t timer;
	
	int trialTime     = 5000; // in milliseconds, i think
	int rasterCounter = 0;
	
	std::vector<int> goSpkCounter(numGO);
	
	for (int trial = 0; trial < numTotalTrials; trial++)
	{
		timer = clock();

		float gGRGO_sum = 0;
		float gMFGO_sum = 0;

		if (trial <= numTuningTrials)
	   	{
			std::cout << "Pre-tuning trial number: " << trial << std::endl;
		}
		else 
		{
			std::cout << "Post-tuning trial number: " << trial << std::endl;
		}
		
		int PSTHCounter = 0;	

		// Homeostatic plasticity trials

		if (trial >= numTuningTrials)
		{
			// Run active granule cell detection 	
			if (trial == preTrialNumber)
			{			
				for (tts = 0; tts < trialTime; tts++)
				{	
					// TODO: get the model for these periods, update accordingly
					if (tts == csStart + csSize)
					{
						// Deliver US 
						joesim->updateErrDrive(0,0.0);
					}
					
					if (tts < csStart || tts >= csStart + csSize)
					{
						// Background MF activity in the Pre and Post CS period
						mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFBG(), joesim->getMZoneList());	
					}
					else if (tts >= csStart && tts < csStart + csPhasicSize) 
					{
						// Phasic MF activity during the CS for a duration set in control.h 
						mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFFreqInCSPhasic(), joesim->getMZoneList());
					}
					else
					{
						// Tonic MF activity during the CS period
						// this never gets reached...
						mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFInCSTonicA(), joesim->getMZoneList());
					}

					bool *isTrueMF = joeMFs->calcTrueMFs(joeMFFreq->getMFBG());
					joesim->updateTrueMFs(isTrueMF);
					joesim->updateMFInput(mfAP);
					joesim->calcActivity(goMin, simNum, GOGR, GRGO, MFGO, gogoW, spillFrac);	
					
					if (tts >= csStart && tts < csStart + csSize)
					{

						mfgoG = joesim->getInputNet()->exportgSum_MFGO();
						grgoG = joesim->getInputNet()->exportgSum_GRGO();
						goSpks = joesim->getInputNet()->exportAPGO();
						for (int i = 0; i < numGO; i++)
						{
								goSpkCounter[i] += goSpks[i];
								gGRGO_sum += grgoG[i];
								gMFGO_sum += mfgoG[i];
						}
					}
					
					// why is this case separate from the above	
					if (tts == csStart + csSize)
					{
						std::sort(goSpkCounter.begin(), goSpkCounter.begin() + 4096);
						
						int m = (goSpkCounter[2047] + goSpkCounter[2048]) / 2.0;
						float goSpkSum = 0;
						
						for (int i = 0; i < numGO; i++)
						{
								goSpkSum += goSpkCounter[i];
						}
						
						std::cout << "Mean GO Rate: " << goSpkSum / (float)numGO << std::endl;

						medTrials += m / 2.0;
						std::cout << "Median GO Rate: " << m / 2.0 << std::endl;

						mTall[trial] = m / 2.0;
						std::cout << "mean gGRGO   = " << gGRGO_sum / (numGO * csSize) << std::endl;
						std::cout << "mean gMFGO   = " << gMFGO_sum / (numGO * csSize) << std::endl;
						std::cout << "GR:MF ratio  = " << gGRGO_sum / gMFGO_sum << std::endl;

						grgoGall[trial] = gGRGO_sum / (numGO * csSize);
						mfgoGall[trial] = gMFGO_sum / (numGO * csSize);

					}
					
					if (trial >= preTrialNumber && tts >= csStart-msPreCS && tts < csStart + csSize + msPostCS)
					{
						//PKJ
						const ct_uint8_t* pcSpks=joesim->getMZoneList()[0]->exportAPPC();
						for (int i = 0; i < numPC; i++)
						{
							allPCRaster[i][rasterCounter] = pcSpks[i];
						}
						
						//NC
						const ct_uint8_t* ncSpks = joesim->getMZoneList()[0]->exportAPNC();
						for (int i = 0; i < numNC; i++)
						{
							allNCRaster[i][rasterCounter] = ncSpks[i];
						}
						
						//BC
						const ct_uint8_t* bcSpks=joesim->getMZoneList()[0]->exportAPBC();
						for (int i = 0; i < numBC; i++){
							allBCRaster[i][rasterCounter] = bcSpks[i];
						}

						//SC
						const ct_uint8_t* scSpks=joesim->getInputNet()->exportAPSC();
						for (int i=0; i < numSC; i++)
						{
							allSCRaster[i][rasterCounter] = scSpks[i];
						}

						PSTHCounter++;
						rasterCounter++;
					}

				}

		 	}
		}	
		// re-initialize spike counter vector	
		goSpkCounter.assign(numGO, 0);	
		timer = clock() - timer;
		std::cout << "Trial time seconds: " << (float)timer / CLOCKS_PER_SEC << std::endl;
	}
	
	delete joestate;
	delete joesim;
	delete joeMFFreq;
	delete joeMFs;

	// Save Data 
	std::ofstream myfilegogoGbin("allGOPSTH_noGOGO_grgoConv" + std::to_string(conv[goRecipParam]) + 
			"_" + std::to_string(simNum) + ".bin", std::ios::out | std::ios::binary);	
	
	for (int i = 0; i < numGO; i++)
	{
		for (int j = 0; j < (csSize + msPreCS + msPostCS); j++)
		{
			myfilegogoGbin.write((char*) &allGOPSTH[i][j], sizeof(ct_uint8_t));
		}
	}

	myfilegogoGbin.close();
	delete2DArray<ct_uint8_t>(allGOPSTH);
	
	std::cout << "Filling BC files" << std::endl;
	
	std::ofstream myfileBCbin("allBCRaster_paramSet" + std::to_string(inputStrength) +
			"_" + std::to_string(simNum) + ".bin", std::ios::out | std::ios::binary);	
	
	for (int i = 0; i < numBC; i++)
	{
		for (int j = 0; j < (numTotalTrials - preTrialNumber) * (csSize + msPreCS + msPostCS); j++)
		{
			myfileBCbin.write((char*) &allBCRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	
	myfileBCbin.close();
	delete2DArray<ct_uint8_t>(allBCRaster);

	std::cout << "Filling SC files" << std::endl;
	std::ofstream myfileSCbin("allSCRaster_paramSet" + std::to_string(inputStrength) +
			"_" + std::to_string(simNum) + ".bin", std::ios::out | std::ios::binary);	
	
	for (int i = 0; i < numSC; i++)
	{
		for (int j = 0; j < (numTotalTrials - preTrialNumber) * (csSize + msPreCS + msPostCS); j++)
		{
			myfileSCbin.write((char*) &allSCRaster[i][j], sizeof(ct_uint8_t));
		}
	}
	
	myfileSCbin.close();
	delete2DArray<ct_uint8_t>(allSCRaster);
}

int* Control::getGRIndicies(float CStonicMFfrac) 
{

	int numMF = 4096;
	int numGR = 1048576;

	bool* tonicMFsA = joeMFFreq->getTonicMFInd();
	bool* tonicMFsB = joeMFFreq->getTonicMFIndOverlap();
	
	int numTonic = numMF*CStonicMFfrac; 
	int numActiveMFs = numTonic;

	std::cout << "Number of CS MossyFibers:	" << numActiveMFs << std::endl;
	
	int *activeMFIndA = new int[numActiveMFs];
	int *activeMFIndB = new int[numActiveMFs];
	
	int counterMFA=0;
	int counterMFB=0;
	
	for (int i = 0; i < numMF; i++)
	{	
		if (tonicMFsA[i])
		{
			activeMFIndA[counterMFA] = i;
			counterMFA++;
		}
		
		if (tonicMFsB[i])
		{
			activeMFIndB[counterMFB] = i;
			counterMFB++;
		}
	}

	std::cout << "NumMFs in A: " << counterMFA << std::endl;
	std::cout << "NumMFs in B: " << counterMFB << std::endl;
	
	std::vector<int> MFtoGRs;	
	int numPostSynGRs;
	// why is this labelled a bool when its an int array?
	int *pActiveGRsBool = new int[numGR]();
	for (int i = 0; i < numActiveMFs; i++)
	{
		MFtoGRs = joestate->getInnetConStateInternal()->getpMFfromMFtoGRCon(activeMFIndA[i]);
		numPostSynGRs = MFtoGRs.size();
		
		for (int j = 0; j < numPostSynGRs; j++)
		{
			pActiveGRsBool[MFtoGRs[j]]++;
		}
	}
	
	int counterGR = 0;
	
	for (int i = 0; i < numGR; i++)
	{
		if (pActiveGRsBool[i] >= 1) counterGR++;
	}

	int *pActiveGRs = new int[counterGR](); 
	int counterAGR = 0;
	
	for (int i = 0; i < numGR; i++)
	{
		if (pActiveGRsBool[i] >= 1)
		{		
			pActiveGRs[counterAGR] = i;
			counterAGR++;
		}
	}	

	return pActiveGRs;
}

// NOTE: this function is basically the same as the above.
int Control::getNumGRIndicies(float CStonicMFfrac) 
{
	int numMF = 4096;
	int numGR = 1048576;

	float CSphasicMFfrac = 0.0;
	float contextMFfrac = 0.0;
	
	bool* contextMFs = joeMFFreq->getContextMFInd();
	bool* phasicMFs = joeMFFreq->getPhasicMFInd();
	bool* tonicMFs = joeMFFreq->getTonicMFInd();
	
	int numContext = numMF*contextMFfrac; 
	int numPhasic = numMF*CSphasicMFfrac; 
	int numTonic = numMF*CStonicMFfrac; 
	int numActiveMFs = numContext+numPhasic+numTonic;
	
	int *activeMFInd;
	activeMFInd = new int[numActiveMFs];
	
	int counterMF=0;
	for (int i = 0; i < numMF; i++)
	{	
		if (contextMFs[i] || tonicMFs[i] || phasicMFs[i])
		{
			activeMFInd[counterMF] = i;
			counterMF++;
		}
	}

	std::vector<int> MFtoGRs;	
	int numPostSynGRs;
	int pActiveGRsBool[numGR] = {};

	for (int i = 0; i < numActiveMFs; i++)
	{
		MFtoGRs = joestate->getInnetConStateInternal()->getpMFfromMFtoGRCon(activeMFInd[i]);
		numPostSynGRs = MFtoGRs.size();
		
		for (int j = 0; j < numPostSynGRs; j++)
		{
			pActiveGRsBool[MFtoGRs[j]]++;
		}
	}
	
	int counterGR = 0;
	for (int i = 0; i < numGR; i++)
	{
		if (pActiveGRsBool[i] >= 1)
		{		
			counterGR++;
		}
	}
	
	return counterGR;
}

