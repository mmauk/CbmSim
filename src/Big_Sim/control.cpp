#include <time.h>
#include "control.h"

Control::Control(std::string actParamFile)
{
	actParams = new ActivityParams(actParamFile);
}

Control::~Control()
{
	delete actParams;
}

void Control::runSimulationWithGRdata(int goRecipParam, int numTuningTrials, int numGrDetectionTrials,
		int numTrainingTrials, int simNum, int csSize, float csFracMFs, float goMin, float GOGR,
		float GRGO, float MFGO, float csMinRate, float csMaxRate, float gogoW, int inputStrength, 
		float spillFrac)
{
	// set all relevant variables to the sim	
	SetSim simulation(conParams, actParams, goRecipParam, simNum);
	joestate  = simulation.getstate();
	joesim    = simulation.getsim();
	joeMFFreq = simulation.getMFFreq(csMinRate, csMaxRate);
	joeMFs    = simulation.getMFs();


	// allocate and fill all of the output arrays	
	initializeOutputArrays(numPC, numNC, numSC, numBC, numGO, csSize, numTrainingTrials);

	// run all trials of sim
	runTrials(joesim, joeMFs, joeMFFreq, numTuningTrials, numGrDetectionTrials, numTrainingTrials,
		simNum, csSize, goMin, GOGR, GRGO, MFGO, csMinRate, csMaxRate, gogoW, spillFrac);

	// Save Data 
	saveOutputArraysToFile(numGO, numBC, numSC, numTrainingTrials, csSize, goRecipParam,
		simNum, inputStrength);

	// deallocate output arrays
	deleteOutputArrays();
}

void Control::initializeOutputArrays(int numPC, int numNC, int numSC, int numBC, int numGO,
	int csSize, int numTrainingTrials)
{
	int allGOPSTHColSize = csSize + msPreCS + msPostCS;
	int rasterColumnSize = allGOPSTHColSize * numTrainingTrials;	

	// Allocate and Initialize PSTH and Raster arrays
	allPCRaster = allocate2DArray<ct_uint8_t>(numPC, rasterColumnSize);	
	std::fill(allPCRaster[0], allPCRaster[0] +
			numPC * rasterColumnSize, 0);
	
	allNCRaster = allocate2DArray<ct_uint8_t>(numNC, rasterColumnSize);	
	std::fill(allNCRaster[0], allNCRaster[0] +
			numNC * rasterColumnSize, 0);

	allSCRaster = allocate2DArray<ct_uint8_t>(numSC, rasterColumnSize);	
	std::fill(allSCRaster[0], allSCRaster[0] +
			numSC * rasterColumnSize, 0);

	allBCRaster = allocate2DArray<ct_uint8_t>(numBC, rasterColumnSize);	
	std::fill(allBCRaster[0], allBCRaster[0] +
			numBC * rasterColumnSize, 0);
	
	allGOPSTH = allocate2DArray<ct_uint8_t>(numGO, allGOPSTHColSize);	
	std::fill(allGOPSTH[0], allGOPSTH[0] + numGO * allGOPSTHColSize, 0);
}

void Control::runTrials(CBMSimCore *joesim, PoissonRegenCells *joeMFs, ECMFPopulation *joeMFFreq,
	int numTuningTrials, int numGrDetectionTrials, int numTrainingTrials, int simNum,
	int csSize, float goMin, float GOGR, float GRGO, float MFGO,
	float csMinRate, float csMaxRate, float gogoW, float spillFrac)
{
	int preTrialNumber   = numTuningTrials + numGrDetectionTrials;
	// numTotalTrials should be numTrainingTrials	
	int numTotalTrials   = preTrialNumber + numTrainingTrials;  

	float medTrials;
	clock_t timer;

	int rasterCounter = 0;
	int *goSpkCounter = new int[numGO];

	for (int trial = 0; trial < numTrainingTrials; trial++)
	{
		timer = clock();
		
		// re-initialize spike counter vector	
		std::fill(goSpkCounter, goSpkCounter + numGO, 0);	

		int PSTHCounter = 0;	
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;


		trial <= numTuningTrials ?
			std::cout << "Pre-tuning trial number: " << trial << std::endl :
			std::cout << "Post-tuning trial number: " << trial << std::endl;
		
		// Homeostatic plasticity trials
		if (trial >= numTuningTrials)
		{	
		
			for (tts = 0; tts < trialTime; tts++)
			{			
				// TODO: get the model for these periods, update accordingly
				if (tts == csStart + csSize)
				{
					// Deliver US 
					joesim->updateErrDrive(0, 0.0);
				}
				
				if (tts < csStart || tts >= csStart + csSize)
				{
					// Background MF activity in the Pre and Post CS period
					mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFBG(),
							joesim->getMZoneList());	
				}
				else if (tts >= csStart && tts < csStart + csPhasicSize) 
				{
					// Phasic MF activity during the CS for a duration set in control.h 
					mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFFreqInCSPhasic(),
							joesim->getMZoneList());
				}
				else
				{
					// Tonic MF activity during the CS period
					// this never gets reached...
					mfAP = joeMFs->calcPoissActivity(joeMFFreq->getMFInCSTonicA(),
							joesim->getMZoneList());
				}

				bool *isTrueMF = joeMFs->calcTrueMFs(joeMFFreq->getMFBG());
				joesim->updateTrueMFs(isTrueMF);
				joesim->updateMFInput(mfAP);
				joesim->calcActivity(goMin, simNum, GOGR, GRGO, MFGO, gogoW, spillFrac);	
				
				if (tts >= csStart && tts < csStart + csSize)
				{

					mfgoG  = joesim->getInputNet()->exportgSum_MFGO();
					grgoG  = joesim->getInputNet()->exportgSum_GRGO();
					goSpks = joesim->getInputNet()->exportAPGO();
					
					//TODO: change for loop into std::transform
					for (int i = 0; i < numGO; i++)
					{
							goSpkCounter[i] += goSpks[i];
							gGRGO_sum 		+= grgoG[i];
							gMFGO_sum 		+= mfgoG[i];
					}
				}
				// why is this case separate from the above	
				if (tts == csStart + csSize)
				{
					countGOSpikes(goSpkCounter, medTrials);	
					std::cout << "mean gGRGO   = " << gGRGO_sum / (numGO * csSize) << std::endl;
					std::cout << "mean gMFGO   = " << gMFGO_sum / (numGO * csSize) << std::endl;
					std::cout << "GR:MF ratio  = " << gGRGO_sum / gMFGO_sum << std::endl;
				}

				if (trial >= preTrialNumber && tts >= csStart-msPreCS &&
						tts < csStart + csSize + msPostCS)
				{
					fillRasterArrays(joesim, rasterCounter);

					PSTHCounter++;
					rasterCounter++;
				}
			}
		}
		timer = clock() - timer;
		std::cout << "Trial time seconds: " << (float)timer / CLOCKS_PER_SEC << std::endl;
	}

	delete[] goSpkCounter;
}

void Control::saveOutputArraysToFile(int numGO, int numBC, int numSC, int numTrainingTrials, int csSize,
	int goRecipParam, int simNum, int inputStrength)
{
	int allGOPOSTHColSize = csSize + msPreCS + msPostCS; 
	
	// array of possible convergences to test. 
	// TODO: put these into an input file instead of here :weird_champ:
	int conv[8] = {5000, 4000, 3000, 2000, 1000, 500, 250, 125};
	
	// TODO: once get matrix class, rewrite
	std::string allGOPSTHFileName = "allGOPSTH_noGOGO_grgoConv" + std::to_string(conv[goRecipParam]) +
		"_" + std::to_string(simNum) + ".bin";	
	write2DCharArray(allGOPSTHFileName, allGOPSTH, numGO, allGOPOSTHColSize);

	std::cout << "Filling BC files" << std::endl;
	
	std::string allBCRasterFileName = "allBCRaster_paramSet" + std::to_string(inputStrength) +
		"_" + std::to_string(simNum) + ".bin";
	write2DCharArray(allBCRasterFileName, allBCRaster, numBC,
			numTrainingTrials * allGOPOSTHColSize);
	
	std::cout << "Filling SC files" << std::endl;

	std::string allSCRasterFileName = "allSCRaster_paramSet" + std::to_string(inputStrength) +
		"_" + std::to_string(simNum) + ".bin";
	write2DCharArray(allSCRasterFileName, allSCRaster, numSC,
			numTrainingTrials * allGOPOSTHColSize);
}

void Control::countGOSpikes(int *goSpkCounter, float &medTrials)
{
	std::sort(goSpkCounter, goSpkCounter + 4096);
	
	float m = (goSpkCounter[2047] + goSpkCounter[2048]) / 2.0;
	float goSpkSum = 0;

	//TODO: change for loop into std::transform
	for (int i = 0; i < numGO; i++) goSpkSum += goSpkCounter[i];
	
	std::cout << "Mean GO Rate: " << goSpkSum / (float)numGO << std::endl;

	medTrials += m / 2.0;
	std::cout << "Median GO Rate: " << m / 2.0 << std::endl;
}

void Control::fillRasterArrays(CBMSimCore *joesim, int rasterCounter)
{
	const ct_uint8_t* pcSpks = joesim->getMZoneList()[0]->exportAPPC();
	const ct_uint8_t* ncSpks = joesim->getMZoneList()[0]->exportAPNC();
	const ct_uint8_t* bcSpks = joesim->getMZoneList()[0]->exportAPBC();
	const ct_uint8_t* scSpks = joesim->getInputNet()->exportAPSC();
	
	// TODO: yet another reason why an array that knows its size would be helpful!
	int maxCount = std::max({numPC, numNC, numBC, numSC});						

	for (size_t i = 0; i < maxCount; i++)
	{
		if (i < numPC)
		{
			allPCRaster[i][rasterCounter] = pcSpks[i];
		}
		if (i < numNC)
		{
			allNCRaster[i][rasterCounter] = ncSpks[i];
		}	
		if (i < numBC)
		{
			allBCRaster[i][rasterCounter] = bcSpks[i];
		}	
		if (i < numSC)
		{
			allSCRaster[i][rasterCounter] = scSpks[i];
		}	
	}	
}	

// TODO: 1) find better place to put this 2) generalize
void Control::write2DCharArray(std::string outFileName, ct_uint8_t **inArr,
	unsigned int numRow, unsigned int numCol)
{
	std::ofstream outStream(outFileName.c_str(), std::ios::out | std::ios::binary);

	if (!outStream.is_open())
	{
		// NOTE: should throw an error, which would be caught in main
		std::cerr << "couldn't open '" << outFileName << "' for writing." << std::endl;
		return;
	}	
		
	for (size_t i = 0; i < numRow; i++)
	{
		for (size_t j = 0; j < numCol; j++)
		{
			outStream.write((char*) &inArr[i][j], sizeof(ct_uint8_t));
		}
	}

	outStream.close();
}

void Control::deleteOutputArrays()
{
	delete2DArray<ct_uint8_t>(allGOPSTH);
	delete2DArray<ct_uint8_t>(allBCRaster);
	delete2DArray<ct_uint8_t>(allSCRaster);
}

//int* Control::getGRIndicies(CBMState *joestate, ECMFPopulation *joeMFFreq, float csMinRate, float csMaxRate, float CStonicMFfrac) 
//{
//
//	int numMF = 4096;
//	int numGR = 1048576;
//
//	bool* tonicMFsA = joeMFFreq->getTonicMFInd();
//	bool* tonicMFsB = joeMFFreq->getTonicMFIndOverlap();
//	
//	int numTonic = numMF*CStonicMFfrac; 
//	int numActiveMFs = numTonic;
//
//	std::cout << "Number of CS MossyFibers:	" << numActiveMFs << std::endl;
//	
//	int *activeMFIndA = new int[numActiveMFs];
//	int *activeMFIndB = new int[numActiveMFs];
//	
//	int counterMFA=0;
//	int counterMFB=0;
//	
//	for (int i = 0; i < numMF; i++)
//	{	
//		if (tonicMFsA[i])
//		{
//			activeMFIndA[counterMFA] = i;
//			counterMFA++;
//		}
//		
//		if (tonicMFsB[i])
//		{
//			activeMFIndB[counterMFB] = i;
//			counterMFB++;
//		}
//	}
//
//	std::cout << "NumMFs in A: " << counterMFA << std::endl;
//	std::cout << "NumMFs in B: " << counterMFB << std::endl;
//	
//	std::vector<int> MFtoGRs;	
//	int numPostSynGRs;
//	// why is this labelled a bool when its an int array?
//	int *pActiveGRsBool = new int[numGR]();
//	for (int i = 0; i < numActiveMFs; i++)
//	{
//		MFtoGRs = joestate->getInnetConStateInternal()->getpMFfromMFtoGRCon(activeMFIndA[i]);
//		numPostSynGRs = MFtoGRs.size();
//		
//		for (int j = 0; j < numPostSynGRs; j++)
//		{
//			pActiveGRsBool[MFtoGRs[j]]++;
//		}
//	}
//	
//	int counterGR = 0;
//	
//	for (int i = 0; i < numGR; i++)
//	{
//		if (pActiveGRsBool[i] >= 1) counterGR++;
//	}
//
//	int *pActiveGRs = new int[counterGR](); 
//	int counterAGR = 0;
//	
//	for (int i = 0; i < numGR; i++)
//	{
//		if (pActiveGRsBool[i] >= 1)
//		{		
//			pActiveGRs[counterAGR] = i;
//			counterAGR++;
//		}
//	}	
//
//	return pActiveGRs;
//}

// NOTE: this function is basically the same as the above.
//int Control::getNumGRIndicies(CBMState *joestate, ECMFPopulation *joeMFFreq, float csMinRate, float csMaxRate, float CStonicMFfrac) 
//{
//	int numMF = 4096;
//	int numGR = 1048576;
//
//	float CSphasicMFfrac = 0.0;
//	float contextMFfrac  = 0.0;
//	
//	bool* contextMFs = joeMFFreq->getContextMFInd();
//	bool* phasicMFs  = joeMFFreq->getPhasicMFInd();
//	bool* tonicMFs   = joeMFFreq->getTonicMFInd();
//	
//	int numContext 	 = numMF*contextMFfrac; 
//	int numPhasic  	 = numMF*CSphasicMFfrac; 
//	int numTonic   	 = numMF*CStonicMFfrac; 
//	int numActiveMFs = numContext+numPhasic+numTonic;
//	
//	int *activeMFInd;
//	activeMFInd = new int[numActiveMFs];
//	
//	int counterMF = 0;
//	for (int i = 0; i < numMF; i++)
//	{	
//		if (contextMFs[i] || tonicMFs[i] || phasicMFs[i])
//		{
//			activeMFInd[counterMF] = i;
//			counterMF++;
//		}
//	}
//
//	std::vector<int> MFtoGRs;	
//	int numPostSynGRs;
//	int pActiveGRsBool[numGR] = {};
//
//	for (int i = 0; i < numActiveMFs; i++)
//	{
//		MFtoGRs 	  = joestate->getInnetConStateInternal()->getpMFfromMFtoGRCon(activeMFInd[i]);
//		numPostSynGRs = MFtoGRs.size();
//		
//		for (int j = 0; j < numPostSynGRs; j++)
//		{
//			pActiveGRsBool[MFtoGRs[j]]++;
//		}
//	}
//	
//	int counterGR = 0;
//	for (int i = 0; i < numGR; i++)
//	{
//		if (pActiveGRsBool[i] >= 1)
//		{		
//			counterGR++;
//		}
//	}
//	
//	return counterGR;
//}

