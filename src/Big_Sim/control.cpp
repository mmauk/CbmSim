#include <time.h>
#include "control.h"

Control::Control() {}

Control::Control(std::string actParamFile) : ap(actParamFile)
{
	std::cout << "[INFO]: Initializing state..." << std::endl;	
	simState = new CBMState(ap, numMZones);
	std::cout << "[INFO]: Finished initializing state..." << std::endl;
	
	std::cout << "[INFO]: Initializing simulation core..." << std::endl;
	simCore = new CBMSimCore(ap, simState, gpuIndex, gpuP2);	
	std::cout << "[INFO]: Finished initializing simulation core." << std::endl;

	std::cout << "[INFO]: Initializing MF Frequencies..." << std::endl;
	mfFreq = new ECMFPopulation(NUM_MF, mfRandSeed, CSTonicMFFrac, CSPhasicMFFrac,
		  contextMFFrac, nucCollFrac, bgFreqMin, csbgFreqMin, contextFreqMin, 
		  tonicFreqMin, phasicFreqMin, bgFreqMax, csbgFreqMax, contextFreqMax, 
		  tonicFreqMax, phasicFreqMax, collaterals_off, fracImport, secondCS, fracOverlap);
	std::cout << "[INFO]: Finished initializing MF Frequencies." << std::endl;

	std::cout << "[INFO]: Initializing Poisson MF Population..." << std::endl;
	mfs = new PoissonRegenCells(NUM_MF, mfRandSeed, threshDecayTau, ap.msPerTimeStep,
		  	numMZones, NUM_NC);
	std::cout << "[INFO]: Finished initializing Poisson MF Population." << std::endl;

	// allocate and initialize output arrays
	std::cout << "[INFO]: Initializing output arrays..." << std::endl;
	initializeOutputArrays(csLength, numTrainingTrials);	
	std::cout << "[INFO]: Finished initializing output arrays." << std::endl;
}

Control::~Control()
{
	// delete all dynamic objects
	delete simState;
	delete simCore;
	delete mfFreq;
	delete mfs;

	// deallocate output arrays
	deleteOutputArrays();
}

void Control::initializeOutputArrays(int csLength, int numTrainingTrials)
{
	int allGOPSTHColSize = csLength + msPreCS + msPostCS;
	int rasterColumnSize = allGOPSTHColSize * numTrainingTrials;	

	// Allocate and Initialize PSTH and Raster arrays
	allPCRaster = allocate2DArray<ct_uint8_t>(NUM_PC, rasterColumnSize);	
	std::fill(allPCRaster[0], allPCRaster[0] +
			NUM_PC * rasterColumnSize, 0);
	
	allNCRaster = allocate2DArray<ct_uint8_t>(NUM_NC, rasterColumnSize);	
	std::fill(allNCRaster[0], allNCRaster[0] +
			NUM_NC * rasterColumnSize, 0);

	allSCRaster = allocate2DArray<ct_uint8_t>(NUM_SC, rasterColumnSize);	
	std::fill(allSCRaster[0], allSCRaster[0] +
			NUM_SC * rasterColumnSize, 0);

	allBCRaster = allocate2DArray<ct_uint8_t>(NUM_BC, rasterColumnSize);	
	std::fill(allBCRaster[0], allBCRaster[0] +
			NUM_BC * rasterColumnSize, 0);
	
	allGOPSTH = allocate2DArray<ct_uint8_t>(NUM_GO, allGOPSTHColSize);	
	std::fill(allGOPSTH[0], allGOPSTH[0] + NUM_GO * allGOPSTHColSize, 0);
}

void Control::runTrials(int simNum, float GOGR, float GRGO, float MFGO)
{
	int preTrialNumber   = homeoTuningTrials + granuleActDetectTrials;
	int numTotalTrials   = preTrialNumber + numTrainingTrials;  

	float medTrials;
	clock_t timer;

	int rasterCounter = 0;
	int goSpkCounter[NUM_GO] = {0};

	for (int trial = 0; trial < numTotalTrials; trial++)
	{
		timer = clock();
		
		// re-initialize spike counter vector	
		std::fill(goSpkCounter, goSpkCounter + NUM_GO, 0);	

		int PSTHCounter = 0;	
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;

		trial <= homeoTuningTrials ?
			std::cout << "Pre-tuning trial number: " << trial << std::endl :
			std::cout << "Post-tuning trial number: " << trial << std::endl;
		
		// Homeostatic plasticity trials
		if (trial >= homeoTuningTrials)
		{	
		
			for (int tts = 0; tts < trialTime; tts++)
			{			
				// TODO: get the model for these periods, update accordingly
				if (tts == csStart + csLength)
				{
		   		   // Deliver US 
				   // why zoneN and errDriveRelative set manually???
					simCore->updateErrDrive(0, 0.0);
				}
				
				if (tts < csStart || tts >= csStart + csLength)
				{
					// Background MF activity in the Pre and Post CS period
					mfAP = mfs->calcPoissActivity(mfFreq->getMFBG(),
							simCore->getMZoneList());	
				}
				else if (tts >= csStart && tts < csStart + csPhasicSize) 
				{
					// Phasic MF activity during the CS for a duration set in control.h 
					mfAP = mfs->calcPoissActivity(mfFreq->getMFFreqInCSPhasic(),
							simCore->getMZoneList());
				}
				else
				{
					// Tonic MF activity during the CS period
					// this never gets reached...
					mfAP = mfs->calcPoissActivity(mfFreq->getMFInCSTonicA(),
							simCore->getMZoneList());
				}

				bool *isTrueMF = mfs->calcTrueMFs(mfFreq->getMFBG());
				simCore->updateTrueMFs(isTrueMF);
				simCore->updateMFInput(mfAP);
				simCore->calcActivity(goMin, simNum, GOGR, GRGO, MFGO, gogoW, spillFrac);	
				
				if (tts >= csStart && tts < csStart + csLength)
				{

					mfgoG  = simCore->getInputNet()->exportgSum_MFGO();
					grgoG  = simCore->getInputNet()->exportgSum_GRGO();
					goSpks = simCore->getInputNet()->exportAPGO();
					
					//TODO: change for loop into std::transform
					for (int i = 0; i < NUM_GO; i++)
					{
							goSpkCounter[i] += goSpks[i];
							gGRGO_sum 		+= grgoG[i];
							gMFGO_sum 		+= mfgoG[i];
					}
				}
				// why is this case separate from the above	
				if (tts == csStart + csLength)
				{
					countGOSpikes(goSpkCounter, medTrials);	
					std::cout << "mean gGRGO   = " << gGRGO_sum / (NUM_GO * csLength) << std::endl;
					std::cout << "mean gMFGO   = " << gMFGO_sum / (NUM_GO * csLength) << std::endl;
					std::cout << "GR:MF ratio  = " << gGRGO_sum / gMFGO_sum << std::endl;
				}

				if (trial >= preTrialNumber && tts >= csStart-msPreCS &&
						tts < csStart + csLength + msPostCS)
				{
					fillRasterArrays(simCore, rasterCounter);

					PSTHCounter++;
					rasterCounter++;
				}
			}
		}
		timer = clock() - timer;
		std::cout << "Trial time seconds: " << (float)timer / CLOCKS_PER_SEC << std::endl;
	}
}

void Control::saveOutputArraysToFile(int goRecipParam, int simNum)
{
	int allGOPOSTHColSize = csLength + msPreCS + msPostCS; 
	// array of possible convergences to test. 
	// TODO: put these into an input file instead of here :weird_champ:
	int conv[8] = {5000, 4000, 3000, 2000, 1000, 500, 250, 125};
	
	std::string allGOPSTHFileName = "allGOPSTH_noGOGO_grgoConv" + std::to_string(conv[goRecipParam]) +
		"_" + std::to_string(simNum) + ".bin";	
	write2DCharArray(allGOPSTHFileName, allGOPSTH, NUM_GO, allGOPOSTHColSize);

	std::cout << "Filling BC files" << std::endl;
	
	std::string allBCRasterFileName = "allBCRaster_paramSet" + std::to_string(inputStrength) +
		"_" + std::to_string(simNum) + ".bin";
	write2DCharArray(allBCRasterFileName, allBCRaster, NUM_BC,
			numTrainingTrials * allGOPOSTHColSize);
	
	std::cout << "Filling SC files" << std::endl;

	std::string allSCRasterFileName = "allSCRaster_paramSet" + std::to_string(inputStrength) +
		"_" + std::to_string(simNum) + ".bin";
	write2DCharArray(allSCRasterFileName, allSCRaster, NUM_SC,
			numTrainingTrials * allGOPOSTHColSize);
}

void Control::countGOSpikes(int *goSpkCounter, float &medTrials)
{
	std::sort(goSpkCounter, goSpkCounter + 4096);
	
	float m = (goSpkCounter[2047] + goSpkCounter[2048]) / 2.0;
	float goSpkSum = 0;

	//TODO: change for loop into std::transform
	for (int i = 0; i < NUM_GO; i++) goSpkSum += goSpkCounter[i];
	
	std::cout << "Mean GO Rate: " << goSpkSum / (float)NUM_GO << std::endl;

	medTrials += m / 2.0;
	std::cout << "Median GO Rate: " << m / 2.0 << std::endl;
}

void Control::fillRasterArrays(CBMSimCore *simCore, int rasterCounter)
{
	const ct_uint8_t* pcSpks = simCore->getMZoneList()[0]->exportAPPC();
	const ct_uint8_t* ncSpks = simCore->getMZoneList()[0]->exportAPNC();
	const ct_uint8_t* bcSpks = simCore->getMZoneList()[0]->exportAPBC();
	const ct_uint8_t* scSpks = simCore->getInputNet()->exportAPSC();
	
	for (int i = 0; i < NUM_PC; i++)
	{
		allPCRaster[i][rasterCounter] = pcSpks[i];
	}

	for (int i = 0; i < NUM_NC; i++)
	{
		allNCRaster[i][rasterCounter] = ncSpks[i];
	}

	for (int i = 0; i < NUM_BC; i++)
	{
		allBCRaster[i][rasterCounter] = bcSpks[i];
	}

	for (int i = 0; i < NUM_SC; i++)
	{
		allSCRaster[i][rasterCounter] = scSpks[i];
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
	delete2DArray<ct_uint8_t>(allPCRaster);
	delete2DArray<ct_uint8_t>(allNCRaster);
	delete2DArray<ct_uint8_t>(allSCRaster);
	delete2DArray<ct_uint8_t>(allBCRaster);
	delete2DArray<ct_uint8_t>(allGOPSTH);
}

