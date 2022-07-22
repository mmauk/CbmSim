#include <iomanip>
#include <gtk/gtk.h>
#include "control.h"
#include "ttyManip/tty.h"

const std::string BIN_EXT = "bin";

// private utility function. Will move to a better place later
std::string getFileBasename(std::string fullFilePath)
{
	size_t sep = fullFilePath.find_last_of("\\/");
	if (sep != std::string::npos)
	    fullFilePath = fullFilePath.substr(sep + 1, fullFilePath.size() - sep - 1);
	
	size_t dot = fullFilePath.find_last_of(".");
	if (dot != std::string::npos)
	{
	    std::string name = fullFilePath.substr(0, dot);
	}
	else
	{
	    std::string name = fullFilePath;
	}
	return (dot != std::string::npos) ? fullFilePath.substr(0, dot) : fullFilePath;
}

Control::Control() {}

Control::Control(std::string actParamFile)
{
	if (!ap) ap = new ActivityParams(actParamFile);
	if (!simState)
	{
		std::cout << "[INFO]: Initializing state..." << std::endl;
		simState = new CBMState(ap, numMZones);
		std::cout << "[INFO]: Finished initializing state..." << std::endl;
	}
	
	if (!simCore)
	{
		std::cout << "[INFO]: Initializing simulation core..." << std::endl;
		simCore = new CBMSimCore(ap, simState, gpuIndex, gpuP2);
		std::cout << "[INFO]: Finished initializing simulation core." << std::endl;
	}

	if (!mfFreq)
	{
		std::cout << "[INFO]: Initializing MF Frequencies..." << std::endl;
		mfFreq = new ECMFPopulation(NUM_MF, mfRandSeed, CSTonicMFFrac, CSPhasicMFFrac,
			  contextMFFrac, nucCollFrac, bgFreqMin, csbgFreqMin, contextFreqMin, 
			  tonicFreqMin, phasicFreqMin, bgFreqMax, csbgFreqMax, contextFreqMax, 
			  tonicFreqMax, phasicFreqMax, collaterals_off, fracImport, secondCS, fracOverlap);
		std::cout << "[INFO]: Finished initializing MF Frequencies." << std::endl;
	}
	
	if (!mfs)
	{
		std::cout << "[INFO]: Initializing Poisson MF Population..." << std::endl;
		mfs = new PoissonRegenCells(NUM_MF, mfRandSeed, threshDecayTau, ap->msPerTimeStep,
			  	numMZones, NUM_NC);
		std::cout << "[INFO]: Finished initializing Poisson MF Population." << std::endl;
	}

	if (!output_arrays_initialized)
	{
		std::cout << "[INFO]: Initializing output arrays..." << std::endl;
		initializeOutputArrays();
		std::cout << "[INFO]: Finished initializing output arrays." << std::endl;
		output_arrays_initialized = true;
	}
}

Control::~Control()
{
	// delete all dynamic objects
	if (ap) delete ap;
	if (simState) delete simState;
	if (simCore) delete simCore;
	if (mfFreq) delete mfFreq;
	if (mfs) delete mfs;

	// deallocate output arrays
	if (output_arrays_initialized) deleteOutputArrays();
}

void Control::init_activity_params(std::string actParamFile)
{
	if (!ap)
	{
		ap = new ActivityParams(actParamFile);
	}
}

void Control::init_sim_state(std::string stateFile)
{
	if (!ap)
	{
		fprintf(stderr, "[ERROR]: Trying to initialize state without first initializing activity params.\n");
		fprintf(stderr, "[ERROR]: (Hint: Load an activity parameter file first then load the state.\n");
		return;
	}
	if (!simState)
	{
		simState = new CBMState(ap, numMZones, stateFile);
	}
	else
	{
		fprintf(stderr, "[ERROR]: State already initialized.\n");
	}
}

void Control::save_sim_state(std::string stateFile)
{
	if (!(ap && simState && simCore))
	{
		fprintf(stderr, "[ERROR]: Trying to write an uninitialized state to file.\n");
		fprintf(stderr, "[ERROR]: (Hint: Try loading activity parameter file and initializing the state first.)\n");
		return;
	}
	std::fstream outStateFileBuffer(stateFile.c_str(), std::ios::out | std::ios::binary);
	// notice we are using simcore here: in order to be save the state we *should* have
	// initialized not only activity params and state but also the sim core.
	simCore->writeState(ap, outStateFileBuffer);
	outStateFileBuffer.close();
}

void Control::initializeOutputArrays()
{
	allGRPSTH = allocate2DArray<ct_uint8_t>(NUM_GR, PSTHColSize);
	memset(allGRPSTH[0], '\000', (unsigned long)NUM_GR * (unsigned long)PSTHColSize * sizeof(ct_uint8_t));

	//allPCRaster = allocate2DArray<ct_uint8_t>(NUM_PC, rasterColumnSize);
	//std::fill(allPCRaster[0], allPCRaster[0] +
	//		NUM_PC * rasterColumnSize, 0);
	//
	//allNCRaster = allocate2DArray<ct_uint8_t>(NUM_NC, rasterColumnSize);
	//std::fill(allNCRaster[0], allNCRaster[0] +
	//		NUM_NC * rasterColumnSize, 0);

	//allSCRaster = allocate2DArray<ct_uint8_t>(NUM_SC, rasterColumnSize);
	//std::fill(allSCRaster[0], allSCRaster[0] +
	//		NUM_SC * rasterColumnSize, 0);

	//allBCRaster = allocate2DArray<ct_uint8_t>(NUM_BC, rasterColumnSize);
	//std::fill(allBCRaster[0], allBCRaster[0] +
	//		NUM_BC * rasterColumnSize, 0);
	//
	//allGOPSTH = allocate2DArray<ct_uint8_t>(NUM_GO, PSTHColSize);
	//std::fill(allGOPSTH[0], allGOPSTH[0] + NUM_GO * PSTHColSize, 0);
}

void Control::runTrials(int simNum, float GOGR, float GRGO, float MFGO)
{
	int preTrialNumber   = homeoTuningTrials + granuleActDetectTrials;
	int numTotalTrials   = preTrialNumber + numTrainingTrials;  

	float medTrials;
	std::time_t curr_time = std::time(nullptr);
	std::tm *local_time = std::localtime(&curr_time);
	clock_t timer;
	int rasterCounter = 0;
	int goSpkCounter[NUM_GO] = {0};

	FILE *fp = NULL;
	if (sim_vis_mode == TUI)
	{
		init_tty(&fp);
	}

	for (int trial = 0; trial < numTotalTrials; trial++)
	{
		// TODO: ensure we get time value before we pause!
		timer = clock();
		
		// re-initialize spike counter vector
		std::fill(goSpkCounter, goSpkCounter + NUM_GO, 0);

		int PSTHCounter = 0;
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;

		trial <= homeoTuningTrials ?
			std::cout << "Pre-tuning trial number: " << trial + 1 << std::endl :
			std::cout << "Post-tuning trial number: " << trial + 1 << std::endl;
		
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
				simCore->updateTrueMFs(isTrueMF); /* two unnecessary fnctn calls: isTrueMF doesn't change its value! */
				simCore->updateMFInput(mfAP);
				simCore->calcActivity(goMin, simNum, GOGR, GRGO, MFGO, gogoW, spillFrac);
				
				if (tts >= csStart && tts < csStart + csLength)
				{
					// TODO: refactor so that we do not export vars, instead we calc sum inside inputNet
					// e.g. simCore->updategGRGOSum(gGRGOSum); <- call by reference
					// even better: simCore->updateGSum<Granule, Golgi>(gGRGOSum); <- granule and golgi are their own cell objs
					mfgoG  = simCore->getInputNet()->exportgSum_MFGO();
					grgoG  = simCore->getInputNet()->exportgSum_GRGO();
					goSpks = simCore->getInputNet()->exportAPGO();
				
					for (int i = 0; i < NUM_GO; i++)
					{
							goSpkCounter[i] += goSpks[i];
							gGRGO_sum       += grgoG[i];
							gMFGO_sum       += mfgoG[i];
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
					fillOutputArrays(simCore, trial, PSTHCounter, rasterCounter);

					PSTHCounter++;
					rasterCounter++;
				}
				if (sim_vis_mode == GUI)
				{
					if (gtk_events_pending()) gtk_main_iteration(); // place this here?
				}
				else if (sim_vis_mode == TUI) process_input(&fp, tts, trial + 1); /* process user input from kb */
			}
		}
		timer = clock() - timer;
		std::cout << "Trial time seconds: " << (float)timer / CLOCKS_PER_SEC << std::endl;

		// save our outputs arrays after each trial...
		saveOutputArraysToFile(0, trial, local_time, simNum);

		// check the event queue after every iteration
		if (sim_vis_mode == GUI)
		{
			if (sim_is_paused)
			{
				std::cout << "[INFO]: Simulation is paused at end of trial " << trial << "." << std::endl;
				while(true)
				{
					// Weird edge case not taken into account: if there are events pending after user hits continue...
					if (gtk_events_pending() || sim_is_paused) gtk_main_iteration();
					else
					{
						std::cout << "[INFO]: Continuing..." << std::endl;
						break;
					}
				}
			}
		}
	}
	if (sim_vis_mode == TUI) reset_tty(&fp); /* reset the tty for later use */
}


void Control::saveOutputArraysToFile(int goRecipParam, int trial, std::tm *local_time, int simNum)
{
	if (trial > 0 && trial % 3 == 0)
	{
		std::cout << "[INFO]: Saving GR PSTH to file..." << std::endl;
		std::ostringstream allGRPSTHFileBuf;
		allGRPSTHFileBuf << OUTPUT_DATA_PATH
						 << "allGRPSTH"
						 << "_"
						 << std::put_time(local_time, "%d-%m-%Y")
						 << "_"
						 << std::to_string(trial + 1)
						 << "."
						 << BIN_EXT;
		std::string allGRPSTHFileName = allGRPSTHFileBuf.str();
		write2DCharArray(allGRPSTHFileName, allGRPSTH, NUM_GR, PSTHColSize);

		// reset GRPSTH
		memset(allGRPSTH[0], '\000', (unsigned long)NUM_GR * (unsigned long)PSTHColSize * sizeof(ct_uint8_t));
		//for (int i = 0; i < NUM_GR; i++)
		//{
		//	for (int j = 0; j < PSTHColSize; j++)
		//	{
		//		allGRPSTH[i][j] = 0;
		//	}
		//}
	}

	//std::string allGOPSTHFileName = "allGOPSTH_noGOGO_grgoConv" + std::to_string(conv[goRecipParam]) +
	//	"_" + std::to_string(simNum) + ".bin";
	//write2DCharArray(allGOPSTHFileName, allGOPSTH, NUM_GO, PSTHColSize);

	//std::cout << "Filling BC files" << std::endl;
	//
	//std::string allBCRasterFileName = "allBCRaster_paramSet" + std::to_string(inputStrength) +
	//	"_" + std::to_string(simNum) + ".bin";
	//write2DCharArray(allBCRasterFileName, allBCRaster, NUM_BC,
	//		numTrainingTrials * PSTHColSize);
	//
	//std::cout << "Filling SC files" << std::endl;

	//std::string allSCRasterFileName = "allSCRaster_paramSet" + std::to_string(inputStrength) +
	//	"_" + std::to_string(simNum) + ".bin";
	//write2DCharArray(allSCRasterFileName, allSCRaster, NUM_SC,
	//		numTrainingTrials * PSTHColSize);
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

void Control::fillOutputArrays(CBMSimCore *simCore, int trial, int PSTHCounter, int rasterCounter)
{
	const ct_uint8_t* grSpks = simCore->getInputNet()->exportAPGR();
	//const ct_uint8_t* pcSpks = simCore->getMZoneList()[0]->exportAPPC();
	//const ct_uint8_t* ncSpks = simCore->getMZoneList()[0]->exportAPNC();
	//const ct_uint8_t* bcSpks = simCore->getMZoneList()[0]->exportAPBC();
	//const ct_uint8_t* scSpks = simCore->getInputNet()->exportAPSC();
	for (int i = 0; i < NUM_GR; i++)
	{
		allGRPSTH[i][PSTHCounter] += grSpks[i];
	}

	//for (int i = 0; i < NUM_PC; i++)
	//{
	//	allPCRaster[i][rasterCounter] = pcSpks[i];
	//}

	//for (int i = 0; i < NUM_NC; i++)
	//{
	//	allNCRaster[i][rasterCounter] = ncSpks[i];
	//}

	//for (int i = 0; i < NUM_BC; i++)
	//{
	//	allBCRaster[i][rasterCounter] = bcSpks[i];
	//}

	//for (int i = 0; i < NUM_SC; i++)
	//{
	//	allSCRaster[i][rasterCounter] = scSpks[i];
	//}
}

// TODO: 1) find better place to put this 2) generalize
void Control::write2DCharArray(std::string outFileName, ct_uint8_t **inArr,
	unsigned int numRow, unsigned int numCol)
{
	std::fstream outStream(outFileName.c_str(), std::ios::out | std::ios::binary);

	if (!outStream.is_open())
	{
		// NOTE: should throw an error, which would be caught in main
		std::cerr << "couldn't open '" << outFileName << "' for writing." << std::endl;
		return;
	}
	rawBytesRW((char *)inArr[0], numRow * numCol * sizeof(ct_uint8_t), false, outStream);
	//for (size_t i = 0; i < numRow; i++)
	//{
	//	for (size_t j = 0; j < numCol; j++)
	//	{
	//		outStream.write((char*) &inArr[i][j], sizeof(ct_uint8_t));
	//	}
	//}
	outStream.close();
}

void Control::deleteOutputArrays()
{
	delete2DArray<ct_uint8_t>(allGRPSTH);
	//delete2DArray<ct_uint8_t>(allPCRaster);
	//delete2DArray<ct_uint8_t>(allNCRaster);
	//delete2DArray<ct_uint8_t>(allSCRaster);
	//delete2DArray<ct_uint8_t>(allBCRaster);
	//delete2DArray<ct_uint8_t>(allGOPSTH);
}

// NOTE:assumes that we have initialized activity params
void Control::construct_control(enum vis_mode sim_vis_mode)
{
	if (this->sim_vis_mode == NO_VIS) this->sim_vis_mode = sim_vis_mode;
	if (!simState)
	{
		std::cout << "[INFO]: Initializing state..." << std::endl;
		simState = new CBMState(ap, numMZones);
		std::cout << "[INFO]: Finished initializing state..." << std::endl;
	}
	
	if (!simCore)
	{
		std::cout << "[INFO]: Initializing simulation core..." << std::endl;
		simCore = new CBMSimCore(ap, simState, gpuIndex, gpuP2);
		std::cout << "[INFO]: Finished initializing simulation core." << std::endl;
	}

	if (!mfFreq)
	{
		std::cout << "[INFO]: Initializing MF Frequencies..." << std::endl;
		mfFreq = new ECMFPopulation(NUM_MF, mfRandSeed, CSTonicMFFrac, CSPhasicMFFrac,
			  contextMFFrac, nucCollFrac, bgFreqMin, csbgFreqMin, contextFreqMin, 
			  tonicFreqMin, phasicFreqMin, bgFreqMax, csbgFreqMax, contextFreqMax, 
			  tonicFreqMax, phasicFreqMax, collaterals_off, fracImport, secondCS, fracOverlap);
		std::cout << "[INFO]: Finished initializing MF Frequencies." << std::endl;
	}
	
	if (!mfs)
	{
		std::cout << "[INFO]: Initializing Poisson MF Population..." << std::endl;
		mfs = new PoissonRegenCells(NUM_MF, mfRandSeed, threshDecayTau, ap->msPerTimeStep,
			  	numMZones, NUM_NC);
		std::cout << "[INFO]: Finished initializing Poisson MF Population." << std::endl;
	}

	if (!output_arrays_initialized)
	{
		// allocate and initialize output arrays
		std::cout << "[INFO]: Initializing output arrays..." << std::endl;
		initializeOutputArrays();
		std::cout << "[INFO]: Finished initializing output arrays." << std::endl;
		output_arrays_initialized = true;
	}
}

