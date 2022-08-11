#include <iomanip>
#include <gtk/gtk.h>
#include "control.h"
#include "fileIO/build_file.h"
#include "ttyManip/tty.h"

const std::string BIN_EXT = "bin";

// private utility function. TODO: move to a better place
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

Control::Control(parsed_build_file &p_file)
{
	if (!con_params_populated) populate_con_params(p_file);
	if (!act_params_populated) populate_act_params(p_file);
	if (!simState) simState = new CBMState(numMZones);
	if (!simCore) simCore = new CBMSimCore(simState, gpuIndex, gpuP2);
	if (!mfFreq)
	{
		mfFreq = new ECMFPopulation(num_mf, mfRandSeed, CSTonicMFFrac, CSPhasicMFFrac,
			  contextMFFrac, nucCollFrac, bgFreqMin, csbgFreqMin, contextFreqMin, 
			  tonicFreqMin, phasicFreqMin, bgFreqMax, csbgFreqMax, contextFreqMax, 
			  tonicFreqMax, phasicFreqMax, collaterals_off, fracImport, secondCS, fracOverlap);
	}
	if (!mfs)
	{
		mfs = new PoissonRegenCells(num_mf, mfRandSeed, threshDecayTau, msPerTimeStep,
			  	numMZones, num_nc);
	}

	if (!output_arrays_initialized) initializeOutputArrays();
}

Control::Control(std::string sim_file_name)
{
	std::fstream sim_file_buf(sim_file_name.c_str(), std::ios::in | std::ios::binary);
	if (!con_params_populated) read_con_params(sim_file_buf);
	if (!act_params_populated) read_act_params(sim_file_buf);
	if (!simState) simState = new CBMState(numMZones, sim_file_buf);
	if (!simCore) simCore = new CBMSimCore(simState, gpuIndex, gpuP2);
	if (!mfFreq)
	{
		mfFreq = new ECMFPopulation(num_mf, mfRandSeed,
			  CSTonicMFFrac, CSPhasicMFFrac, contextMFFrac, nucCollFrac,
			  bgFreqMin, csbgFreqMin, contextFreqMin, tonicFreqMin, phasicFreqMin, bgFreqMax,
			  csbgFreqMax, contextFreqMax, tonicFreqMax, phasicFreqMax, collaterals_off,
			  fracImport, secondCS, fracOverlap);
	}
	if (!mfs)
	{
		mfs = new PoissonRegenCells(num_mf, mfRandSeed,
				threshDecayTau, msPerTimeStep, numMZones, num_nc);
	}
	if (!output_arrays_initialized) initializeOutputArrays();
	sim_file_buf.close();
}

Control::~Control()
{
	// delete all dynamic objects
	if (simState) delete simState;
	if (simCore)  delete simCore;
	if (mfFreq)   delete mfFreq;
	if (mfs)      delete mfs;

	// deallocate output arrays
	if (output_arrays_initialized) deleteOutputArrays();
}

void Control::build_sim(parsed_build_file &p_file)
{
	// not sure if we want to save mfFreq and mfs in the simulation file
	if (!(con_params_populated && act_params_populated && simState))
	{
		populate_con_params(p_file);
		populate_act_params(p_file);
		simState = new CBMState(numMZones);
	}
}

void Control::init_activity_params(std::string actParamFile)
{
	// do nothing for now so we can compile.
	// TODO: deprecate
	//if (!act_params_populated())
	//{
	//	populate_act_params
	//}
}

void Control::init_sim_state(std::string stateFile)
{
	if (!con_params_populated)
	{
		fprintf(stderr, "[ERROR]: Trying to initialize state without first connectivity params.\n");
		fprintf(stderr, "[ERROR]: (Hint: Load a connectivity parameter file first then load the state.\n");
		return;
	}
	if (!act_params_populated)
	{
		fprintf(stderr, "[ERROR]: Trying to initialize state without first initializing activity params.\n");
		fprintf(stderr, "[ERROR]: (Hint: Load an activity parameter file first then load the state.\n");
		return;
	}
	if (!simState) simState = new CBMState(numMZones, stateFile);
	else fprintf(stderr, "[ERROR]: State already initialized.\n");
}

void Control::save_sim_state_to_file(std::string outStateFile)
{
	if (!(con_params_populated && act_params_populated && simState))
	{
		fprintf(stderr, "[ERROR]: Trying to write an uninitialized state to file.\n");
		fprintf(stderr, "[ERROR]: (Hint: Try loading activity parameter file and initializing the state first.)\n");
		return;
	}
	std::fstream outStateFileBuffer(outStateFile.c_str(), std::ios::out | std::ios::binary);
	if (simCore)
	{
		// if we have simCore, save from simcore else save from state.
		// this covers both edge case scenarios of if the user wants to save to
		// file an initialized bunny or if the user is using the gui and forgot 
		// to save to file after initializing state, but wants to do so after 
		// initializing simcore.
		simCore->writeState(outStateFileBuffer);
	}
	else simState->writeState(outStateFileBuffer); 
	outStateFileBuffer.close();
}

void Control::save_sim_to_file(std::string outSimFile)
{
	if (!(con_params_populated && act_params_populated && simState))
	{
		fprintf(stderr, "[ERROR]: Trying to write an uninitialized simulation to file.\n");
		fprintf(stderr, "[ERROR]: (Hint: Try loading a build file first.)\n");
		return;
	}
	std::fstream outSimFileBuffer(outSimFile.c_str(), std::ios::out | std::ios::binary);
	write_con_params(outSimFileBuffer);
	write_act_params(outSimFileBuffer);
	if (simCore) simCore->writeState(outSimFileBuffer);
	else simState->writeState(outSimFileBuffer); 
	outSimFileBuffer.close();
}

void Control::initializeOutputArrays()
{
	//allGRPSTH = allocate2DArray<ct_uint8_t>(NUM_GR, PSTHColSize);
	//memset(allGRPSTH[0], '\000', (unsigned long)NUM_GR * (unsigned long)PSTHColSize * sizeof(ct_uint8_t));

	// DEBUG: looking at a sample of the granules of size 4096
	sampleGRRaster = allocate2DArray<ct_uint8_t>(4096, rasterColumnSize);
	std::fill(sampleGRRaster[0], sampleGRRaster[0] +
			4096 * rasterColumnSize, 0);

	allMFRaster = allocate2DArray<ct_uint8_t>(num_mf, rasterColumnSize);
	std::fill(allMFRaster[0], allMFRaster[0] +
			num_mf * rasterColumnSize, 0);

	allGORaster = allocate2DArray<ct_uint8_t>(num_go, rasterColumnSize);
	std::fill(allGORaster[0], allGORaster[0] +
			num_go * rasterColumnSize, 0);

	allPCRaster = allocate2DArray<ct_uint8_t>(num_pc, rasterColumnSize);
	std::fill(allPCRaster[0], allPCRaster[0] +
			num_pc * rasterColumnSize, 0);
	
	allNCRaster = allocate2DArray<ct_uint8_t>(num_nc, rasterColumnSize);
	std::fill(allNCRaster[0], allNCRaster[0] +
			num_nc * rasterColumnSize, 0);

	allSCRaster = allocate2DArray<ct_uint8_t>(num_sc, rasterColumnSize);
	std::fill(allSCRaster[0], allSCRaster[0] +
			num_sc * rasterColumnSize, 0);

	allBCRaster = allocate2DArray<ct_uint8_t>(num_bc, rasterColumnSize);
	std::fill(allBCRaster[0], allBCRaster[0] +
			num_bc * rasterColumnSize, 0);

	allIORaster = allocate2DArray<ct_uint8_t>(num_io, rasterColumnSize);
	std::fill(allIORaster[0], allIORaster[0] +
			num_io * rasterColumnSize, 0);
	
	output_arrays_initialized = true;
	//allGOPSTH = allocate2DArray<ct_uint8_t>(NUM_GO, PSTHColSize);
	//std::fill(allGOPSTH[0], allGOPSTH[0] + NUM_GO * PSTHColSize, 0);
}

void Control::runExperiment(experiment &experiment)
{
	float medTrials;
	std::time_t curr_time = std::time(nullptr);
	std::tm *local_time = std::localtime(&curr_time);
	clock_t timer;
	
	int rasterCounter = 0;
	int goSpkCounter[num_go] = {0};

	for (int trial = 0; trial < experiment.num_trials; trial++)
	{
		std::string trialName = experiment.trials[trial].TrialName;

		int useCS     = experiment.trials[trial].CSuse;
		int onsetCS   = experiment.trials[trial].CSonset;
		int offsetCS  = experiment.trials[trial].CSoffset;
		int percentCS = experiment.trials[trial].CSpercent;
		int useUS     = experiment.trials[trial].USuse;
		int onsetUS   = experiment.trials[trial].USonset;
		
		//std::cout << "'useCS': " << useCS << std::endl;
		//std::cout << "'onsetCS': " << onsetCS << std::endl;
		//std::cout << "'offsetCS': " << offsetCS << std::endl;
		//std::cout << "'useUS': " << useUS << std::endl;
		//std::cout << "'onsetUS': " << onsetUS << std::endl;

		timer = clock();
		int PSTHCounter = 0;
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;
		memset(goSpkCounter, 0, num_go * sizeof(int));

		for (int ts = 0; ts < trialTime; ts++)
		{
			if (useUS && ts == onsetUS) /* deliver the US */
			{
				simCore->updateErrDrive(0, 0.0);
			}
			if (ts < onsetCS || ts >= offsetCS)
			{
				mfAP = mfs->calcPoissActivity(mfFreq->getMFBG(),
					  simCore->getMZoneList());
			}
			if (ts >= onsetCS && ts < offsetCS)
			{
				if (useCS)
				{
					mfAP = mfs->calcPoissActivity(mfFreq->getMFInCSTonicA(),
						  simCore->getMZoneList());
				}
				else
				{
					mfAP = mfs->calcPoissActivity(mfFreq->getMFBG(),
						  simCore->getMZoneList());
				}
			}
			
			bool *isTrueMF = mfs->calcTrueMFs(mfFreq->getMFBG());
			simCore->updateTrueMFs(isTrueMF);
			simCore->updateMFInput(mfAP);
			simCore->calcActivity(mfgoW, gogrW, grgoW, gogoW, spillFrac); 

			if (ts >= onsetCS && ts < offsetCS)
			{
				mfgoG  = simCore->getInputNet()->exportgSum_MFGO();
				grgoG  = simCore->getInputNet()->exportgSum_GRGO();
				goSpks = simCore->getInputNet()->exportAPGO();
			
				for (int i = 0; i < num_go; i++)
				{
						goSpkCounter[i] += goSpks[i];
						gGRGO_sum       += grgoG[i];
						gMFGO_sum       += mfgoG[i];
				}
			}
			
			/* upon offset of CS, report what we got*/
			if (ts == offsetCS)
			{
				countGOSpikes(goSpkCounter, medTrials);
				std::cout << "mean gGRGO   = " << gGRGO_sum / (num_go * (offsetCS - onsetCS)) << std::endl;
				std::cout << "mean gMFGO   = " << gMFGO_sum / (num_go * (offsetCS - onsetCS)) << std::endl;
				std::cout << "GR:MF ratio  = " << gGRGO_sum / gMFGO_sum << std::endl;
			}

			if (sim_vis_mode == GUI)
			{
				if (gtk_events_pending()) gtk_main_iteration();
			}
		}
		
		timer = clock() - timer;
		std::cout << "[INFO]: " << trialName << " took " << (float)timer / CLOCKS_PER_SEC << "s."
				  << std::endl;
		
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
}
void gen_gr_sample(int gr_indices[], int sample_size, int data_size)
{
	CRandomSFMT0 randGen(0); // replace seed later
	bool chosen[data_size] = {false}; 
	int counter = 0;
	while (counter < sample_size)
	{
		int index = randGen.IRandom(0, data_size - 1);
		if (!chosen[index])
		{
			gr_indices[counter] = index;
			chosen[index] = true;
			counter++;
		} 
	}
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
	int goSpkCounter[num_go] = {0};
	int gr_indices[4096] = {0};
	gen_gr_sample(gr_indices, 4096, num_gr);

	GOGR = 0.017;
	GRGO = 0.0007 * 0.9;
	MFGO = 0.00350 * 0.9;

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
		std::fill(goSpkCounter, goSpkCounter + num_go, 0);

		int PSTHCounter = 0;
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;
		float gGOGR_sum = 0;
		float gMFGR_sum = 0;

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
				simCore->calcActivity(MFGO, GOGR, GRGO, gogoW, spillFrac);
				
				if (tts >= csStart && tts < csStart + csLength)
				{
					// TODO: refactor so that we do not export vars, instead we calc sum inside inputNet
					// e.g. simCore->updategGRGOSum(gGRGOSum); <- call by reference
					// even better: simCore->updateGSum<Granule, Golgi>(gGRGOSum); <- granule and golgi are their own cell objs
					mfgoG  = simCore->getInputNet()->exportgSum_MFGO();
					grgoG  = simCore->getInputNet()->exportgSum_GRGO();
					//mfgrG  = simCore->getInputNet()->exportGESumGR(); // excite
					//gogrG  = simCore->getInputNet()->exportGISumGR(); // inhibit
					goSpks = simCore->getInputNet()->exportAPGO();
				
					for (int i = 0; i < num_go; i++)
					{
							goSpkCounter[i] += goSpks[i];
							gGRGO_sum       += grgoG[i];
							gMFGO_sum       += mfgoG[i];
					}
					//for (int i = 0; i < 10000; i++)
					//{
					//		gGOGR_sum       += gogrG[i];
					//		gMFGR_sum       += mfgrG[i];
					//}
				}
				if (tts == csStart + csLength)
				{
					countGOSpikes(goSpkCounter, medTrials);
					std::cout << "mean gGRGO   = " << gGRGO_sum / (num_go * csLength) << std::endl;
					std::cout << "mean gMFGO   = " << gMFGO_sum / (num_go * csLength) << std::endl;
					//std::cout << "mean gGOGR   = " << gGOGR_sum / (10000 * csLength) << std::endl;
					//std::cout << "mean gMFGR   = " << gMFGR_sum / (10000 * csLength) << std::endl;
					std::cout << "GR:MF ratio  = " << gGRGO_sum / gMFGO_sum << std::endl;
				}

				if (trial >= preTrialNumber && tts >= csStart-msPreCS &&
						tts < csStart + csLength + msPostCS)
				{
					fillOutputArrays(gr_indices, mfAP, simCore, trial, PSTHCounter, rasterCounter);

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
	saveOutputArraysToFile(0, 0, local_time, 0);
}


void Control::saveOutputArraysToFile(int goRecipParam, int trial, std::tm *local_time, int simNum)
{
	std::cout << "Filling MF files..." << std::endl;
	std::string allMFRasterFileName = OUTPUT_DATA_PATH + "allMFRaster.bin";
	write2DCharArray(allMFRasterFileName, allMFRaster, num_mf, rasterColumnSize);

	std::cout << "Filling GO files..." << std::endl;
	std::string allGORasterFileName = OUTPUT_DATA_PATH + "allGORaster.bin";
	write2DCharArray(allGORasterFileName, allGORaster, num_go, rasterColumnSize);

	std::cout << "Filling GR files..." << std::endl;
	std::string sampleGRRasterFileName = OUTPUT_DATA_PATH + "sampleGRRaster.bin";
	write2DCharArray(sampleGRRasterFileName, sampleGRRaster, 4096, rasterColumnSize);

	std::cout << "Filling PC files..." << std::endl;
	std::string allPCRasterFileName = OUTPUT_DATA_PATH + "allPCRaster.bin";
	write2DCharArray(allPCRasterFileName, allPCRaster, num_pc, rasterColumnSize);

	std::cout << "Filling NC files..." << std::endl;
	std::string allNCRasterFileName = OUTPUT_DATA_PATH + "allNCRaster.bin";
	write2DCharArray(allNCRasterFileName, allNCRaster, num_nc, rasterColumnSize);

	std::cout << "Filling SC files..." << std::endl;
	std::string allSCRasterFileName = OUTPUT_DATA_PATH + "allSCRaster.bin";
	write2DCharArray(allSCRasterFileName, allSCRaster, num_sc, rasterColumnSize);

	std::cout << "Filling BC files..." << std::endl;
	std::string allBCRasterFileName = OUTPUT_DATA_PATH + "allBCRaster.bin";
	write2DCharArray(allBCRasterFileName, allBCRaster, num_bc, rasterColumnSize);

	std::cout << "Filling IO files..." << std::endl;
	std::string allIORasterFileName = OUTPUT_DATA_PATH + "allIORaster.bin";
	write2DCharArray(allIORasterFileName, allIORaster, num_io, rasterColumnSize);

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

	for (int i = 0; i < num_go; i++) goSpkSum += goSpkCounter[i];
	
	std::cout << "Mean GO Rate: " << goSpkSum / (float)num_go << std::endl;

	medTrials += m / 2.0;
	std::cout << "Median GO Rate: " << m / 2.0 << std::endl;
}
void Control::fillOutputArrays(int *gr_indices, const ct_uint8_t *mfAP, CBMSimCore *simCore, int trial, int PSTHCounter, int rasterCounter)
{
	const ct_uint8_t* goSpks = simCore->getInputNet()->exportAPGO();
	const ct_uint8_t* grSpks = simCore->getInputNet()->exportAPGR();
	const ct_uint8_t* pcSpks = simCore->getMZoneList()[0]->exportAPPC();
	const ct_uint8_t* ncSpks = simCore->getMZoneList()[0]->exportAPNC();
	const ct_uint8_t* scSpks = simCore->getInputNet()->exportAPSC();
	const ct_uint8_t* bcSpks = simCore->getMZoneList()[0]->exportAPBC();
	const ct_uint8_t* ioSpks = simCore->getMZoneList()[0]->exportAPIO();

	for (int i = 0; i < num_mf; i++)
	{
		allMFRaster[i][rasterCounter] = mfAP[i];
	}

	for (int i = 0; i < num_go; i++)
	{
		allGORaster[i][rasterCounter] = goSpks[i];
	}

	// 4096 sample size
	for (int i = 0; i < 4096; i++)
	{
		sampleGRRaster[i][rasterCounter] = grSpks[gr_indices[i]];
	}

	for (int i = 0; i < num_pc; i++)
	{
		allPCRaster[i][rasterCounter] = pcSpks[i];
	}

	for (int i = 0; i < num_nc; i++)
	{
		allNCRaster[i][rasterCounter] = ncSpks[i];
	}

	for (int i = 0; i < num_sc; i++)
	{
		allSCRaster[i][rasterCounter] = scSpks[i];
	}

	for (int i = 0; i < num_bc; i++)
	{
		allBCRaster[i][rasterCounter] = bcSpks[i];
	}

	for (int i = 0; i < num_io; i++)
	{
		allIORaster[i][rasterCounter] = ioSpks[i];
	}

}

// TODO: 1) find better place to put this 2) generalize
void Control::write2DCharArray(std::string outFileName, ct_uint8_t **inArr,
	unsigned int numRow, unsigned int numCol)
{
	std::fstream outStream(outFileName.c_str(), std::ios::out | std::ios::binary);

	if (!outStream.is_open())
	{
		std::cerr << "couldn't open '" << outFileName << "' for writing." << std::endl;
		exit(-1);
	}
	rawBytesRW((char *)inArr[0], numRow * numCol * sizeof(ct_uint8_t), false, outStream);
	outStream.close();
}

void Control::deleteOutputArrays()
{
	delete2DArray<ct_uint8_t>(allMFRaster);
	delete2DArray<ct_uint8_t>(allGORaster);
	delete2DArray<ct_uint8_t>(sampleGRRaster);
	delete2DArray<ct_uint8_t>(allPCRaster);
	delete2DArray<ct_uint8_t>(allNCRaster);
	delete2DArray<ct_uint8_t>(allSCRaster);
	delete2DArray<ct_uint8_t>(allBCRaster);
	delete2DArray<ct_uint8_t>(allIORaster);
}

// NOTE: assumes that we have initialized activity params
// TODO: find a better design than this: why else would we have a constructor???
void Control::construct_control(enum vis_mode sim_vis_mode)
{
	if (this->sim_vis_mode == NO_VIS) this->sim_vis_mode = sim_vis_mode;
	if (!simState) simState = new CBMState(numMZones);
	
	if (!simCore) simCore = new CBMSimCore(simState, gpuIndex, gpuP2);

	if (!mfFreq)
	{
		mfFreq = new ECMFPopulation(NUM_MF, mfRandSeed, CSTonicMFFrac, CSPhasicMFFrac,
			  contextMFFrac, nucCollFrac, bgFreqMin, csbgFreqMin, contextFreqMin, 
			  tonicFreqMin, phasicFreqMin, bgFreqMax, csbgFreqMax, contextFreqMax, 
			  tonicFreqMax, phasicFreqMax, collaterals_off, fracImport, secondCS, fracOverlap);
	}
	
	if (!mfs)
	{
		mfs = new PoissonRegenCells(NUM_MF, mfRandSeed, threshDecayTau, msPerTimeStep,
			  	numMZones, NUM_NC);
	}

	if (!output_arrays_initialized)
	{
		// allocate and initialize output arrays
		initializeOutputArrays();
	}
}

