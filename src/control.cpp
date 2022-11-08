#include <iomanip>
#include <gtk/gtk.h>

#include "control.h"
#include "file_parse.h"
#include "tty.h"
#include "array_util.h"
#include "gui.h" /* tenuous inclide at best :pogO: */

const std::string BIN_EXT = "bin";
const std::string CELL_IDS[NUM_CELL_TYPES] = {"MF", "GR", "GO", "BC", "SC", "PC", "IO", "NC"}; 
 
Control::Control(parsed_commandline &p_cl)
{
	tokenized_file t_file;
	lexed_file l_file;
	if (!p_cl.build_file.empty())
	{
		visual_mode = "TUI";
		run_mode = "build";
		curr_build_file_name = p_cl.build_file;
		out_sim_file_name = p_cl.output_sim_file;
		parsed_build_file pb_file;
		tokenize_file(p_cl.build_file, t_file);
		lex_tokenized_file(t_file, l_file);
		parse_lexed_build_file(l_file, pb_file);
		if (!con_params_populated) populate_con_params(pb_file);
	}
	else if (!p_cl.session_file.empty())
	{
		visual_mode = p_cl.vis_mode;
		run_mode = "run";
		curr_sess_file_name = p_cl.session_file;
		curr_sim_file_name  = p_cl.input_sim_file;
		out_sim_file_name   = p_cl.output_sim_file;
		parsed_sess_file s_file;
		tokenize_file(curr_sess_file_name, t_file);
		lex_tokenized_file(t_file, l_file);
		parse_lexed_sess_file(l_file, s_file);
		translate_parsed_trials(s_file, td);
		trials_data_initialized = true;

		// TODO: move this somewhere else yike
		trialTime   = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["trialTime"].value);
		msPreCS     = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["msPreCS"].value);
		msPostCS    = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["msPostCS"].value);
		PSTHColSize = msPreCS + td.cs_lens[0] + msPostCS;

		set_plasticity_modes(p_cl);
		get_raster_filenames(p_cl.raster_files);
		get_weights_filenames(p_cl.weights_files);
		init_sim(s_file, curr_sim_file_name);
	}
}

Control::~Control()
{
	// delete allocated trials_data memory
	if (trials_data_initialized) delete_trials_data(td);

	// delete all dynamic objects
	if (simState) delete simState;
	if (simCore)  delete simCore;
	if (mfFreq)   delete mfFreq;
	if (mfs)      delete mfs;

	// deallocate output arrays
	if (internal_arrays_initialized) delete_rast_internal();
	if (output_arrays_initialized) deleteOutputArrays();
	if (spike_sums_initialized) delete_spike_sums();
}

void Control::build_sim()
{
	if (!simState) simState = new CBMState(numMZones);
}

void Control::set_plasticity_modes(parsed_commandline &p_cl)
{
	if (p_cl.pfpc_plasticity == "off") pf_pc_plast = OFF;
	else if (p_cl.pfpc_plasticity == "graded") pf_pc_plast = GRADED;
	else if (p_cl.pfpc_plasticity == "dual") pf_pc_plast = DUAL;
	else if (p_cl.pfpc_plasticity == "cascade") pf_pc_plast = CASCADE;

	if (p_cl.mfnc_plasticity == "off") mf_nc_plast = OFF;
	else if (p_cl.mfnc_plasticity == "graded") mf_nc_plast = GRADED;
	/* TODO: implement cmdline functionality to enable these */
	else if (p_cl.mfnc_plasticity == "dual") mf_nc_plast = DUAL;
	else if (p_cl.mfnc_plasticity == "cascade") mf_nc_plast = CASCADE;
}


void Control::init_sim(parsed_sess_file &s_file, std::string in_sim_filename)
{
	//if (sim_initialized) 
	//{
	//	if (curr_sim_file_name != in_sim_filename)
	//	{
	//		std::cout << "[INFO]: Deallocating previous simulation...\n";
	//		this->~Control();
	//		std::cout << "[INFO]: Finished deallocating previous simulation.\n";
	//	}
	//	else
	//	{
	//		std::cout << "[ERROR]: Trying to load in an already initialized file.\n";
	//		return;
	//	}
	//}
	std::cout << "[INFO]: Initializing simulation...\n";
	std::fstream sim_file_buf(in_sim_filename.c_str(), std::ios::in | std::ios::binary);
	read_con_params(sim_file_buf);
	populate_act_params(s_file);
	simState = new CBMState(numMZones, sim_file_buf);
	simCore  = new CBMSimCore(simState, gpuIndex, gpuP2);
	mfFreq   = new ECMFPopulation(num_mf, mfRandSeed, CSTonicMFFrac, CSPhasicMFFrac,
								  contextMFFrac, nucCollFrac, bgFreqMin, csbgFreqMin,
								  contextFreqMin, tonicFreqMin, phasicFreqMin, bgFreqMax,
								  csbgFreqMax, contextFreqMax, tonicFreqMax, phasicFreqMax,
								  collaterals_off, fracImport, secondCS, fracOverlap);
	mfs = new PoissonRegenCells(mfRandSeed, threshDecayTau, numMZones);
	initialize_rast_cell_nums();
	initialize_cell_spikes();
	initialize_rast_internal();
	initializeOutputArrays();
	initialize_spike_sums();
	sim_file_buf.close();
	sim_initialized = true;
	std::cout << "[INFO]: Simulation initialized.\n";
}

void Control::reset_sim(std::string in_sim_filename)
{
	std::fstream sim_file_buf(in_sim_filename.c_str(), std::ios::in | std::ios::binary);
	read_con_params(sim_file_buf);
	//read_act_params(sim_file_buf);
	simState->readState(sim_file_buf);
	// TODO: simCore, mfFreq, mfs
	
	reset_rast_internal();
	resetOutputArrays();
	reset_spike_sums();
	sim_file_buf.close();
	curr_sim_file_name = in_sim_filename;
}

void Control::save_sim_to_file(std::string outSimFile)
{
	std::fstream outSimFileBuffer(outSimFile.c_str(), std::ios::out | std::ios::binary);
	write_con_params(outSimFileBuffer);
	if (!simCore) simState->writeState(outSimFileBuffer);
	else simCore->writeState(outSimFileBuffer);
	outSimFileBuffer.close();
}

void Control::save_pfpc_weights_to_file(std::string out_pfpc_file)
{
	// TODO: make a boolean on weights loaded
	if (!simCore)
	{
		fprintf(stderr, "[ERROR]: Trying to write uninitialized weights to file.\n");
		fprintf(stderr, "[ERROR]: (Hint: Try initializing a sim or loading the weights first.)\n");
		return;
	}
	const float *pfpc_weights = simCore->getMZoneList()[0]->exportPFPCWeights();
	std::fstream outPFPCFileBuffer(out_pfpc_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)pfpc_weights, num_gr * sizeof(float), false, outPFPCFileBuffer);
	outPFPCFileBuffer.close();
}

void Control::load_pfpc_weights_from_file(std::string in_pfpc_file)
{
	if (!simCore)
	{
		fprintf(stderr, "[ERROR]: Trying to read weights to uninitialized simulation.\n");
		fprintf(stderr, "[ERROR]: (Hint: Try initializing a sim first.)\n");
		return;
	}
	std::fstream inPFPCFileBuffer(in_pfpc_file.c_str(), std::ios::in | std::ios::binary);
	simCore->getMZoneList()[0]->load_pfpc_weights_from_file(inPFPCFileBuffer);
	inPFPCFileBuffer.close();
} 

void Control::save_mfdcn_weights_to_file(std::string out_mfdcn_file)
{
	if (!simCore)
	{
		fprintf(stderr, "[ERROR]: Trying to write uninitialized weights to file.\n");
		fprintf(stderr, "[ERROR]: (Hint: Try initializing a sim or loading the weights first.)\n");
		return;
	}
	// TODO: make a export function for mfdcn weights
	const float *mfdcn_weights = simCore->getMZoneList()[0]->exportMFDCNWeights();
	std::fstream outMFDCNFileBuffer(out_mfdcn_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)mfdcn_weights, num_nc * num_p_nc_from_mf_to_nc * sizeof(const float), false, outMFDCNFileBuffer);
	outMFDCNFileBuffer.close();
}

void Control::load_mfdcn_weights_from_file(std::string in_mfdcn_file)
{
	if (!simCore)
	{
		fprintf(stderr, "[ERROR]: Trying to read weights to uninitialized simulation.\n");
		fprintf(stderr, "[ERROR]: (Hint: Try initializing a sim first.)\n");
		return;
	}
	std::fstream inMFDCNFileBuffer(in_mfdcn_file.c_str(), std::ios::in | std::ios::binary);
	simCore->getMZoneList()[0]->load_mfdcn_weights_from_file(inMFDCNFileBuffer);
	inMFDCNFileBuffer.close();
}

void Control::save_raster_to_file(std::string raster_file_name, enum cell_id id)
{
	std::fstream out_rast_file_buf(raster_file_name.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)rast_output[id], rast_sizes[id], false, out_rast_file_buf);
	out_rast_file_buf.close();
}

void Control::get_raster_filenames(std::map<std::string, std::string> &raster_files)
{
	if (!raster_files.empty())
	{
		for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
		{
			if (raster_files.find(CELL_IDS[i]) != raster_files.end())
			{
				rf_names[i] = raster_files[CELL_IDS[i]];
			}
		}
	}
}

void Control::get_weights_filenames(std::map<std::string, std::string> &weights_files)
{
	if (!weights_files.empty())
	{
		if (weights_files.find("PFPC") != weights_files.end())
		{
			pf_pc_weights_file = weights_files["PFPC"];
		}
		if (weights_files.find("MFNC") != weights_files.end())
		{
			mf_nc_weights_file = weights_files["MFNC"];
		}
	}
}

void Control::initialize_rast_cell_nums()
{
	rast_cell_nums[MF] = num_mf;
	rast_cell_nums[GR] = 4096;
	rast_cell_nums[GO] = num_go;
	rast_cell_nums[BC] = num_bc;
	rast_cell_nums[SC] = num_sc;
	rast_cell_nums[PC] = num_pc;
	rast_cell_nums[IO] = num_io;
	rast_cell_nums[NC] = num_nc;
}

void Control::initialize_cell_spikes()
{
	cell_spks[MF] = mfs->getAPs();
	/* NOTE: incurs a call to cudaMemcpy from device to host, but initializing so is not repeatedly called */
	cell_spks[GR] = simCore->getInputNet()->exportAPGR(); 
	cell_spks[GO] = simCore->getInputNet()->exportAPGO(); 
	cell_spks[BC] = simCore->getMZoneList()[0]->exportAPBC(); 
	cell_spks[SC] = simCore->getMZoneList()[0]->exportAPSC();
	cell_spks[PC] = simCore->getMZoneList()[0]->exportAPPC();
	cell_spks[IO] = simCore->getMZoneList()[0]->exportAPIO();
	cell_spks[NC] = simCore->getMZoneList()[0]->exportAPNC();
}

void Control::initialize_spike_sums()
{
	spike_sums[MF].num_cells  = num_mf;
	spike_sums[GR].num_cells  = num_gr;
	spike_sums[GO].num_cells  = num_go;
	spike_sums[BC].num_cells  = num_bc;
	spike_sums[SC].num_cells  = num_sc;
	spike_sums[PC].num_cells  = num_pc;
	spike_sums[IO].num_cells  = num_io;
	spike_sums[NC].num_cells = num_nc;

	FOREACH(spike_sums, ssp)
	{
		ssp->non_cs_spike_sum = 0;
		ssp->cs_spike_sum     = 0;
		ssp->non_cs_spike_counter = new uint32_t[ssp->num_cells];
		ssp->cs_spike_counter = new uint32_t[ssp->num_cells];
		memset((void *)ssp->non_cs_spike_counter, 0, ssp->num_cells * sizeof(uint32_t));
		memset((void *)ssp->cs_spike_counter, 0, ssp->num_cells * sizeof(uint32_t));
	}
	spike_sums_initialized = true;
}

void Control::initialize_rast_internal()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty())
			rast_internal[i] = allocate2DArray<uint8_t>(rast_cell_nums[i], PSTHColSize);
	}

	// TODO: find a way to initialize only within gui mode
	all_pc_vm_rast_internal = allocate2DArray<float>(num_pc, PSTHColSize);
	all_nc_vm_rast_internal = allocate2DArray<float>(num_nc, PSTHColSize);
	all_io_vm_rast_internal = allocate2DArray<float>(num_io, PSTHColSize);

	internal_arrays_initialized = true;
}

void Control::initializeOutputArrays()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty())
		{
			rast_sizes[i] = rast_cell_nums[i] * PSTHColSize * td.num_trials / BITS_PER_BYTE;
			rast_output[i] = (uint8_t *)calloc(rast_sizes[i], sizeof(uint8_t));
		}
	}

	if (!pf_pc_weights_file.empty())
	{
	// TODO: find a way to initialize only within gui mode

		sample_pfpc_syn_weights = new float[4096];
	}
	output_arrays_initialized = true;
}

void save_arr_as_csv(float in_arr[], uint32_t arr_len, std::string file_name)
{
	std::fstream out_file_buf(file_name.c_str(), std::ios::out);
	for (uint32_t i = 0; i < arr_len; i++)
	{
		out_file_buf << in_arr[i];
		if (i == arr_len - 1) out_file_buf << "\n";
		else out_file_buf << ", ";
	}
	out_file_buf.close();
}

void Control::runSession(struct gui *gui)
{
	float medTrials;
	double start, end;
	int goSpkCounter[num_go] = {0};
	if (gui == NULL) run_state = IN_RUN_NO_PAUSE;
	trial = 0;
	while (trial < td.num_trials && run_state != NOT_IN_RUN)
	{
		std::string trialName = td.trial_names[trial];

		uint32_t useCS        = td.use_css[trial];
		uint32_t onsetCS      = td.cs_onsets[trial];
		uint32_t csLength     = td.cs_lens[trial];
		uint32_t percentCS    = td.cs_percents[trial];
		uint32_t useUS        = td.use_uss[trial];
		uint32_t onsetUS      = td.us_onsets[trial];
		
		int PSTHCounter = 0;
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;

		memset(goSpkCounter, 0, num_go * sizeof(int));

		std::cout << "[INFO]: Trial number: " << trial + 1 << "\n";
		start = omp_get_wtime();
		for (int ts = 0; ts < trialTime; ts++)
		{
			if (useUS == 1 && ts == onsetUS) /* deliver the US */
			{
				simCore->updateErrDrive(0, 0.3);
			}
			if (ts >= onsetCS && ts < onsetCS + csLength)
			{
				mfAP = (useCS == 1) ? mfs->calcPoissActivity(mfFreq->getMFInCSTonicA(), simCore->getMZoneList())
									: mfs->calcPoissActivity(mfFreq->getMFBG(), simCore->getMZoneList());
			}
			else
			{
				mfAP = mfs->calcPoissActivity(mfFreq->getMFBG(),
					  simCore->getMZoneList());
			}
			
			bool *isTrueMF = mfs->calcTrueMFs(mfFreq->getMFBG()); /* only used for mfdcn plasticity */
			simCore->updateTrueMFs(isTrueMF);
			simCore->updateMFInput(mfAP);
			simCore->calcActivity(spillFrac, pf_pc_plast, mf_nc_plast); 
			//update_spike_sums(ts, onsetCS, onsetCS + csLength);

			if (ts >= onsetCS && ts < onsetCS + csLength)
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
			if (ts == onsetCS + csLength)
			{
				countGOSpikes(goSpkCounter, medTrials);
				std::cout << "[INFO]: Mean gGRGO   = " << gGRGO_sum / (num_go * csLength) << "\n";
				std::cout << "[INFO]: Mean gMFGO   = " << gMFGO_sum / (num_go * csLength) << "\n";
				std::cout << "[INFO]: GR:MF ratio  = " << gGRGO_sum / gMFGO_sum << "\n";
			}
			
			/* data collection */
			if (ts >= onsetCS - msPreCS && ts < onsetCS + csLength + msPostCS)
			{
				fill_rast_internal(PSTHCounter);
				PSTHCounter++;
			}

			if (gui != NULL)
			{
				if (gtk_events_pending()) gtk_main_iteration();
			}
		}
		end = omp_get_wtime();
		std::cout << "[INFO]: '" << trialName << "' took " << (end - start) << "s.\n";
		
		if (gui != NULL)
		{
			// for now, compute the mean and median firing rates for all cells if win is visible
			// if (firing_rates_win_visible(gui))
			// {
			// 	calculate_firing_rates(onsetCS, onsetCS + csLength);
			// 	gdk_threads_add_idle((GSourceFunc)update_fr_labels, gui);
			// }
			if (run_state == IN_RUN_PAUSE)
			{
				std::cout << "[INFO]: Simulation is paused at end of trial " << trial+1 << ".\n";
				while(run_state == IN_RUN_PAUSE)
				{
					if (gtk_events_pending()) gtk_main_iteration();
				}
				std::cout << "[INFO]: Continuing...\n";
			}
			//reset_spike_sums();
		}
		fillOutputArrays();
		trial++;
	}
	if (run_state == NOT_IN_RUN) std::cout << "[INFO]: Simulation terminated.\n";
	else if (run_state == IN_RUN_NO_PAUSE) std::cout << "[INFO]: Simulation Completed.\n";
	
	if (gui == NULL) saveOutputArraysToFile();
	run_state = NOT_IN_RUN;
}

void Control::reset_spike_sums()
{
		for (int i = 0; i < NUM_CELL_TYPES; i++)
		{
			spike_sums[i].cs_spike_sum = 0;
			spike_sums[i].non_cs_spike_sum = 0;
			memset((void *)(spike_sums[i].non_cs_spike_counter), 0, spike_sums[i].num_cells * sizeof(uint32_t));
			memset((void *)(spike_sums[i].cs_spike_counter), 0, spike_sums[i].num_cells * sizeof(uint32_t));
		}
}

void Control::reset_rast_internal()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty()) 
			memset(rast_internal[i][0], '\000', rast_cell_nums[i] * PSTHColSize * sizeof(uint8_t));
	}
}

void Control::resetOutputArrays()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty()) 
			memset(rast_output[i], '\000', rast_sizes[i] * sizeof(uint8_t));
	}
	memset(sample_pfpc_syn_weights, 0.0, 4096 * sizeof(float));
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

void Control::saveOutputArraysToFile()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty())
		{
			std::cout << "[INFO]: Filling " << CELL_IDS[i] << " raster file...\n";
			save_raster_to_file(rf_names[i], (enum cell_id)i);
		}
	}
}

void Control::update_spike_sums(int tts, float onset_cs, float offset_cs)
{
	// update cs spikes
	if (tts >= onset_cs && tts < offset_cs)
	{
		for (int i = 0; i < NUM_CELL_TYPES; i++)
		{
			for (int j = 0; j < spike_sums[i].num_cells; j++)
			{
				spike_sums[i].cs_spike_sum += cell_spks[i][j];
				spike_sums[i].cs_spike_counter[j] += cell_spks[i][j];
			}
		}
	}
	// update non-cs spikes
	else if (tts < onset_cs)
	{
		for (int i = 0; i < NUM_CELL_TYPES; i++)
		{
			for (int j = 0; j < spike_sums[i].num_cells; j++)
			{
				spike_sums[i].non_cs_spike_sum += cell_spks[i][j];
				spike_sums[i].non_cs_spike_counter[j] += cell_spks[i][j];
			}
		}
	}
}


void Control::calculate_firing_rates(float onset_cs, float offset_cs)
{
	float non_cs_time_secs = (onset_cs - 1) / 1000.0; // why only pre-cs? (Ask Joe)
	float cs_time_secs = (offset_cs - onset_cs) / 1000.0;

	for (int i = 0; i < NUM_CELL_TYPES; i++)
	{
		// sort sums for medians 
		std::sort(spike_sums[i].non_cs_spike_counter,
			spike_sums[i].non_cs_spike_counter + spike_sums[i].num_cells);
		std::sort(spike_sums[i].cs_spike_counter,
			spike_sums[i].cs_spike_counter + spike_sums[i].num_cells);
		
		// calculate medians
		firing_rates[i].non_cs_median_fr =
			(spike_sums[i].non_cs_spike_counter[spike_sums[i].num_cells / 2 - 1]
		   + spike_sums[i].non_cs_spike_counter[spike_sums[i].num_cells / 2]) / (2.0 * non_cs_time_secs);
		firing_rates[i].cs_median_fr     =
			(spike_sums[i].cs_spike_counter[spike_sums[i].num_cells / 2 - 1]
		   + spike_sums[i].cs_spike_counter[spike_sums[i].num_cells / 2]) / (2.0 * cs_time_secs);
		
		// calculate means
		firing_rates[i].non_cs_mean_fr = spike_sums[i].non_cs_spike_sum / (non_cs_time_secs * spike_sums[i].num_cells);
		firing_rates[i].cs_mean_fr     = spike_sums[i].cs_spike_sum / (cs_time_secs * spike_sums[i].num_cells);
	}
}

void Control::countGOSpikes(int *goSpkCounter, float &medTrials)
{
	float isi = (td.us_onsets[0] - td.cs_onsets[0]) / 1000.0;
	std::sort(goSpkCounter, goSpkCounter + num_go);
	
	float m = (goSpkCounter[num_go / 2 - 1] + goSpkCounter[num_go / 2]) / 2.0;
	float goSpkSum = 0;

	for (int i = 0; i < num_go; i++) goSpkSum += goSpkCounter[i];

	// NOTE: 1.0s below should really be the isi
	std::cout << "[INFO]: Mean GO Rate: " << goSpkSum / ((float)num_go * isi) << std::endl;

	medTrials += m / isi;
	std::cout << "[INFO]: Median GO Rate: " << m / isi << std::endl;
}

void Control::fill_rast_internal(int PSTHCounter)
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty())
		{
			/* GR spikes are only spikes not saved on host every time step:
			 * InNet::exportAPGR makes cudaMemcpy call before returning pointer to mem address */
			if (CELL_IDS[i] == "GR") cell_spks[i] = simCore->getInputNet()->exportAPGR();
			for (uint32_t j = 0; j < rast_cell_nums[i]; j++)
			{
				rast_internal[i][j][PSTHCounter] = cell_spks[i][j];
			}
		}
	}

	// TODO: might want to make this an array
	const float* vm_pc = simCore->getMZoneList()[0]->exportVmPC();
	for (int i = 0; i < num_pc; i++)
	{
		all_pc_vm_rast_internal[i][PSTHCounter] = vm_pc[i];
	}
	const float* vm_io = simCore->getMZoneList()[0]->exportVmIO();
	for (int i = 0; i < num_io; i++)
	{
		all_io_vm_rast_internal[i][PSTHCounter] = vm_io[i];
	}
	const float* vm_nc = simCore->getMZoneList()[0]->exportVmNC();
	for (int i = 0; i < num_nc; i++)
	{
		all_nc_vm_rast_internal[i][PSTHCounter] = vm_nc[i];
	}
}

void Control::fillOutputArrays()
{
	uint32_t offset_common = trial * PSTHColSize / BITS_PER_BYTE;
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty())
			pack_2d_byte_array(rast_internal[i], rast_cell_nums[i], PSTHColSize, rast_output[i], offset_common * rast_cell_nums[i]);
	}
}

void Control::delete_spike_sums()
{
	FOREACH(spike_sums, ssp)
	{
		delete[] ssp->non_cs_spike_counter;
		delete[] ssp->cs_spike_counter;
	}
}

void Control::delete_rast_internal()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty()) delete2DArray<uint8_t>(rast_internal[i]);
	}
	delete2DArray<float>(all_pc_vm_rast_internal);
	delete2DArray<float>(all_nc_vm_rast_internal);
	delete2DArray<float>(all_io_vm_rast_internal);
}

void Control::deleteOutputArrays()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty()) free(rast_output[i]);
	}
	delete[] sample_pfpc_syn_weights;
}

