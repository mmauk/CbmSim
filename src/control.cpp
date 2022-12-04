#include <sys/stat.h> // mkdir (POSIX ONLY)
#include <iomanip>
#include <gtk/gtk.h>

#include "control.h"
#include "file_parse.h"
#include "tty.h"
#include "array_util.h"
#include "gui.h" /* tenuous inclide at best :pogO: */

const std::string BIN_EXT = ".bin";
const std::string SIM_EXT = ".sim";
const std::string CELL_IDS[NUM_CELL_TYPES] = {"MF", "GR", "GO", "BC", "SC", "PC", "IO", "NC"}; 

Control::Control(parsed_commandline &p_cl)
{
	use_gui = (p_cl.vis_mode == "GUI") ? true : false;
	if (!p_cl.build_file.empty())
	{
		tokenized_file t_file;
		lexed_file l_file;
		parsed_build_file pb_file;
		tokenize_file(p_cl.build_file, t_file);
		lex_tokenized_file(t_file, l_file);
		parse_lexed_build_file(l_file, pb_file);
		if (!con_params_populated) populate_con_params(pb_file);
	}
	else if (!p_cl.session_file.empty())
	{
		initialize_session(p_cl.session_file);
		set_plasticity_modes(p_cl.pfpc_plasticity, p_cl.mfnc_plasticity);
		data_out_path = OUTPUT_DATA_PATH + p_cl.output_basename;
		data_out_base_name = p_cl.output_basename;
		// NOTE: make the output directory here, so in case of error, user not
		// run an entire simulation just to not have files save
		int status = mkdir(data_out_path.c_str(), 0775);
		if (status == -1)
		{
			std::cerr << "[ERROR]: Could not create directory '" << data_out_path << "'. Maybe it already exists\n"
					  << "[ERROR]: Exiting...\n";
			exit(10);
		}
		create_out_sim_filename(p_cl);
		create_raster_filenames(p_cl);
		create_psth_filenames(p_cl);
		create_weights_filenames(p_cl);
		init_sim(p_cl.input_sim_file);
	}
	else // user ran executable with no args FIXME: find out how to initialize with gui, couple similar parts of code
	{
		set_plasticity_modes("graded", "graded"); 
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
	if (raster_arrays_initialized) delete_rasters();
	if (psth_arrays_initialized)   delete_psths();
	if (spike_sums_initialized)    delete_spike_sums();
}

void Control::build_sim()
{
	// TODO: create a separate function to create the state,
	// have the constructor allocate memory and initialize values
	if (!simState) simState = new CBMState(numMZones);
}

void Control::set_plasticity_modes(std::string pfpc_plasticity, std::string mfnc_plasticity)
{
	if (pfpc_plasticity == "off") pf_pc_plast = OFF;
	else if (pfpc_plasticity == "graded") pf_pc_plast = GRADED;
	else if (pfpc_plasticity == "binary") pf_pc_plast = BINARY;
	else if (pfpc_plasticity == "cascade") pf_pc_plast = CASCADE;


	if (mfnc_plasticity == "off") mf_nc_plast = OFF;
	else if (mfnc_plasticity == "graded") mf_nc_plast = GRADED;
	/* TODO: implement cmdline functionality to enable these */
	else if (mfnc_plasticity == "binary") mf_nc_plast = BINARY;
	else if (mfnc_plasticity == "cascade") mf_nc_plast = CASCADE;
}

void Control::initialize_session(std::string sess_file)
{
	std::cout << "[INFO]: Initializing session...\n";
	tokenized_file t_file;
	lexed_file l_file;
	tokenize_file(sess_file, t_file);
	lex_tokenized_file(t_file, l_file);
	parse_lexed_sess_file(l_file, s_file);
	translate_parsed_trials(s_file, td);

	trialTime   = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["trialTime"].value);
	msPreCS     = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["msPreCS"].value);
	msPostCS    = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["msPostCS"].value);
	PSTHColSize = msPreCS + td.cs_lens[0] + msPostCS;

	trials_data_initialized = true;
	std::cout << "[INFO]: Session initialized.\n";
}

void Control::init_sim(std::string in_sim_filename)
{
	std::cout << "[INFO]: Initializing simulation...\n";
	std::fstream sim_file_buf(in_sim_filename.c_str(), std::ios::in | std::ios::binary);
	read_con_params(sim_file_buf);
	populate_act_params(s_file); // FIXME: place act params in separate file so we don't have to make s_file class attrib
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
	initialize_rasters();
	initialize_psth_save_funcs();
	initialize_raster_save_funcs();
	initialize_psths();
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
	
	reset_rasters();
	reset_psths();
	reset_spike_sums();
	sim_file_buf.close();
	// TODO: more things to reset?
}

void Control::save_sim_to_file(std::string outSimFile)
{
	if (!outSimFile.empty())
	{
		std::cout << "[INFO]: Saving simulation to '" << outSimFile << "'\n";
		std::fstream outSimFileBuffer(outSimFile.c_str(), std::ios::out | std::ios::binary);
		write_con_params(outSimFileBuffer);
		if (!simCore) simState->writeState(outSimFileBuffer);
		else simCore->writeState(outSimFileBuffer);
		outSimFileBuffer.close();
	}
}

void Control::save_pfpc_weights_to_file(std::string out_pfpc_file)
{
	if (!out_pfpc_file.empty())
	{
		std::cout << "[INFO]: Saving PFPC weights to '" <<  out_pfpc_file << "'\n";
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
	if (!out_mfdcn_file.empty())
	{
		std::cout << "[INFO]: Saving MFNC weights to '" <<  out_mfdcn_file << "'\n";
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

void Control::create_out_sim_filename(parsed_commandline &p_cl)
{
	if (!p_cl.output_sim_file.empty())
	{
		out_sim_name = data_out_path + "/" + data_out_base_name + SIM_EXT;
	}
}

// TODO: combine two below funcs into one for generality
void Control::create_raster_filenames(parsed_commandline &p_cl)
{
	if (!p_cl.raster_files.empty())
	{
		for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
		{
			std::string cell_id = CELL_IDS[i];
			if (p_cl.raster_files[cell_id] && !data_out_path.empty())
			{
				rf_names[i] = data_out_path + "/" + data_out_base_name
											+ "_" + cell_id + "_RASTER_"
											+ get_current_time_as_string()
											+ BIN_EXT;
			}
		}
	}
}

void Control::create_psth_filenames(parsed_commandline &p_cl)
{
	if (!p_cl.psth_files.empty())
	{
		for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
		{
			std::string cell_id = CELL_IDS[i];
			if (p_cl.psth_files[cell_id] && !data_out_path.empty())
			{
				pf_names[i] = data_out_path + "/" + data_out_base_name 
							+ "_" + cell_id + "_PSTH_"
							+ get_current_time_as_string()
							+ BIN_EXT;
			}
		}
	}
}

void Control::create_weights_filenames(parsed_commandline &p_cl)
{
	if (!p_cl.weights_files.empty())
	{
		if (p_cl.weights_files["PFPC"] && !data_out_path.empty())
		{
			pf_pc_weights_file = data_out_path + "/" + data_out_base_name
							   + "_PFPC_WEIGHTS_" + get_current_time_as_string()
							   + BIN_EXT;
		}
		if (p_cl.weights_files["MFNC"] && !data_out_path.empty())
		{
			mf_nc_weights_file = data_out_path + "/" + data_out_base_name
							   + "_MFNC_WEIGHTS_" + get_current_time_as_string()
							   + BIN_EXT;
		}
	}
}

void Control::initialize_rast_cell_nums()
{
	rast_cell_nums[MF] = num_mf;
	rast_cell_nums[GR] = num_gr; 
	rast_cell_nums[GO] = num_go;
	rast_cell_nums[BC] = num_bc;
	rast_cell_nums[SC] = num_sc;
	rast_cell_nums[PC] = num_pc;
	rast_cell_nums[IO] = num_io;
	rast_cell_nums[NC] = num_nc;
}

void Control::initialize_cell_spikes()
{
	cell_spikes[MF] = mfs->getAPs();
	/* NOTE: incurs a call to cudaMemcpy from device to host, but initializing so is not repeatedly called */
	cell_spikes[GR] = simCore->getInputNet()->exportAPGR(); 
	cell_spikes[GO] = simCore->getInputNet()->exportAPGO(); 
	cell_spikes[BC] = simCore->getMZoneList()[0]->exportAPBC(); 
	cell_spikes[SC] = simCore->getMZoneList()[0]->exportAPSC();
	cell_spikes[PC] = simCore->getMZoneList()[0]->exportAPPC();
	cell_spikes[IO] = simCore->getMZoneList()[0]->exportAPIO();
	cell_spikes[NC] = simCore->getMZoneList()[0]->exportAPNC();
}

void Control::initialize_spike_sums()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		spike_sums[i].non_cs_spike_sum = 0;
		spike_sums[i].cs_spike_sum = 0;
		spike_sums[i].non_cs_spike_counter = (uint32_t *)calloc(rast_cell_nums[i], sizeof(uint32_t));
		spike_sums[i].cs_spike_counter = (uint32_t *)calloc(rast_cell_nums[i], sizeof(uint32_t));
	}
	spike_sums_initialized = true;
}

void Control::initialize_rasters()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty() || use_gui)
		{
			/* granules are saved every trial, so their raster size is PSTHColSize  x num_gr */
			uint32_t row_size = (CELL_IDS[i] == "GR") ? PSTHColSize : PSTHColSize * td.num_trials;
			rasters[i] = allocate2DArray<uint8_t>(row_size, rast_cell_nums[i]);
		}
	}

	if (use_gui)
	{
		// TODO: find a way to initialize only within gui mode
		pc_vm_raster = allocate2DArray<float>(PSTHColSize, num_pc);
		nc_vm_raster = allocate2DArray<float>(PSTHColSize, num_nc);
		io_vm_raster = allocate2DArray<float>(PSTHColSize, num_io);
	}

	raster_arrays_initialized = true;
}

void Control::initialize_psth_save_funcs()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		psth_save_funcs[i] = [this, i](std::string filename)
		{
			if (!filename.empty())
			{
				std::cout << "[INFO]: Saving " << CELL_IDS[i] << " psth to '" <<  filename << "'\n";
				write2DArray<uint8_t>(filename, this->psths[i], this->PSTHColSize, this->rast_cell_nums[i]);
			}
		};
	}
}

void Control::initialize_raster_save_funcs()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		rast_save_funcs[i] = [this, i](std::string filename)
		{
			if (!filename.empty() && CELL_IDS[i] != "GR")
			{
				uint32_t row_size = (CELL_IDS[i] == "GR") ? this->PSTHColSize : this->PSTHColSize * this->td.num_trials;
				std::cout << "[INFO]: Saving " << CELL_IDS[i] << " raster to '" <<  filename << "'\n";
				write2DArray<uint8_t>(filename, this->rasters[i], row_size, this->rast_cell_nums[i]);
			}
		};
	}
}

void Control::initialize_psths()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!pf_names[i].empty() || use_gui)
			// TODO: make data type bigger for psth
			psths[i] = allocate2DArray<uint8_t>(PSTHColSize, rast_cell_nums[i]);
	}
	psth_arrays_initialized = true;
}

void Control::runSession(struct gui *gui)
{
	double start, end;
	int goSpkCounter[num_go];
	if (!use_gui) run_state = IN_RUN_NO_PAUSE;
	trial = 0;
	raster_counter = 0;
	while (trial < td.num_trials && run_state != NOT_IN_RUN)
	{
		std::string trialName = td.trial_names[trial];

		uint32_t useCS        = td.use_css[trial];
		uint32_t onsetCS      = td.cs_onsets[trial];
		uint32_t csLength     = td.cs_lens[trial];
		//uint32_t percentCS    = td.cs_percents[trial]; // unused for now
		uint32_t useUS        = td.use_uss[trial];
		uint32_t onsetUS      = td.us_onsets[trial];

		int PSTHCounter = 0;
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;

		memset(goSpkCounter, 0, num_go * sizeof(int));

		std::cout << "[INFO]: Trial number: " << trial + 1 << "\n";
		start = omp_get_wtime();
		for (uint32_t ts = 0; ts < trialTime; ts++)
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
				countGOSpikes(goSpkCounter);
				std::cout << "[INFO]: Mean gGRGO   = " << gGRGO_sum / (num_go * csLength) << "\n";
				std::cout << "[INFO]: Mean gMFGO   = " << gMFGO_sum / (num_go * csLength) << "\n";
				std::cout << "[INFO]: GR:MF ratio  = " << gGRGO_sum / gMFGO_sum << "\n";
			}
			
			/* data collection */
			if (ts >= onsetCS - msPreCS && ts < onsetCS + csLength + msPostCS)
			{
				fill_rasters(raster_counter, PSTHCounter);
				fill_psths(PSTHCounter);
				PSTHCounter++;
				raster_counter++;
			}

			if (use_gui)
			{
				update_spike_sums(ts, onsetCS, onsetCS + csLength);
				if (gtk_events_pending()) gtk_main_iteration();
			}
		}
		end = omp_get_wtime();
		std::cout << "[INFO]: '" << trialName << "' took " << (end - start) << "s.\n";
		
		if (use_gui)
		{
			// for now, compute the mean and median firing rates for all cells if win is visible
			if (firing_rates_win_visible(gui))
			{
				calculate_firing_rates(onsetCS, onsetCS + csLength);
				gdk_threads_add_idle((GSourceFunc)update_fr_labels, gui);
			}
			if (run_state == IN_RUN_PAUSE)
			{
				std::cout << "[INFO]: Simulation is paused at end of trial " << trial+1 << ".\n";
				while (run_state == IN_RUN_PAUSE)
				{
					if (gtk_events_pending()) gtk_main_iteration();
				}
				std::cout << "[INFO]: Continuing...\n";
			}
			reset_spike_sums();
		}
		// save gr rasters into new file every trial 
		//std::string trial_gr_raster_name = OUTPUT_DATA_PATH + get_file_basename(rf_names[GR])
		//							  + "_trial_" + std::to_string(trial) + "." + BIN_EXT;
		//save_raster(trial_gr_raster_name, GR);
		trial++;
	}
	if (run_state == NOT_IN_RUN) std::cout << "[INFO]: Simulation terminated.\n";
	else if (run_state == IN_RUN_NO_PAUSE) std::cout << "[INFO]: Simulation Completed.\n";
	
	if (!use_gui)
	{
		save_rasters();
		save_psths();
		// TODO: place all next files into class attrib,
		// so that once fully develop -o opt in run mode and tui or gui mode,
		// calls will implicitly use this class's private attribs, as we are moving
		// away from file dialogue in favour of auto-generated file names
		save_pfpc_weights_to_file(pf_pc_weights_file);
		save_mfdcn_weights_to_file(mf_nc_weights_file);
		save_sim_to_file(out_sim_name);
	}

	run_state = NOT_IN_RUN;
}

void Control::reset_spike_sums()
{
		for (int i = 0; i < NUM_CELL_TYPES; i++)
		{
			spike_sums[i].cs_spike_sum = 0;
			spike_sums[i].non_cs_spike_sum = 0;
			memset(spike_sums[i].cs_spike_counter, 0, rast_cell_nums[i] * sizeof(uint32_t));
			memset(spike_sums[i].non_cs_spike_counter, 0, rast_cell_nums[i] * sizeof(uint32_t));
		}
}

void Control::reset_rasters()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty() || use_gui)
		{
			uint32_t row_size = (CELL_IDS[i] == "GR") ? PSTHColSize : PSTHColSize * td.num_trials;
			memset(rasters[i][0], '\000', row_size * rast_cell_nums[i] * sizeof(uint8_t));
		}
	}
}

void Control::reset_psths()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!pf_names[i].empty())
		{
			memset(psths[i][0], '\000', rast_cell_nums[i] * PSTHColSize * sizeof(uint8_t));
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

void Control::save_rasters()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty())
			rast_save_funcs[i](rf_names[i]);
	}
}

void Control::save_psths()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!pf_names[i].empty())
			psth_save_funcs[i](pf_names[i]);
	}
}

void Control::update_spike_sums(int tts, float onset_cs, float offset_cs)
{
	// update cs spikes
	if (tts >= onset_cs && tts < offset_cs)
	{
		for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
		{
			for (uint32_t j = 0; j < rast_cell_nums[i]; j++)
			{
				spike_sums[i].cs_spike_sum += cell_spikes[i][j];
				spike_sums[i].cs_spike_counter[j] += cell_spikes[i][j];
			}
		}
	}
	// update non-cs spikes
	else if (tts < onset_cs)
	{
		for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
		{
			for (uint32_t j = 0; j < rast_cell_nums[i]; j++)
			{
				spike_sums[i].non_cs_spike_sum += cell_spikes[i][j];
				spike_sums[i].non_cs_spike_counter[j] += cell_spikes[i][j];
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
		std::sort(spike_sums[i].cs_spike_counter,
			spike_sums[i].cs_spike_counter + rast_cell_nums[i]);
		std::sort(spike_sums[i].non_cs_spike_counter,
			spike_sums[i].non_cs_spike_counter + rast_cell_nums[i]);
		
		// calculate medians
		firing_rates[i].non_cs_median_fr =
			(spike_sums[i].non_cs_spike_counter[rast_cell_nums[i] / 2 - 1]
		   + spike_sums[i].non_cs_spike_counter[rast_cell_nums[i] / 2]) / (2.0 * non_cs_time_secs);
		firing_rates[i].cs_median_fr     =
			(spike_sums[i].cs_spike_counter[rast_cell_nums[i] / 2 - 1]
		   + spike_sums[i].cs_spike_counter[rast_cell_nums[i] / 2]) / (2.0 * cs_time_secs);
		
		// calculate means
		firing_rates[i].non_cs_mean_fr = spike_sums[i].non_cs_spike_sum / (non_cs_time_secs * rast_cell_nums[i]);
		firing_rates[i].cs_mean_fr     = spike_sums[i].cs_spike_sum / (cs_time_secs * rast_cell_nums[i]);
	}
}

void Control::countGOSpikes(int *goSpkCounter)
{
	float isi = (td.us_onsets[0] - td.cs_onsets[0]) / 1000.0;
	std::sort(goSpkCounter, goSpkCounter + num_go);
	
	float m = (goSpkCounter[num_go / 2 - 1] + goSpkCounter[num_go / 2]) / 2.0;
	float goSpkSum = 0;

	for (int i = 0; i < num_go; i++) goSpkSum += goSpkCounter[i];

	std::cout << "[INFO]: Mean GO Rate: " << goSpkSum / ((float)num_go * isi) << std::endl;
	std::cout << "[INFO]: Median GO Rate: " << m / isi << std::endl;
}

void Control::fill_rasters(uint32_t raster_counter, uint32_t psth_counter)
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		uint32_t temp_counter = raster_counter;
		if (!rf_names[i].empty() || use_gui)
		{
			/* GR spikes are only spikes not saved on host every time step:
			 * InNet::exportAPGR makes cudaMemcpy call before returning pointer to mem address */
			if (CELL_IDS[i] == "GR")
			{
				cell_spikes[i] = simCore->getInputNet()->exportAPGR();
				temp_counter = psth_counter;
			}
			for (uint32_t j = 0; j < rast_cell_nums[i]; j++)
			{
				rasters[i][temp_counter][j] = cell_spikes[i][j];
			}
		}
	}

	if (use_gui)
	{
		const float* vm_pc = simCore->getMZoneList()[0]->exportVmPC();
		for (int i = 0; i < num_pc; i++)
		{
			pc_vm_raster[psth_counter][i] = vm_pc[i];
		}
		const float* vm_io = simCore->getMZoneList()[0]->exportVmIO();
		for (int i = 0; i < num_io; i++)
		{
			io_vm_raster[psth_counter][i] = vm_io[i];
		}
		const float* vm_nc = simCore->getMZoneList()[0]->exportVmNC();
		for (int i = 0; i < num_nc; i++)
		{
			nc_vm_raster[psth_counter][i] = vm_nc[i];
		}
	}
}

void Control::fill_psths(uint32_t psth_counter)
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!pf_names[i].empty() || use_gui)
		{
			for (uint32_t j = 0; j < rast_cell_nums[i]; j++)
			{
				psths[i][psth_counter][j] += cell_spikes[i][j];
			}
		}
	}
}

void Control::delete_spike_sums()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		free(spike_sums[i].non_cs_spike_counter);
		free(spike_sums[i].cs_spike_counter);
	}
}

void Control::delete_rasters()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty() || use_gui) delete2DArray<uint8_t>(rasters[i]);
	}
	if (use_gui)
	{
		delete2DArray<float>(pc_vm_raster);
		delete2DArray<float>(nc_vm_raster);
		delete2DArray<float>(io_vm_raster);
	}
}

void Control::delete_psths()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!pf_names[i].empty()) delete2DArray<uint8_t>(psths[i]);
	}
}

