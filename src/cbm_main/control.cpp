#include <iomanip>
#include <gtk/gtk.h>

#include "control.h"
#include "file_parse.h"
#include "tty.h"
#include "array_util.h"
#include "gui.h" /* tenuous inclide at best :pogO: */

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

		pf_pc_plast_on = (p_cl.pfpc_plasticity == "on") ? 1 : 0;
		mf_nc_plast_on = (p_cl.mfnc_plasticity == "on") ? 1 : 0;

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

void Control::save_sim_state_to_file(std::string outStateFile)
{
	if (!(con_params_populated && act_params_populated && simState))
	{
		fprintf(stderr, "[ERROR]: Trying to write an uninitialized state to file.\n");
		fprintf(stderr, "[ERROR]: (Hint: Try loading a sim file first.)\n");
		return;
	}
	std::fstream outStateFileBuffer(outStateFile.c_str(), std::ios::out | std::ios::binary);
	if (!simCore) simState->writeState(outStateFileBuffer);
	else simCore->writeState(outStateFileBuffer);
	outStateFileBuffer.close();
}

void Control::save_sim_to_file(std::string outSimFile)
{
	std::fstream outSimFileBuffer(outSimFile.c_str(), std::ios::out | std::ios::binary);
	write_con_params(outSimFileBuffer);
	//write_act_params(outSimFileBuffer);
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

void Control::save_gr_psth_to_file(std::string out_gr_psth_file)
{
	std::fstream out_gr_psth_file_buffer(out_gr_psth_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)sampleGRRaster, sample_gr_rast_size, false, out_gr_psth_file_buffer);
	out_gr_psth_file_buffer.close();
}

void Control::save_go_psth_to_file(std::string out_go_psth_file)
{
	std::fstream out_go_psth_file_buffer(out_go_psth_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)allGORaster, all_go_rast_size, false, out_go_psth_file_buffer);
	out_go_psth_file_buffer.close();
}

void Control::save_pc_psth_to_file(std::string out_pc_psth_file)
{
	std::fstream out_pc_psth_file_buffer(out_pc_psth_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)allPCRaster, all_pc_rast_size, false, out_pc_psth_file_buffer);
	out_pc_psth_file_buffer.close();
}

void Control::save_nc_psth_to_file(std::string out_nc_psth_file)
{
	std::fstream out_nc_psth_file_buffer(out_nc_psth_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)allNCRaster, all_nc_rast_size, false, out_nc_psth_file_buffer);
	out_nc_psth_file_buffer.close();
}

void Control::save_io_psth_to_file(std::string out_io_psth_file)
{
	std::fstream out_io_psth_file_buffer(out_io_psth_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)allIORaster, all_io_rast_size, false, out_io_psth_file_buffer);
	out_io_psth_file_buffer.close();
}

void Control::save_bc_psth_to_file(std::string out_bc_psth_file)
{
	std::fstream out_bc_psth_file_buffer(out_bc_psth_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)allBCRaster, all_bc_rast_size, false, out_bc_psth_file_buffer);
	out_bc_psth_file_buffer.close();
}

void Control::save_sc_psth_to_file(std::string out_sc_psth_file)
{
	std::fstream out_sc_psth_file_buffer(out_sc_psth_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)allSCRaster, all_sc_rast_size, false, out_sc_psth_file_buffer);
	out_sc_psth_file_buffer.close();
}

void Control::save_mf_psth_to_file(std::string out_mf_psth_file)
{
	std::fstream out_mf_psth_file_buffer(out_mf_psth_file.c_str(), std::ios::out | std::ios::binary);
	rawBytesRW((char *)allMFRaster, all_mf_rast_size, false, out_mf_psth_file_buffer);
	out_mf_psth_file_buffer.close();
}

void Control::get_raster_filenames(std::map<std::string, std::string> &raster_files)
{
	if (!raster_files.empty())
	{
		if (raster_files.find("MF") != raster_files.end())
		{
			mf_raster_file = raster_files["MF"];
		}
		if (raster_files.find("GR") != raster_files.end())
		{
			gr_raster_file = raster_files["GR"];
		}
		if (raster_files.find("GO") != raster_files.end())
		{
			go_raster_file = raster_files["GO"];
		}
		if (raster_files.find("BC") != raster_files.end())
		{
			bc_raster_file = raster_files["BC"];
		}
		if (raster_files.find("SC") != raster_files.end())
		{
			sc_raster_file = raster_files["SC"];
		}
		if (raster_files.find("PC") != raster_files.end())
		{
			pc_raster_file = raster_files["PC"];
		}
		if (raster_files.find("IO") != raster_files.end())
		{
			io_raster_file = raster_files["IO"];
		}
		if (raster_files.find("NC") != raster_files.end())
		{
			nc_raster_file = raster_files["NC"];
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

void Control::initialize_spike_sums()
{
	spike_sums[MF].num_cells  = num_mf;
	spike_sums[GR].num_cells  = num_gr;
	spike_sums[GO].num_cells  = num_go;
	spike_sums[BC].num_cells  = num_bc;
	spike_sums[SC].num_cells  = num_sc;
	spike_sums[PC].num_cells  = num_pc;
	spike_sums[IO].num_cells  = num_io;
	spike_sums[DCN].num_cells = num_nc;

	FOREACH(spike_sums, ssp)
	{
		ssp->non_cs_spike_sum = 0;
		ssp->cs_spike_sum     = 0;
		ssp->non_cs_spike_counter = new ct_uint32_t[ssp->num_cells];
		ssp->cs_spike_counter = new ct_uint32_t[ssp->num_cells];
		memset((void *)ssp->non_cs_spike_counter, 0, ssp->num_cells * sizeof(ct_uint32_t));
		memset((void *)ssp->cs_spike_counter, 0, ssp->num_cells * sizeof(ct_uint32_t));
	}
	spike_sums_initialized = true;
}

void Control::initialize_rast_internal()
{
	if (!mf_raster_file.empty())
	{
		all_mf_rast_internal    = allocate2DArray<ct_uint8_t>(num_mf, PSTHColSize);
	}
	if (!gr_raster_file.empty())
	{
		sample_gr_rast_internal = allocate2DArray<ct_uint8_t>(4096, PSTHColSize);
	}
	if (!go_raster_file.empty())
	{
		all_go_rast_internal    = allocate2DArray<ct_uint8_t>(num_go, PSTHColSize);
	}
	if (!bc_raster_file.empty())
	{
		all_bc_rast_internal    = allocate2DArray<ct_uint8_t>(num_bc, PSTHColSize);
	}
	if (!sc_raster_file.empty())
	{
		all_sc_rast_internal    = allocate2DArray<ct_uint8_t>(num_sc, PSTHColSize);
	}
	if (!pc_raster_file.empty())
	{
		all_pc_rast_internal    = allocate2DArray<ct_uint8_t>(num_pc, PSTHColSize);
	}
	if (!io_raster_file.empty())
	{
		all_io_rast_internal    = allocate2DArray<ct_uint8_t>(num_io, PSTHColSize);
	}
	if (!nc_raster_file.empty())
	{
		all_nc_rast_internal    = allocate2DArray<ct_uint8_t>(num_nc, PSTHColSize);
	}

	// TODO: find a way to initialize only within gui mode
	all_pc_vm_rast_internal = allocate2DArray<float>(num_pc, PSTHColSize);
	all_nc_vm_rast_internal = allocate2DArray<float>(num_nc, PSTHColSize);
	all_io_vm_rast_internal = allocate2DArray<float>(num_io, PSTHColSize);

	internal_arrays_initialized = true;
}

void Control::initializeOutputArrays()
{
	if (!mf_raster_file.empty())
	{
		all_mf_rast_size = num_mf * PSTHColSize * td.num_trials / BITS_PER_BYTE;
		allMFRaster      = (ct_uint8_t *)calloc(all_mf_rast_size, sizeof(ct_uint8_t));
	}
	if (!gr_raster_file.empty())
	{
		sample_gr_rast_size = 4096 * PSTHColSize * td.num_trials / BITS_PER_BYTE;
		sampleGRRaster      = (ct_uint8_t *)calloc(sample_gr_rast_size, sizeof(ct_uint8_t));
	}
	if (!go_raster_file.empty())
	{
		all_go_rast_size = num_go * PSTHColSize * td.num_trials / BITS_PER_BYTE;
		allGORaster      = (ct_uint8_t *)calloc(all_go_rast_size, sizeof(ct_uint8_t));
	}
	if (!bc_raster_file.empty())
	{
		all_bc_rast_size = num_bc * PSTHColSize * td.num_trials / BITS_PER_BYTE;
		allBCRaster      = (ct_uint8_t *)calloc(all_bc_rast_size, sizeof(ct_uint8_t));
	}
	if (!sc_raster_file.empty())
	{
		all_sc_rast_size = num_sc * PSTHColSize * td.num_trials / BITS_PER_BYTE;
		allSCRaster      = (ct_uint8_t *)calloc(all_sc_rast_size, sizeof(ct_uint8_t));
	}
	if (!pc_raster_file.empty())
	{
		all_pc_rast_size = num_pc * PSTHColSize * td.num_trials / BITS_PER_BYTE;
		allPCRaster      = (ct_uint8_t *)calloc(all_pc_rast_size, sizeof(ct_uint8_t));
	}
	if (!io_raster_file.empty())
	{
		all_io_rast_size = num_io * PSTHColSize * td.num_trials / BITS_PER_BYTE;
		allIORaster      = (ct_uint8_t *)calloc(all_io_rast_size, sizeof(ct_uint8_t));
	}
	if (!nc_raster_file.empty())
	{
		all_nc_rast_size = num_nc * PSTHColSize * td.num_trials / BITS_PER_BYTE;
		allNCRaster      = (ct_uint8_t *)calloc(all_nc_rast_size, sizeof(ct_uint8_t));
	}

	if (!pf_pc_weights_file.empty())
	{
	// TODO: find a way to initialize only within gui mode

		sample_pfpc_syn_weights = new float[4096];
	}
	output_arrays_initialized = true;
}

void save_arr_as_csv(float in_arr[], ct_uint32_t arr_len, std::string file_name)
{
	std::fstream out_file_buf(file_name.c_str(), std::ios::out);
	for (ct_uint32_t i = 0; i < arr_len; i++)
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

		ct_uint32_t useCS        = td.use_css[trial];
		ct_uint32_t onsetCS      = td.cs_onsets[trial];
		ct_uint32_t csLength     = td.cs_lens[trial];
		ct_uint32_t percentCS    = td.cs_percents[trial];
		ct_uint32_t useUS        = td.use_uss[trial];
		ct_uint32_t onsetUS      = td.us_onsets[trial];
		
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
			if (ts < onsetCS || ts >= onsetCS + csLength)
			{
				mfAP = mfs->calcPoissActivity(mfFreq->getMFBG(),
					  simCore->getMZoneList());
			}
			if (ts >= onsetCS && ts < onsetCS + csLength)
			{
				mfAP = (useCS == 1) ? mfs->calcPoissActivity(mfFreq->getMFInCSTonicA(), simCore->getMZoneList())
									: mfs->calcPoissActivity(mfFreq->getMFBG(), simCore->getMZoneList());
			}
			
			bool *isTrueMF = mfs->calcTrueMFs(mfFreq->getMFBG()); /* only used for mfdcn plasticity */
			simCore->updateTrueMFs(isTrueMF);
			simCore->updateMFInput(mfAP);
			simCore->calcActivity(spillFrac, pf_pc_plast_on); 
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
			memset((void *)(spike_sums[i].non_cs_spike_counter), 0, spike_sums[i].num_cells * sizeof(ct_uint32_t));
			memset((void *)(spike_sums[i].cs_spike_counter), 0, spike_sums[i].num_cells * sizeof(ct_uint32_t));
		}
}

void Control::reset_rast_internal()
{
	memset(all_mf_rast_internal[0], '\000', num_mf * PSTHColSize * sizeof(ct_uint8_t));
	memset(all_go_rast_internal[0], '\000', num_go * PSTHColSize * sizeof(ct_uint8_t));
	memset(sample_gr_rast_internal[0], '\000', 4096 * PSTHColSize * sizeof(ct_uint8_t));
	memset(all_pc_rast_internal[0], '\000', num_mf * PSTHColSize * sizeof(ct_uint8_t));
	memset(all_nc_rast_internal[0], '\000', num_mf * PSTHColSize * sizeof(ct_uint8_t));
	memset(all_sc_rast_internal[0], '\000', num_mf * PSTHColSize * sizeof(ct_uint8_t));
	memset(all_bc_rast_internal[0], '\000', num_mf * PSTHColSize * sizeof(ct_uint8_t));
	memset(all_io_rast_internal[0], '\000', num_mf * PSTHColSize * sizeof(ct_uint8_t));
}

void Control::resetOutputArrays()
{
	memset(allMFRaster, '\000', all_mf_rast_size * sizeof(ct_uint8_t));
	memset(allGORaster, '\000', all_go_rast_size * sizeof(ct_uint8_t));
	memset(sampleGRRaster, '\000', sample_gr_rast_size * sizeof(ct_uint8_t));
	memset(allPCRaster, '\000', all_pc_rast_size * sizeof(ct_uint8_t));
	memset(allNCRaster, '\000', all_nc_rast_size * sizeof(ct_uint8_t));
	memset(allSCRaster, '\000', all_sc_rast_size * sizeof(ct_uint8_t));
	memset(allBCRaster, '\000', all_bc_rast_size * sizeof(ct_uint8_t));
	memset(allIORaster, '\000', all_io_rast_size * sizeof(ct_uint8_t));

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
	if (!mf_raster_file.empty())
	{
		std::cout << "[INFO]: Filling MF files...\n";
		save_mf_psth_to_file(mf_raster_file);
	}
	if (!gr_raster_file.empty())
	{
		std::cout << "[INFO]: Filling GR file...\n";
		save_gr_psth_to_file(gr_raster_file);
	}
	if (!go_raster_file.empty())
	{
		std::cout << "[INFO]: Filling GO file...\n";
		save_go_psth_to_file(go_raster_file);
	}
	if (!bc_raster_file.empty())
	{
		std::cout << "[INFO]: Filling BC file...\n";
		save_bc_psth_to_file(bc_raster_file);
	}
	if (!sc_raster_file.empty())
	{
		std::cout << "[INFO]: Filling SC file...\n";
		save_sc_psth_to_file(sc_raster_file);
	}
	if (!pc_raster_file.empty())
	{
		std::cout << "[INFO]: Filling PC file...\n";
		save_pc_psth_to_file(pc_raster_file);
	}
	if (!io_raster_file.empty())
	{
		std::cout << "[INFO]: Filling IO file...\n";
		save_io_psth_to_file(io_raster_file);
	}
	if (!nc_raster_file.empty())
	{
		std::cout << "[INFO]: Filling NC file...\n";
		save_nc_psth_to_file(nc_raster_file);
	}
}

void Control::update_spike_sums(int tts, float onset_cs, float offset_cs)
{
	cell_spikes[MF]  = mfAP;
	cell_spikes[GR]  = simCore->getInputNet()->exportAPGR();
	cell_spikes[GO]  = simCore->getInputNet()->exportAPGO();
	cell_spikes[BC]  = simCore->getMZoneList()[0]->exportAPBC();
	cell_spikes[SC]  = simCore->getMZoneList()[0]->exportAPSC();
	cell_spikes[PC]  = simCore->getMZoneList()[0]->exportAPPC();
	cell_spikes[IO]  = simCore->getMZoneList()[0]->exportAPIO();
	cell_spikes[DCN] = simCore->getMZoneList()[0]->exportAPNC();

	// update cs spikes
	if (tts >= onset_cs && tts < offset_cs)
	{
		for (int i = 0; i < NUM_CELL_TYPES; i++)
		{
			for (int j = 0; j < spike_sums[i].num_cells; j++)
			{
				spike_sums[i].cs_spike_sum += cell_spikes[i][j];
				spike_sums[i].cs_spike_counter[j] += cell_spikes[i][j];
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
	if (!mf_raster_file.empty())
	{
		for (int i = 0; i < num_mf; i++)
		{
			all_mf_rast_internal[i][PSTHCounter] = mfAP[i];
		}
	}
	if (!go_raster_file.empty())
	{
		const ct_uint8_t* goSpks = simCore->getInputNet()->exportAPGO();
		for (int i = 0; i < num_go; i++)
		{
			all_go_rast_internal[i][PSTHCounter] = goSpks[i];
		}
	}
	if (!gr_raster_file.empty())
	{
		const ct_uint8_t* grSpks = simCore->getInputNet()->exportAPGR(); /* reading from gpu mem to non-pinned host mem is slow! just calling this slows down by ~30% ! */
		for (int i = 0; i < 4096; i++)
		{
			sample_gr_rast_internal[i][PSTHCounter] = grSpks[i];
		}
	}
	if (!pc_raster_file.empty())
	{
		const ct_uint8_t* pcSpks = simCore->getMZoneList()[0]->exportAPPC();
		for (int i = 0; i < num_pc; i++)
		{
			all_pc_rast_internal[i][PSTHCounter] = pcSpks[i];
		}
	}
	if (!nc_raster_file.empty())
	{
		const ct_uint8_t* ncSpks = simCore->getMZoneList()[0]->exportAPNC();
		for (int i = 0; i < num_nc; i++)
		{
			all_nc_rast_internal[i][PSTHCounter] = ncSpks[i];
		}
	}
	if (!sc_raster_file.empty())
	{
		const ct_uint8_t* scSpks = simCore->getMZoneList()[0]->exportAPSC();
		for (int i = 0; i < num_sc; i++)
		{
			all_sc_rast_internal[i][PSTHCounter] = scSpks[i];
		}
	}
	if (!bc_raster_file.empty())
	{
		const ct_uint8_t* bcSpks = simCore->getMZoneList()[0]->exportAPBC();
		for (int i = 0; i < num_bc; i++)
		{
			all_bc_rast_internal[i][PSTHCounter] = bcSpks[i];
		}
	}
	if (!io_raster_file.empty())
	{
		const ct_uint8_t* ioSpks = simCore->getMZoneList()[0]->exportAPIO();
		for (int i = 0; i < num_io; i++)
		{
			all_io_rast_internal[i][PSTHCounter] = ioSpks[i];
		}
	}
	if (!nc_raster_file.empty())
	{
		const ct_uint8_t* ncSpks = simCore->getMZoneList()[0]->exportAPNC();
		for (int i = 0; i < num_nc; i++)
		{
			all_nc_rast_internal[i][PSTHCounter] = ncSpks[i];
		}
	}

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
	if (!mf_raster_file.empty())
	{
		pack_2d_byte_array(all_mf_rast_internal, num_mf, PSTHColSize, allMFRaster, offset_common * num_mf);
	}
	if (!gr_raster_file.empty())
	{
		pack_2d_byte_array(sample_gr_rast_internal, 4096, PSTHColSize, sampleGRRaster, offset_common * 4096);
	}
	if (!go_raster_file.empty())
	{
		pack_2d_byte_array(all_go_rast_internal, num_go, PSTHColSize, allGORaster, offset_common * num_go);
	}
	if (!bc_raster_file.empty())
	{
		pack_2d_byte_array(all_bc_rast_internal, num_bc, PSTHColSize, allBCRaster, offset_common * num_bc);
	}
	if (!sc_raster_file.empty())
	{
		pack_2d_byte_array(all_sc_rast_internal, num_sc, PSTHColSize, allSCRaster, offset_common * num_sc);
	}
	if (!pc_raster_file.empty())
	{
		pack_2d_byte_array(all_pc_rast_internal, num_pc, PSTHColSize, allPCRaster, offset_common * num_pc);
	}
	if (!io_raster_file.empty())
	{
		pack_2d_byte_array(all_io_rast_internal, num_io, PSTHColSize, allIORaster, offset_common * num_io);
	}
	if (!nc_raster_file.empty())
	{
		pack_2d_byte_array(all_nc_rast_internal, num_nc, PSTHColSize, allNCRaster, offset_common * num_nc);
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
	if (!mf_raster_file.empty()) delete2DArray<ct_uint8_t>(all_mf_rast_internal);
	if (!gr_raster_file.empty()) delete2DArray<ct_uint8_t>(sample_gr_rast_internal);
	if (!go_raster_file.empty()) delete2DArray<ct_uint8_t>(all_go_rast_internal);
	if (!bc_raster_file.empty()) delete2DArray<ct_uint8_t>(all_bc_rast_internal);
	if (!sc_raster_file.empty()) delete2DArray<ct_uint8_t>(all_sc_rast_internal);
	if (!pc_raster_file.empty()) delete2DArray<ct_uint8_t>(all_pc_rast_internal);
	if (!io_raster_file.empty()) delete2DArray<ct_uint8_t>(all_io_rast_internal);
	if (!nc_raster_file.empty()) delete2DArray<ct_uint8_t>(all_nc_rast_internal);

	delete2DArray<float>(all_pc_vm_rast_internal);
	delete2DArray<float>(all_nc_vm_rast_internal);
	delete2DArray<float>(all_io_vm_rast_internal);
}

void Control::deleteOutputArrays()
{
	if (!mf_raster_file.empty()) free(allMFRaster);
	if (!gr_raster_file.empty()) free(sampleGRRaster);
	if (!go_raster_file.empty()) free(allGORaster);
	if (!bc_raster_file.empty()) free(allBCRaster);
	if (!sc_raster_file.empty()) free(allSCRaster);
	if (!pc_raster_file.empty()) free(allPCRaster);
	if (!io_raster_file.empty()) free(allIORaster);
	if (!nc_raster_file.empty()) free(allNCRaster);

	delete[] sample_pfpc_syn_weights;
}

