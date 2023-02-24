#include <sys/stat.h> // mkdir (POSIX ONLY)
#include <iomanip>
#include <gtk/gtk.h>

#include "logger.h"
#include "control.h"
#include "file_parse.h"
#include "tty.h"
#include "array_util.h"
#include "gui.h" /* tenuous inclide at best :pogO: */

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
		data_out_path = OUTPUT_DATA_PATH + p_cl.output_basename;
		data_out_base_name = p_cl.output_basename;
		int status = mkdir(data_out_path.c_str(), 0775);
		if (status == -1)
		{
			LOG_FATAL("Could not create directory '%s'. Maybe it already exists. Exiting...", data_out_path.c_str());
			exit(10);
		}
		data_out_dir_created = true;
		create_out_sim_filename(); //default
	}
	else if (!p_cl.session_file.empty())
	{
		initialize_session(p_cl.session_file);
		cp_to_info_file_data(p_cl, s_file, if_data);
		set_plasticity_modes(p_cl.pfpc_plasticity, p_cl.mfnc_plasticity);
		// assume that validated commandline opts includes 1) input file 2) session file 3) output directory name
		data_out_path = OUTPUT_DATA_PATH + p_cl.output_basename;
		data_out_base_name = p_cl.output_basename;
		data_out_run_name = data_out_path + "/run_" + std::to_string(run_num);
		// NOTE: make the output directory here, so in case of error, user not
		// run an entire simulation just to not have files save
		int status = mkdir(data_out_path.c_str(), 0775);
		if (status == -1)
		{
			LOG_FATAL("Could not create directory '%s'. Maybe it already exists. Exiting...", data_out_path.c_str());
			exit(10);
		}
		data_out_dir_created = true;
		create_out_sim_filename(); //default
		create_out_info_filename(); //default
		create_raster_filenames(p_cl.raster_files); //optional
		create_psth_filenames(p_cl.psth_files); //optional
		create_weights_filenames(p_cl.weights_files); //optional
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
	else if (mfnc_plasticity == "binary") mf_nc_plast = BINARY;
	else if (mfnc_plasticity == "cascade") mf_nc_plast = CASCADE;
}

void Control::initialize_session(std::string sess_file)
{
	LOG_DEBUG("Initializing session...");
	tokenized_file t_file;
	lexed_file l_file;
	tokenize_file(sess_file, t_file);
	lex_tokenized_file(t_file, l_file);
	parse_lexed_sess_file(l_file, s_file);
	translate_parsed_trials(s_file, td);

	trialTime   = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["trialTime"].value);
	msPreCS     = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["msPreCS"].value);
	msPostCS    = std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["msPostCS"].value);
	msMeasure = msPreCS + td.cs_lens[0] + msPostCS;

	trials_data_initialized = true;
	LOG_DEBUG("Session initialized.");
}

void Control::init_sim(std::string in_sim_filename)
{
	LOG_DEBUG("Initializing simulation...");
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
	initialize_raster_save_funcs();
	initialize_psth_save_funcs();
	initialize_rasters();
	initialize_psths();
	initialize_spike_sums();
	sim_file_buf.close();
	sim_initialized = true;
	LOG_DEBUG("Simulation initialized.");
}


void Control::reset_sim()
{
	// move previous run to "{data_out}/run_(i-1)/"
	int status = mkdir(data_out_run_name.c_str(), 0775);
	if (status == -1)
	{
		LOG_FATAL("Could not create directory '%s'. Maybe it already exists. Exiting...", data_out_run_name.c_str());
		exit(10);
	}

	struct dirent *dp;
	char data_out_path_abs[64];
	realpath(data_out_path.c_str(), data_out_path_abs);
	std::string abs_out_path_cpp_str = std::string(data_out_path_abs);
	DIR *dir = opendir(abs_out_path_cpp_str.c_str());
	if (!dir)
	{
		//LOG_FATAL("Could not find output directory '%s'", abs_out_path_cpp_str.c_str());
		exit(11);
	}
	while ((dp = readdir(dir)))
	{
		if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
		{
			printf("%s\n", dp->d_name);
		}
	}

	// update current output name to "{data_out}/run_(i)"
	run_num++;
	data_out_run_name = "run_" + std::to_string(run_num);
	status = mkdir(data_out_run_name.c_str(), 0775);
	if (status == -1)
	{
		LOG_FATAL("Could not create directory '%s'. Maybe it already exists. Exiting...", data_out_run_name.c_str());
		exit(12);
	}

	//simState->readState(sim_file_buf);
	//// TODO: simCore, mfFreq, mfs
	//
	//reset_rasters();
	//reset_psths();
	//reset_spike_sums();
}

void Control::save_sim_to_file()
{
	if (out_sim_filename_created)
	{
		LOG_DEBUG("Saving simulation to file...");
		std::fstream outSimFileBuffer(out_sim_name.c_str(), std::ios::out | std::ios::binary);
		write_con_params(outSimFileBuffer);
		if (!simCore) simState->writeState(outSimFileBuffer);
		else simCore->writeState(outSimFileBuffer);
		outSimFileBuffer.close();
	}
}

void Control::write_header_info(std::fstream &out_buf)
{
	uint32_t col_1_remaining = HEADER_COL_1_WIDTH-RUN_START_DATE_LBL.length(); 
	out_buf << "########################### BEGIN SESSION RECORD #############################\n"; 
	out_buf << "#" << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
	out_buf << "#" << std::setw(1) << "" << RUN_START_DATE_LBL << std::setw(col_1_remaining)
			  << std::right << DEFAULT_DATE_FORMAT << " : " << std::setw(HEADER_COL_2_WIDTH)
			  << std::left << if_data.start_date << "#\n";

	col_1_remaining = HEADER_COL_1_WIDTH - RUN_START_TIME_LBL.length();
	out_buf << "#" << std::setw(1) << "" << RUN_START_TIME_LBL << std::setw(col_1_remaining) << std::right
			  << (if_data.locale + " " + DEFAULT_TIME_FORMAT) << " : " << std::setw(HEADER_COL_2_WIDTH)
			  << std::left << if_data.start_time << "#\n";

	col_1_remaining = HEADER_COL_1_WIDTH - RUN_END_DATE_LBL.length();
	out_buf << "#" << std::setw(1) << "" << RUN_END_DATE_LBL << std::setw(col_1_remaining) << std::right 
			  << DEFAULT_DATE_FORMAT << " : " << std::setw(HEADER_COL_2_WIDTH) << std::left
			  << if_data.end_date << "#\n";

	col_1_remaining = HEADER_COL_1_WIDTH - RUN_END_TIME_LBL.length();
	out_buf << "#" << std::setw(1) << "" << RUN_END_TIME_LBL << std::setw(col_1_remaining) << std::right
			  << (if_data.locale + " " + DEFAULT_TIME_FORMAT) << " : "
			  << std::setw(HEADER_COL_2_WIDTH) << std::left << if_data.end_time << "#\n";

	col_1_remaining = HEADER_COL_1_WIDTH - CBM_SIM_VER_LBL.length();
	out_buf << "#" << std::setw(1) << "" << CBM_SIM_VER_LBL << std::setw(col_1_remaining) << "" 
			  << " : " << std::setw(HEADER_COL_2_WIDTH) << std::left << if_data.sim_version << "#\n";

	col_1_remaining = HEADER_COL_1_WIDTH - USERNAME_LBL.length();
	out_buf << "#" << std::setw(1) << "" << USERNAME_LBL << std::setw(col_1_remaining) << ""
			  << " : " << std::setw(HEADER_COL_2_WIDTH) << std::left << if_data.username << "#\n";

	out_buf << "#" << std::right << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
	out_buf << "#" << std::setfill('#') << std::setw(INFO_FILE_COL_WIDTH-1) << "\n";
	out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
}

void Control::write_cmdline_info(std::fstream &out_buf)
{
	uint32_t col_1_remaining;
	out_buf << "############################ COMMANDLINE INFO ################################\n";
	out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - CMD_LBL.length();
	out_buf << "#" << std::setw(1) << "" << CMD_LBL << std::setw(col_1_remaining) << std::right << " : "
			  << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left << if_data.p_cl.cmd_name << "#\n";

	col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - VIS_MODE_LBL.length();
	out_buf << "#" << std::setw(1) << "" << VIS_MODE_LBL << std::setw(col_1_remaining) << std::right
			  << " : " << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left << if_data.p_cl.vis_mode << "#\n";

	col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - IN_FILE_LBL.length();
	out_buf << "#" << std::setw(1) << "" << IN_FILE_LBL << std::setw(col_1_remaining) << std::right
			  << " : " << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
			  << strip_file_path(if_data.p_cl.input_sim_file) << "#\n";

	col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - SESS_FILE_LBL.length();
	out_buf << "#" << std::setw(1) << "" << SESS_FILE_LBL << std::setw(col_1_remaining) << std::right
			  << " : " << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
			  << strip_file_path(if_data.p_cl.session_file) << "#\n";

	col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - OUT_DIR_LBL.length();
	out_buf << "#" << std::setw(1) << "" << OUT_DIR_LBL << std::setw(col_1_remaining) << std::right
			  << " : " << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
			  << if_data.p_cl.output_sim_file << "#\n";

	col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - PFPC_PLAST_LBL.length();
	out_buf << "#" << std::setw(1) << "" << PFPC_PLAST_LBL << std::setw(col_1_remaining) << std::right
			  << " : " << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left << if_data.p_cl.pfpc_plasticity << "#\n";

	col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - MFNC_PLAST_LBL.length();
	out_buf << "#" << std::setw(1) << "" << MFNC_PLAST_LBL << std::setw(col_1_remaining) << std::right
			  << " : " << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left << if_data.p_cl.mfnc_plasticity << "#\n";

	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	uint32_t col_2_width = INFO_FILE_COL_WIDTH - FILE_SAVE_LBL.length() - 5;
	out_buf << "#" << std::left << std::setw(1) << "" << FILE_SAVE_LBL << std::setw(1) << std::right
			  << " : " << std::setw(col_2_width) << "#\n";

	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	col_2_width = INFO_FILE_COL_WIDTH - RAST_FILE_LBL.length() - TAB_WIDTH - 4;
	out_buf << "#" << std::setw(TAB_WIDTH) << "" << RAST_FILE_LBL << std::setw(1) << std::right
			  << " : " << std::setw(col_2_width) << "#\n";
	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	col_2_width = INFO_FILE_COL_WIDTH - 2 * TAB_WIDTH - 8;
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty())
		{
			out_buf << "#" << std::setw(2*TAB_WIDTH) << "" << std::left << CELL_IDS[i] << " : " << std::left
					  << std::setw(col_2_width) << strip_file_path(rf_names[i]) << "#\n";
		}
	}

	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	col_2_width = INFO_FILE_COL_WIDTH - PSTH_FILE_LBL.length() - TAB_WIDTH - 4;
	out_buf << "#" << std::setw(TAB_WIDTH) << "" << PSTH_FILE_LBL << std::setw(1) << std::right
			  << " : " << std::setw(col_2_width) << "#\n";
	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	col_2_width = INFO_FILE_COL_WIDTH - PSTH_FILE_LBL.length() - TAB_WIDTH - 8;
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!pf_names[i].empty())
		{
			out_buf << "#" << std::setw(2*TAB_WIDTH) << "" << std::left << CELL_IDS[i] << " : " << std::left
					  << std::setw(col_2_width) << strip_file_path(pf_names[i]) << "#\n";
		}
	}
	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	col_2_width = INFO_FILE_COL_WIDTH - WEIGHTS_FILE_LBL.length() - TAB_WIDTH - 4;
	out_buf << "#" << std::setw(TAB_WIDTH) << "" << WEIGHTS_FILE_LBL << std::setw(1) << std::right
			  << " : " << std::setw(col_2_width) << "#\n";
	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	if (!pfpc_weights_file.empty())
	{
		col_2_width = INFO_FILE_COL_WIDTH - PFPC_FILE_LBL.length() - TAB_WIDTH - 10;
		out_buf << "#" << std::setw(2*TAB_WIDTH) << "" << std::left << PFPC_FILE_LBL << " : " << std::left
				  << std::setw(col_2_width) << strip_file_path(pfpc_weights_file) << "#\n";
	}
	if (!mfnc_weights_file.empty())
	{
		col_2_width = INFO_FILE_COL_WIDTH - MFNC_FILE_LBL.length() - TAB_WIDTH - 10;
		out_buf << "#" << std::setw(2*TAB_WIDTH) << "" << std::left << MFNC_FILE_LBL << " : " << std::left
				<< std::setw(col_2_width) << strip_file_path(mfnc_weights_file) << "#\n";
	}

	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
	out_buf << "#" << std::setfill('#') << std::setw(INFO_FILE_COL_WIDTH-1) << "\n";
	out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
}

void Control::write_sess_info(std::fstream &out_buf)
{
	out_buf << "############################## SESSION INFO ##################################\n";
	out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	uint32_t col_2_width = INFO_FILE_COL_WIDTH - TRIAL_DEFINE_LBL.length() - TAB_WIDTH - 1;
	out_buf << "#" << std::left << std::setw(1) << "" << TRIAL_DEFINE_LBL << std::setw(1) << std::right
			  << " : " << std::setw(col_2_width) << "#\n";
	out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	for (auto trial : if_data.s_file.parsed_trial_info.trial_map)
	{
		col_2_width = INFO_FILE_COL_WIDTH - trial.first.length() - TAB_WIDTH - 4;
		out_buf << "#" << std::setw(TAB_WIDTH) << "" << std::left << trial.first
				  << std::right << " : " << std::setw(col_2_width) << "#\n";
		out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

		uint32_t max_trial_param_len = get_max_key_len(trial.second);

		for (auto var : trial.second)
		{
			col_2_width = INFO_FILE_COL_WIDTH - max_trial_param_len - 2 * TAB_WIDTH - 6;
			out_buf << "#" << std::setw(2*TAB_WIDTH) << "" << std::left << std::setw(max_trial_param_len)
					  << var.first << std::right << " : " << std::left << std::setw(col_2_width) << var.second.value << "#\n";
		}
		out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
	}

	col_2_width = INFO_FILE_COL_WIDTH - BLOCK_DEFINE_LBL.length() - 5;
	out_buf << "#" << std::setw(1) << "" << BLOCK_DEFINE_LBL << std::setw(1) << std::right
			  << " : " << std::setw(col_2_width) << "#\n";
	out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

	for (auto block : if_data.s_file.parsed_trial_info.block_map)
	{
		col_2_width = INFO_FILE_COL_WIDTH - block.first.length() - TAB_WIDTH - 4;
		out_buf << "#" << std::setw(TAB_WIDTH) << "" << std::left << block.first << " : "
				  << std::right << std::setw(col_2_width) << "#\n";
		out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 

		uint32_t max_block_param_len = get_max_first_len(block.second);

		for (auto pair : block.second)
		{
			col_2_width = INFO_FILE_COL_WIDTH - max_block_param_len - 2 * TAB_WIDTH - 6;
			out_buf << "#" << std::setw(2*TAB_WIDTH) << "" << std::left << std::setw(max_block_param_len)
					  << pair.first << std::right << " : " << std::left << std::setw(col_2_width) << pair.second << "#\n";
		}
		out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
	}

	col_2_width = INFO_FILE_COL_WIDTH - SESSION_DEFINE_LBL.length() - 5;
	out_buf << "#" << std::setw(1) << "" << SESSION_DEFINE_LBL << std::setw(1) << std::right
			  << " : " << std::setw(col_2_width) << "#\n";
	out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
	for (auto pair : if_data.s_file.parsed_trial_info.session)
	{
		col_2_width = INFO_FILE_COL_WIDTH - pair.first.length() - TAB_WIDTH - 6;
		out_buf << "#" << std::setw(TAB_WIDTH) << "" << std::left << std::setw(pair.first.length())
				  << pair.first << std::right << " : " << std::left << std::setw(col_2_width) << pair.second << "#\n";
	}
	out_buf << "#" << std::right << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH-1) << "#\n"; 
	out_buf << "############################ END SESSION RECORD ##############################\n";
}

void Control::save_info_to_file()
{
	std::fstream out_if_data_buf(out_info_name.c_str(), std::ios::out);
	write_header_info(out_if_data_buf);
	write_cmdline_info(out_if_data_buf);
	write_sess_info(out_if_data_buf);
	out_if_data_buf.close();
}

void Control::save_pfpc_weights_to_file(int32_t trial)
{
	if (pfpc_weights_filenames_created)
	{
		std::string curr_pfpc_weights_filename = pfpc_weights_file;
		if (trial != -1) // nonnegative indicates we want to append the trial to the file basename
		{
			curr_pfpc_weights_filename = data_out_path + "/" + get_file_basename(pfpc_weights_file)
									   + "_TRIAL_" + std::to_string(trial) + BIN_EXT;
		}
		LOG_DEBUG("Saving granule to purkinje weights to file...");
		if (!simCore)
		{
			LOG_ERROR("Trying to write uninitialized weights to file.");
			LOG_ERROR("(Hint: Try initializing a sim or loading the weights first.)");
			return;
		}
		const float *pfpc_weights = simCore->getMZoneList()[0]->exportPFPCWeights();
		std::fstream outPFPCFileBuffer(curr_pfpc_weights_filename.c_str(), std::ios::out | std::ios::binary);
		rawBytesRW((char *)pfpc_weights, num_gr * sizeof(float), false, outPFPCFileBuffer);
		outPFPCFileBuffer.close();
	}
}

void Control::load_pfpc_weights_from_file(std::string in_pfpc_file)
{
	if (!simCore)
	{
		LOG_ERROR("Trying to read weights to uninitialized simulation.");
		LOG_ERROR("(Hint: Try initializing a sim first.)");
		return;
	}
	std::fstream inPFPCFileBuffer(in_pfpc_file.c_str(), std::ios::in | std::ios::binary);
	simCore->getMZoneList()[0]->load_pfpc_weights_from_file(inPFPCFileBuffer);
	inPFPCFileBuffer.close();
} 

void Control::save_mfdcn_weights_to_file(int32_t trial)
{
	if (mfnc_weights_filenames_created)
	{
		std::string curr_mfnc_weights_filename = mfnc_weights_file;
		if (trial != -1) // nonnegative indicates we want to append the trial to the file basename
		{
			curr_mfnc_weights_filename = data_out_path + "/" + get_file_basename(curr_mfnc_weights_filename)
									   + "_TRIAL_" + std::to_string(trial) + BIN_EXT;
		}
		LOG_DEBUG("Saving mossy fiber to deep nucleus weigths to file...");
		if (!simCore)
		{
			LOG_ERROR("Trying to write uninitialized weights to file.");
			LOG_ERROR("(Hint: Try initializing a sim or loading the weights first.)");
			return;
		}
		// TODO: make a export function for mfdcn weights
		const float *mfdcn_weights = simCore->getMZoneList()[0]->exportMFDCNWeights();
		std::fstream outMFDCNFileBuffer(curr_mfnc_weights_filename.c_str(), std::ios::out | std::ios::binary);
		rawBytesRW((char *)mfdcn_weights, num_nc * num_p_nc_from_mf_to_nc * sizeof(const float), false, outMFDCNFileBuffer);
		outMFDCNFileBuffer.close();
	}
}

void Control::load_mfdcn_weights_from_file(std::string in_mfdcn_file)
{
	if (!simCore)
	{
		LOG_ERROR("Trying to read weights to uninitialized simulation.");
		LOG_ERROR("(Hint: Try initializing a sim first.)");
		return;
	}
	std::fstream inMFDCNFileBuffer(in_mfdcn_file.c_str(), std::ios::in | std::ios::binary);
	simCore->getMZoneList()[0]->load_mfdcn_weights_from_file(inMFDCNFileBuffer);
	inMFDCNFileBuffer.close();
}

void Control::create_out_sim_filename()
{
	if (data_out_dir_created)
	{
		out_sim_name = data_out_path + "/"
					 + data_out_base_name + "_"
					 + get_current_time_as_string("%m%d%Y")
					 + SIM_EXT;
		out_sim_filename_created = true;
	}
}

void Control::create_out_info_filename()
{
	if (data_out_dir_created)
	{
		out_info_name = data_out_path + "/"
					  + data_out_base_name + "_"
					  + get_current_time_as_string("%m%d%Y")
					  + TXT_EXT;
		out_info_filename_created = true;
	}
}

// TODO: combine two below funcs into one for generality
void Control::create_raster_filenames(std::map<std::string, bool> &rast_map)
{
	if (data_out_dir_created)
	{
		for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
		{
			std::string cell_id = CELL_IDS[i];
			if (rast_map[cell_id] || use_gui)
			{
				rf_names[i] = data_out_path + "/" + data_out_base_name
											+ "_" + cell_id + "_RASTER_"
											+ get_current_time_as_string("%m%d%Y")
											+ BIN_EXT;
			}
		}
		raster_filenames_created = true;
	}
}

void Control::create_psth_filenames(std::map<std::string, bool> &psth_map)
{
	if (data_out_dir_created)
	{
		for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
		{
			std::string cell_id = CELL_IDS[i];
			if (psth_map[cell_id] || use_gui)
			{
				pf_names[i] = data_out_path + "/" + data_out_base_name 
							+ "_" + cell_id + "_PSTH_"
							+ get_current_time_as_string("%m%d%Y")
							+ BIN_EXT;
			}
		}
		psth_filenames_created = true;
	}
}

void Control::create_weights_filenames(std::map<std::string, bool> &weights_map) 
{
	if (data_out_dir_created)
	{
		if (weights_map["PFPC"] || use_gui)
		{
			pfpc_weights_file = data_out_path + "/" + data_out_base_name
							   + "_PFPC_WEIGHTS_" + get_current_time_as_string("%m%d%Y")
							   + BIN_EXT;
			pfpc_weights_filenames_created = true; // only useful so far for gui...
		}
		if (weights_map["MFNC"] || use_gui)
		{
			mfnc_weights_file = data_out_path + "/" + data_out_base_name
							   + "_MFNC_WEIGHTS_" + get_current_time_as_string("%m%d%Y")
							   + BIN_EXT;
			mfnc_weights_filenames_created = true; // only useful so far for gui...
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
			/* granules are saved every trial, so their raster size is msMeasure  x num_gr */
			uint32_t row_size = (CELL_IDS[i] == "GR") ? msMeasure : msMeasure * td.num_trials;
			rasters[i] = allocate2DArray<uint8_t>(row_size, rast_cell_nums[i]);
		}
	}

	if (use_gui)
	{
		// TODO: find a way to initialize only within gui mode
		pc_vm_raster = allocate2DArray<float>(msMeasure, num_pc);
		nc_vm_raster = allocate2DArray<float>(msMeasure, num_nc);
		io_vm_raster = allocate2DArray<float>(msMeasure, num_io);
	}

	raster_arrays_initialized = true;
}

void Control::initialize_psth_save_funcs()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		psth_save_funcs[i] = [this, i]()
		{
			if (!pf_names[i].empty())
			{
				LOG_DEBUG("Saving %s psth to file...", CELL_IDS[i].c_str());
				write2DArray<uint8_t>(pf_names[i], this->psths[i], this->msMeasure, this->rast_cell_nums[i]);
			}
		};
	}
}

void Control::initialize_raster_save_funcs()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		rast_save_funcs[i] = [this, i]()
		{
			if (!rf_names[i].empty() && CELL_IDS[i] != "GR")
			{
				uint32_t row_size = (CELL_IDS[i] == "GR") ? this->msMeasure : this->msMeasure * this->td.num_trials;
				LOG_DEBUG("Saving %s raster to file...", CELL_IDS[i].c_str());
				write2DArray<uint8_t>(rf_names[i], this->rasters[i], row_size, this->rast_cell_nums[i]);
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
			psths[i] = allocate2DArray<uint8_t>(msMeasure, rast_cell_nums[i]);
	}
	psth_arrays_initialized = true;
}

void Control::runSession(struct gui *gui)
{
	set_info_file_str_props(BEFORE_RUN, if_data);
	double start, end;
	int goSpkCounter[num_go];
	if (!use_gui) run_state = IN_RUN_NO_PAUSE;
	trial = 0;
	raster_counter = 0;
	while (trial < td.num_trials && run_state != NOT_IN_RUN)
	{
		std::string trialName = td.trial_names[trial];

		uint32_t useCS        = td.use_css[trial];
		uint32_t onsetCS      = pre_collection_ts + td.cs_onsets[trial];
		uint32_t csLength     = td.cs_lens[trial];
		//uint32_t percentCS    = td.cs_percents[trial]; // unused for now
		uint32_t useUS        = td.use_uss[trial];
		uint32_t onsetUS      = pre_collection_ts + td.us_onsets[trial];
		
		int PSTHCounter = 0;
		float gGRGO_sum = 0;
		float gMFGO_sum = 0;

		memset(goSpkCounter, 0, num_go * sizeof(int));

		LOG_INFO("Trial number: %d", trial + 1);
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
				LOG_DEBUG("Mean gGRGO   = %0.4f", gGRGO_sum / (num_go * csLength));
				LOG_DEBUG("Mean gMFGO   = %0.5f", gMFGO_sum / (num_go * csLength));
				LOG_DEBUG("GR:MF ratio  = %0.2f", gGRGO_sum / gMFGO_sum);
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
		LOG_INFO("'%s' took %0.2fs", trialName.c_str(), end - start);
		
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
				LOG_DEBUG("Simulation is paused at end of trial %d.", trial + 1);
				while(run_state == IN_RUN_PAUSE)
				{
					if (gtk_events_pending()) gtk_main_iteration();
				}
				LOG_DEBUG("Continuing...");
			}
			reset_spike_sums();
		}
		// save gr rasters into new file every trial 
		save_gr_raster();
		save_pfpc_weights_to_file(trial);
		save_mfdcn_weights_to_file(trial);
		trial++;
	}
	trial--; // setting so that is valid for drawing go rasters after a sim
	if (run_state == NOT_IN_RUN) LOG_INFO("Simulation terminated.");
	else if (run_state == IN_RUN_NO_PAUSE) LOG_INFO("Simulation Completed.");
	run_state = NOT_IN_RUN;
	set_info_file_str_props(AFTER_RUN, if_data);
	
	if (!use_gui)
	{
		save_rasters();
		save_psths();
		save_sim_to_file();
		save_info_to_file();
	}
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
			uint32_t row_size = (CELL_IDS[i] == "GR") ? msMeasure : msMeasure * td.num_trials;
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
			memset(psths[i][0], '\000', rast_cell_nums[i] * msMeasure * sizeof(uint8_t));
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

void Control::save_gr_raster()
{
	if (!rf_names[GR].empty())
	{
		std::string trial_raster_name = data_out_path + "/" + get_file_basename(rf_names[GR])
									  + "_trial_" + std::to_string(trial) + BIN_EXT;
		LOG_DEBUG("Saving granule raster to file...");
		write2DArray<uint8_t>(trial_raster_name, rasters[GR], num_gr, msMeasure);
	}
}

void Control::save_rasters()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!rf_names[i].empty())
			rast_save_funcs[i]();
	}
}

void Control::save_psths()
{
	for (uint32_t i = 0; i < NUM_CELL_TYPES; i++)
	{
		if (!pf_names[i].empty())
			psth_save_funcs[i]();
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

	LOG_DEBUG("Mean GO Rate: %0.2f", goSpkSum / ((float)num_go * isi));
	LOG_DEBUG("Median GO Rate: %0.1f", m / isi);
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
			/* GR spikes are only spikes not saved on host every time step:
			 * InNet::exportAPGR makes cudaMemcpy call before returning pointer to mem address */
			if (CELL_IDS[i] == "GR")
			{
				cell_spikes[i] = simCore->getInputNet()->exportAPGR();
			}
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

