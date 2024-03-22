#include <gtk/gtk.h>
#include <iomanip>
#include <sys/stat.h> // mkdir (POSIX ONLY)

#include "array_util.h"
#include "control.h"
#include "file_parse.h"
#include "gui.h" /* tenuous inclide at best :pogO: */
#include "logger.h"

Control::Control(parsed_commandline &p_cl) {
  use_gui = (p_cl.vis_mode == "GUI") ? true : false;
  if (!p_cl.build_file.empty()) {
    /* initialize temporary objects for parsing build file */
    tokenized_file t_file;
    lexed_file l_file;
    parsed_build_file pb_file;
    tokenize_file(p_cl.build_file, t_file);
    lex_tokenized_file(t_file, l_file);
    parse_lexed_build_file(l_file, pb_file);
    if (!con_params_populated)
      populate_con_params(pb_file);
    data_out_path = OUTPUT_DATA_PATH + p_cl.output_basename;
    data_out_base_name = p_cl.output_basename;
    /* Next few lines create the output directory.
     * TODO: This might need refactoring. A utility function should do this.
     */
    int status = mkdir(data_out_path.c_str(), 0775);
    if (status == -1) {
      LOG_FATAL("Could not create directory '%s'. Maybe it already exists. "
                "Exiting...",
                data_out_path.c_str());
      exit(10);
    }
    data_out_dir_created = true;
    create_out_sim_filename();
    create_con_arrs_filenames(p_cl.conn_arrs_files);
  } else if (!p_cl.session_file.empty()) {
    initialize_session(p_cl.session_file);
    // cp session info to info file obj
    cp_to_info_file_data(p_cl, s_file, if_data);
    set_plasticity_modes(p_cl.pfpc_plasticity, p_cl.mfnc_plasticity);
    // assume that validated commandline opts includes 1) input file 2) session
    // file 3) output directory name
    data_out_path = OUTPUT_DATA_PATH + p_cl.output_basename;
    data_out_base_name = p_cl.output_basename;
    // NOTE: make the output directory here, so in case of error, user not
    // run an entire simulation just to not have files save
    int status = mkdir(data_out_path.c_str(), 0775);
    if (status == -1) {
      LOG_FATAL("Could not create directory '%s'. Maybe it already exists. "
                "Exiting...",
                data_out_path.c_str());
      exit(10);
    }
    data_out_dir_created = true;
    // create various output filenames once session is initialized
    create_out_sim_filename();                       // default
    create_out_info_filename();                      // default
    create_raster_filenames(p_cl.raster_files);      // optional
    create_psth_filenames(p_cl.psth_files);          // optional
    create_weights_filenames(p_cl.weights_files);    // optional
    create_con_arrs_filenames(p_cl.conn_arrs_files); // optional
    if (!p_cl.gr_psth_file.empty()) {
      in_gr_psth_filename = p_cl.gr_psth_file;
      use_gr_act_from_poiss = true;
    }
    init_sim(p_cl.input_sim_file);
    if (!p_cl.altered_weights_file.empty()) {
      load_pfpc_weights_from_file(p_cl.altered_weights_file); // optional
    }
    if (!p_cl.weight_mask_file.empty()) {
      load_pfpc_weight_mask_from_file(p_cl.weight_mask_file);
      use_pfpc_weight_mask = true;
    }
  } else if (!p_cl.conn_arrs_files.empty()) {
    data_out_path = OUTPUT_DATA_PATH + p_cl.output_basename;
    data_out_base_name = p_cl.output_basename;
    LOG_DEBUG("Using '%s' as the output directory...", data_out_path.c_str());
    int status = mkdir(data_out_path.c_str(), 0775);
    if (status == -1) {
      LOG_DEBUG("Could not create directory '%s'. Maybe it already exists. "
                "Exiting...",
                data_out_path.c_str());
      exit(10);
    }
    data_out_dir_created = true;
    create_con_arrs_filenames(p_cl.conn_arrs_files);
    std::fstream sim_file_buf(p_cl.input_sim_file.c_str(),
                              std::ios::in | std::ios::binary);
    read_con_params(sim_file_buf);
    simState = new CBMState(numMZones, sim_file_buf);
    sim_file_buf.close();
  } else { // user ran program with no args
    set_plasticity_modes("graded", "graded");
  }
}

Control::~Control() {
  // delete allocated trials_data memory
  if (trials_data_initialized)
    delete_trials_data(td);

  // delete all dynamic objects
  if (simState)
    delete simState;
  if (simCore)
    delete simCore;
  if (mfs)
    delete mfs;

  // deallocate output arrays
  if (raster_arrays_initialized)
    delete_rasters();
  if (psth_arrays_initialized)
    delete_psths();
  if (spike_sums_initialized)
    delete_spike_sums();
}

void Control::build_sim() {
  if (!simState)
    simState = new CBMState(numMZones);
}

void Control::set_plasticity_modes(std::string pfpc_plasticity,
                                   std::string mfnc_plasticity) {
  if (pfpc_plasticity == "off")
    pf_pc_plast = OFF;
  else if (pfpc_plasticity == "graded")
    pf_pc_plast = GRADED;
  else if (pfpc_plasticity == "binary")
    pf_pc_plast = BINARY;
  else if (pfpc_plasticity == "cascade")
    pf_pc_plast = CASCADE;

  if (mfnc_plasticity == "off")
    mf_nc_plast = OFF;
  else if (mfnc_plasticity == "graded")
    mf_nc_plast = GRADED;
  else if (mfnc_plasticity == "binary")
    mf_nc_plast = BINARY;
  else if (mfnc_plasticity == "cascade")
    mf_nc_plast = CASCADE;
}

void Control::initialize_session(std::string sess_file) {
  LOG_DEBUG("Initializing session...");
  // create temporary objects for parsing session file
  tokenized_file t_file;
  lexed_file l_file;
  tokenize_file(sess_file, t_file);
  lex_tokenized_file(t_file, l_file);
  parse_lexed_sess_file(l_file, s_file);
  // this function is required to turn object of objects into object of arrays
  // (performance benefit)
  translate_parsed_trials(s_file, td);

  // for now, manually use string to int for these parameters. clunky.
  trialTime = std::stoi(
      s_file.parsed_var_sections["trial_spec"].param_map["trialTime"]);
  msPreCS =
      std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["msPreCS"]);
  msPostCS =
      std::stoi(s_file.parsed_var_sections["trial_spec"].param_map["msPostCS"]);
  msMeasure = msPreCS + td.cs_lens[0] + msPostCS;

  trials_data_initialized = true;
  LOG_DEBUG("Session initialized.");
}

/**
 *  @details The order of what we read and when matters: connectivity params,
 *  activity params, then the simulation state are written to file in that
 *  order.
 */
void Control::init_sim(std::string in_sim_filename) {
  LOG_DEBUG("Initializing simulation...");
  std::fstream sim_file_buf(in_sim_filename.c_str(),
                            std::ios::in | std::ios::binary);
  read_con_params(sim_file_buf);
  populate_act_params(s_file);
  simState = new CBMState(numMZones, sim_file_buf);
  simCore = new CBMSimCore(simState, in_gr_psth_filename, gpuIndex, gpuP2);
  if (!use_gr_act_from_poiss) {
    mfs = new ECMFPopulation(num_mf, mfRandSeed, CSTonicMFFrac, CSPhasicMFFrac,
                             contextMFFrac, nucCollFrac, bgFreqMin, csbgFreqMin,
                             contextFreqMin, tonicFreqMin, phasicFreqMin,
                             bgFreqMax, csbgFreqMax, contextFreqMax,
                             tonicFreqMax, phasicFreqMax, collaterals_off,
                             fracImport, secondCS, fracOverlap, numMZones);
    simCore->setTrueMFs(mfs->getCollateralIds());
  }
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

/**
 *  @details This function has not been finished: still need to reset
 *  the sim_core, mfFreq, and mfs.
 */
void Control::reset_sim(std::string in_sim_filename) {
  std::fstream sim_file_buf(in_sim_filename.c_str(),
                            std::ios::in | std::ios::binary);
  read_con_params(sim_file_buf);
  // read_act_params(sim_file_buf);
  simState->readState(sim_file_buf);
  // TODO: simCore, mfs

  reset_rasters();
  reset_psths();
  reset_spike_sums();
  sim_file_buf.close();
  // TODO: more things to reset?
}

void Control::save_sim_to_file() {
  if (out_sim_filename_created) {
    LOG_DEBUG("Saving simulation to file...");
    std::fstream outSimFileBuffer(out_sim_name.c_str(),
                                  std::ios::out | std::ios::binary);
    write_con_params(outSimFileBuffer);
    if (!simCore)
      simState->writeState(outSimFileBuffer);
    else
      simCore->writeState(outSimFileBuffer, use_gr_act_from_poiss);
    outSimFileBuffer.close();
  }
}

/**
 *  @details This function is messy, but it's what I came up with. This and
 *  other related info file writing functions are rather rigid in terms of
 *  how the output info file is formatted.
 */
void Control::write_header_info(std::fstream &out_buf) {
  uint32_t col_1_remaining = HEADER_COL_1_WIDTH - RUN_START_DATE_LBL.length();
  out_buf << "########################### BEGIN SESSION RECORD "
             "#############################\n";
  out_buf << "#" << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";
  out_buf << "#" << std::setw(1) << "" << RUN_START_DATE_LBL
          << std::setw(col_1_remaining) << std::right << DEFAULT_DATE_FORMAT
          << " : " << std::setw(HEADER_COL_2_WIDTH) << std::left
          << if_data.start_date << "#\n";

  col_1_remaining = HEADER_COL_1_WIDTH - RUN_START_TIME_LBL.length();
  out_buf << "#" << std::setw(1) << "" << RUN_START_TIME_LBL
          << std::setw(col_1_remaining) << std::right
          << (if_data.locale + " " + DEFAULT_TIME_FORMAT) << " : "
          << std::setw(HEADER_COL_2_WIDTH) << std::left << if_data.start_time
          << "#\n";

  col_1_remaining = HEADER_COL_1_WIDTH - RUN_END_DATE_LBL.length();
  out_buf << "#" << std::setw(1) << "" << RUN_END_DATE_LBL
          << std::setw(col_1_remaining) << std::right << DEFAULT_DATE_FORMAT
          << " : " << std::setw(HEADER_COL_2_WIDTH) << std::left
          << if_data.end_date << "#\n";

  col_1_remaining = HEADER_COL_1_WIDTH - RUN_END_TIME_LBL.length();
  out_buf << "#" << std::setw(1) << "" << RUN_END_TIME_LBL
          << std::setw(col_1_remaining) << std::right
          << (if_data.locale + " " + DEFAULT_TIME_FORMAT) << " : "
          << std::setw(HEADER_COL_2_WIDTH) << std::left << if_data.end_time
          << "#\n";

  col_1_remaining = HEADER_COL_1_WIDTH - CBM_SIM_VER_LBL.length();
  out_buf << "#" << std::setw(1) << "" << CBM_SIM_VER_LBL
          << std::setw(col_1_remaining) << ""
          << " : " << std::setw(HEADER_COL_2_WIDTH) << std::left
          << if_data.sim_version << "#\n";

  col_1_remaining = HEADER_COL_1_WIDTH - USERNAME_LBL.length();
  out_buf << "#" << std::setw(1) << "" << USERNAME_LBL
          << std::setw(col_1_remaining) << ""
          << " : " << std::setw(HEADER_COL_2_WIDTH) << std::left
          << if_data.username << "#\n";

  out_buf << "#" << std::right << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";
  out_buf << "#" << std::setfill('#') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "\n";
  out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "#\n";
}

void Control::write_cmdline_info(std::fstream &out_buf) {
  uint32_t col_1_remaining;
  out_buf << "############################ COMMANDLINE INFO "
             "################################\n";
  out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "#\n";

  col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - CMD_LBL.length();
  out_buf << "#" << std::setw(1) << "" << CMD_LBL << std::setw(col_1_remaining)
          << std::right << " : " << std::setw(CMDLINE_SECTION_COL_2_WIDTH)
          << std::left << if_data.p_cl.cmd_name << "#\n";

  col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - VIS_MODE_LBL.length();
  out_buf << "#" << std::setw(1) << "" << VIS_MODE_LBL
          << std::setw(col_1_remaining) << std::right << " : "
          << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
          << if_data.p_cl.vis_mode << "#\n";

  col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - IN_FILE_LBL.length();
  out_buf << "#" << std::setw(1) << "" << IN_FILE_LBL
          << std::setw(col_1_remaining) << std::right << " : "
          << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
          << strip_file_path(if_data.p_cl.input_sim_file) << "#\n";

  col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - SESS_FILE_LBL.length();
  out_buf << "#" << std::setw(1) << "" << SESS_FILE_LBL
          << std::setw(col_1_remaining) << std::right << " : "
          << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
          << strip_file_path(if_data.p_cl.session_file) << "#\n";

  col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - OUT_DIR_LBL.length();
  out_buf << "#" << std::setw(1) << "" << OUT_DIR_LBL
          << std::setw(col_1_remaining) << std::right << " : "
          << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
          << if_data.p_cl.output_sim_file << "#\n";

  col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - PFPC_PLAST_LBL.length();
  out_buf << "#" << std::setw(1) << "" << PFPC_PLAST_LBL
          << std::setw(col_1_remaining) << std::right << " : "
          << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
          << if_data.p_cl.pfpc_plasticity << "#\n";

  col_1_remaining = CMDLINE_SECTION_COL_1_WIDTH - MFNC_PLAST_LBL.length();
  out_buf << "#" << std::setw(1) << "" << MFNC_PLAST_LBL
          << std::setw(col_1_remaining) << std::right << " : "
          << std::setw(CMDLINE_SECTION_COL_2_WIDTH) << std::left
          << if_data.p_cl.mfnc_plasticity << "#\n";

  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";

  uint32_t col_2_width = INFO_FILE_COL_WIDTH - FILE_SAVE_LBL.length() - 5;
  out_buf << "#" << std::left << std::setw(1) << "" << FILE_SAVE_LBL
          << std::setw(1) << std::right << " : " << std::setw(col_2_width)
          << "#\n";

  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";

  col_2_width = INFO_FILE_COL_WIDTH - RAST_FILE_LBL.length() - TAB_WIDTH - 4;
  out_buf << "#" << std::setw(TAB_WIDTH) << "" << RAST_FILE_LBL << std::setw(1)
          << std::right << " : " << std::setw(col_2_width) << "#\n";
  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";

  col_2_width = INFO_FILE_COL_WIDTH - 2 * TAB_WIDTH - 8;
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!rf_names[i].empty()) {
      out_buf << "#" << std::setw(2 * TAB_WIDTH) << "" << std::left
              << CELL_IDS[i] << " : " << std::left << std::setw(col_2_width)
              << strip_file_path(rf_names[i]) << "#\n";
    }
  }

  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";

  col_2_width = INFO_FILE_COL_WIDTH - PSTH_FILE_LBL.length() - TAB_WIDTH - 4;
  out_buf << "#" << std::setw(TAB_WIDTH) << "" << PSTH_FILE_LBL << std::setw(1)
          << std::right << " : " << std::setw(col_2_width) << "#\n";
  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";

  col_2_width = INFO_FILE_COL_WIDTH - PSTH_FILE_LBL.length() - TAB_WIDTH - 8;
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!pf_names[i].empty()) {
      out_buf << "#" << std::setw(2 * TAB_WIDTH) << "" << std::left
              << CELL_IDS[i] << " : " << std::left << std::setw(col_2_width)
              << strip_file_path(pf_names[i]) << "#\n";
    }
  }
  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";

  col_2_width = INFO_FILE_COL_WIDTH - WEIGHTS_FILE_LBL.length() - TAB_WIDTH - 4;
  out_buf << "#" << std::setw(TAB_WIDTH) << "" << WEIGHTS_FILE_LBL
          << std::setw(1) << std::right << " : " << std::setw(col_2_width)
          << "#\n";
  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";

  if (!pfpc_weights_file.empty()) {
    col_2_width = INFO_FILE_COL_WIDTH - PFPC_FILE_LBL.length() - TAB_WIDTH - 10;
    out_buf << "#" << std::setw(2 * TAB_WIDTH) << "" << std::left
            << PFPC_FILE_LBL << " : " << std::left << std::setw(col_2_width)
            << strip_file_path(pfpc_weights_file) << "#\n";
  }
  if (!mfnc_weights_file.empty()) {
    col_2_width = INFO_FILE_COL_WIDTH - MFNC_FILE_LBL.length() - TAB_WIDTH - 10;
    out_buf << "#" << std::setw(2 * TAB_WIDTH) << "" << std::left
            << MFNC_FILE_LBL << " : " << std::left << std::setw(col_2_width)
            << strip_file_path(mfnc_weights_file) << "#\n";
  }

  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";
  out_buf << "#" << std::setfill('#') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "\n";
  out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "#\n";
}

void Control::write_sess_info(std::fstream &out_buf) {
  out_buf << "############################## SESSION INFO "
             "##################################\n";
  out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "#\n";

  uint32_t col_2_width =
      INFO_FILE_COL_WIDTH - TRIAL_DEFINE_LBL.length() - TAB_WIDTH - 1;
  out_buf << "#" << std::left << std::setw(1) << "" << TRIAL_DEFINE_LBL
          << std::setw(1) << std::right << " : " << std::setw(col_2_width)
          << "#\n";
  out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "#\n";

  for (auto trial : if_data.s_file.parsed_trial_info.trial_map) {
    col_2_width = INFO_FILE_COL_WIDTH - trial.first.length() - TAB_WIDTH - 4;
    out_buf << "#" << std::setw(TAB_WIDTH) << "" << std::left << trial.first
            << std::right << " : " << std::setw(col_2_width) << "#\n";
    out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
            << "#\n";

    uint32_t max_trial_param_len = get_max_key_len(trial.second);

    for (auto var : trial.second) {
      col_2_width =
          INFO_FILE_COL_WIDTH - max_trial_param_len - 2 * TAB_WIDTH - 6;
      out_buf << "#" << std::setw(2 * TAB_WIDTH) << "" << std::left
              << std::setw(max_trial_param_len) << var.first << std::right
              << " : " << std::left << std::setw(col_2_width) << var.second
              << "#\n";
    }
    out_buf << "#" << std::right << std::setfill(' ')
            << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";
  }

  col_2_width = INFO_FILE_COL_WIDTH - BLOCK_DEFINE_LBL.length() - 5;
  out_buf << "#" << std::setw(1) << "" << BLOCK_DEFINE_LBL << std::setw(1)
          << std::right << " : " << std::setw(col_2_width) << "#\n";
  out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "#\n";

  for (auto block : if_data.s_file.parsed_trial_info.block_map) {
    col_2_width = INFO_FILE_COL_WIDTH - block.first.length() - TAB_WIDTH - 4;
    out_buf << "#" << std::setw(TAB_WIDTH) << "" << std::left << block.first
            << " : " << std::right << std::setw(col_2_width) << "#\n";
    out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
            << "#\n";

    uint32_t max_block_param_len = get_max_first_len(block.second);

    for (auto pair : block.second) {
      col_2_width =
          INFO_FILE_COL_WIDTH - max_block_param_len - 2 * TAB_WIDTH - 6;
      out_buf << "#" << std::setw(2 * TAB_WIDTH) << "" << std::left
              << std::setw(max_block_param_len) << pair.first << std::right
              << " : " << std::left << std::setw(col_2_width) << pair.second
              << "#\n";
    }
    out_buf << "#" << std::right << std::setfill(' ')
            << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";
  }

  col_2_width = INFO_FILE_COL_WIDTH - SESSION_DEFINE_LBL.length() - 5;
  out_buf << "#" << std::setw(1) << "" << SESSION_DEFINE_LBL << std::setw(1)
          << std::right << " : " << std::setw(col_2_width) << "#\n";
  out_buf << "#" << std::setfill(' ') << std::setw(INFO_FILE_COL_WIDTH - 1)
          << "#\n";
  for (auto pair : if_data.s_file.parsed_trial_info.session) {
    col_2_width = INFO_FILE_COL_WIDTH - pair.first.length() - TAB_WIDTH - 6;
    out_buf << "#" << std::setw(TAB_WIDTH) << "" << std::left
            << std::setw(pair.first.length()) << pair.first << std::right
            << " : " << std::left << std::setw(col_2_width) << pair.second
            << "#\n";
  }
  out_buf << "#" << std::right << std::setfill(' ')
          << std::setw(INFO_FILE_COL_WIDTH - 1) << "#\n";
  out_buf << "############################ END SESSION RECORD "
             "##############################\n";
}

void Control::save_info_to_file() {
  std::fstream out_if_data_buf(out_info_name.c_str(), std::ios::out);
  write_header_info(out_if_data_buf);
  write_cmdline_info(out_if_data_buf);
  write_sess_info(out_if_data_buf);
  out_if_data_buf.close();
}

void Control::save_pfpc_weights_to_file() {
  if (pfpc_weights_filenames_created) {
    LOG_DEBUG("Saving granule to purkinje weights to file...");
    if (!simCore) {
      LOG_ERROR("Trying to write uninitialized weights to file.");
      LOG_ERROR("(Hint: Try initializing a sim or loading the weights first.)");
      return;
    }
    const float *pfpc_weights = simCore->getMZoneList()[0]->exportPFPCWeights();
    std::fstream outPFPCFileBuffer(pfpc_weights_file.c_str(),
                                   std::ios::out | std::ios::binary);
    rawBytesRW((char *)pfpc_weights, num_gr * sizeof(float), false,
               outPFPCFileBuffer);
    outPFPCFileBuffer.close();
  }
}

void Control::save_pfpc_weights_at_trial_to_file(uint32_t trial) {
  if (pfpc_weights_filenames_created) {
    LOG_DEBUG("Saving granule to purkinje weights to file...");
    if (!simCore) {
      LOG_ERROR("Trying to write uninitialized weights to file.");
      LOG_ERROR("(Hint: Try initializing a sim or loading the weights first.)");
      return;
    }
    std::string curr_trial_weight_name =
        data_out_path + "/" + get_file_basename(pfpc_weights_file) + "_TRIAL_" +
        std::to_string(trial) + WEIGHTS_EXT[0];
    const float *pfpc_weights = simCore->getMZoneList()[0]->exportPFPCWeights();
    std::fstream outPFPCFileBuffer(curr_trial_weight_name.c_str(),
                                   std::ios::out | std::ios::binary);
    rawBytesRW((char *)pfpc_weights, num_gr * sizeof(float), false,
               outPFPCFileBuffer);
    outPFPCFileBuffer.close();
  }
}

void Control::load_pfpc_weights_from_file(std::string in_pfpc_file) {
  if (!simCore) {
    LOG_ERROR("Trying to read weights to uninitialized simulation.");
    LOG_ERROR("(Hint: Try initializing a sim first.)");
    return;
  }
  std::fstream inPFPCFileBuffer(in_pfpc_file.c_str(),
                                std::ios::in | std::ios::binary);
  simCore->getMZoneList()[0]->load_pfpc_weights_from_file(inPFPCFileBuffer);
  inPFPCFileBuffer.close();
}

void Control::load_pfpc_weight_mask_from_file(std::string weight_mask_file) {
  if (!simCore) {
    LOG_ERROR("Trying to read weights to uninitialized simulation.");
    LOG_ERROR("(Hint: Try initializing a sim first.)");
    return;
  }
  std::fstream inMaskBuffer(weight_mask_file.c_str(),
                            std::ios::in | std::ios::binary);
  simCore->getMZoneList()[0]->load_pfpc_weight_mask_from_file(inMaskBuffer);
  inMaskBuffer.close();
}

void Control::save_mfdcn_weights_to_file() {
  if (mfnc_weights_filenames_created) {
    LOG_DEBUG("Saving mossy fiber to deep nucleus weigths to file...");
    if (!simCore) {
      LOG_ERROR("Trying to write uninitialized weights to file.");
      LOG_ERROR("(Hint: Try initializing a sim or loading the weights first.)");
      return;
    }
    const float *mfdcn_weights =
        simCore->getMZoneList()[0]->exportMFDCNWeights();
    std::fstream outMFDCNFileBuffer(mfnc_weights_file.c_str(),
                                    std::ios::out | std::ios::binary);
    rawBytesRW((char *)mfdcn_weights,
               num_nc * num_p_nc_from_mf_to_nc * sizeof(const float), false,
               outMFDCNFileBuffer);
    outMFDCNFileBuffer.close();
  }
}

void Control::load_mfdcn_weights_from_file(std::string in_mfdcn_file) {
  if (!simCore) {
    LOG_ERROR("Trying to read weights to uninitialized simulation.");
    LOG_ERROR("(Hint: Try initializing a sim first.)");
    return;
  }
  std::fstream inMFDCNFileBuffer(in_mfdcn_file.c_str(),
                                 std::ios::in | std::ios::binary);
  simCore->getMZoneList()[0]->load_mfdcn_weights_from_file(inMFDCNFileBuffer);
  inMFDCNFileBuffer.close();
}

void Control::create_out_sim_filename() {
  if (data_out_dir_created) {
    out_sim_name = data_out_path + "/" + data_out_base_name + SIM_EXT;
    out_sim_filename_created = true;
  }
}

void Control::create_out_info_filename() {
  if (data_out_dir_created) {
    out_info_name = data_out_path + "/" + data_out_base_name + TXT_EXT;
    out_info_filename_created = true;
  }
}

void Control::create_raster_filenames(std::map<std::string, bool> &rast_map) {
  if (data_out_dir_created) {
    for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
      if (rast_map[CELL_IDS[i]] || use_gui) {
        rf_names[i] = data_out_path + "/" + data_out_base_name + RAST_EXT[i];
      }
    }
    raster_filenames_created = true;
  }
}

void Control::create_psth_filenames(std::map<std::string, bool> &psth_map) {
  if (data_out_dir_created) {
    for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
      if (psth_map[CELL_IDS[i]] || use_gui) {
        pf_names[i] = data_out_path + "/" + data_out_base_name + PSTH_EXT[i];
      }
    }
    psth_filenames_created = true;
  }
}

void Control::create_weights_filenames(
    std::map<std::string, bool> &weights_map) {
  if (data_out_dir_created) {
    if (weights_map["PFPC"] || use_gui) {
      pfpc_weights_file =
          data_out_path + "/" + data_out_base_name + WEIGHTS_EXT[0];
      pfpc_weights_filenames_created = true; // only useful so far for gui...
    }
    if (weights_map["MFNC"] || use_gui) {
      mfnc_weights_file =
          data_out_path + "/" + data_out_base_name + WEIGHTS_EXT[1];
      mfnc_weights_filenames_created = true; // only useful so far for gui...
    }
  }
}

void Control::create_con_arrs_filenames(
    std::map<std::string, bool> &conn_arrs_map) {
  if (data_out_dir_created) {
    for (uint32_t i = 0; i < NUM_SYN_CONS; i++) {
      if (conn_arrs_map[SYN_CONS_IDS[i]] || use_gui) {
        pre_con_arrs_names[i] =
            data_out_path + "/" + data_out_base_name + "_PRE" + SYN_CONS_EXT[i];
        post_con_arrs_names[i] = data_out_path + "/" + data_out_base_name +
                                 "_POST" + SYN_CONS_EXT[i];
        LOG_DEBUG("Created filename: %s\n", pre_con_arrs_names[i].c_str());
        LOG_DEBUG("Created filename: %s\n", post_con_arrs_names[i].c_str());
      }
    }
    con_arrs_filenames_created = true;
  }
}

void Control::initialize_rast_cell_nums() {
  rast_cell_nums[MF] = num_mf;
  rast_cell_nums[GR] = num_gr;
  rast_cell_nums[GO] = num_go;
  rast_cell_nums[BC] = num_bc;
  rast_cell_nums[SC] = num_sc;
  rast_cell_nums[PC] = num_pc;
  rast_cell_nums[IO] = num_io;
  rast_cell_nums[NC] = num_nc;
}

void Control::initialize_cell_spikes() {
  if (use_gr_act_from_poiss) {
    cell_spikes[GR] = simCore->getPoissGrs()->getGRAPs();
  } else {
    cell_spikes[MF] = mfs->getAPs();
    cell_spikes[GR] = simCore->getInputNet()->exportAPGR();
    cell_spikes[GO] = simCore->getInputNet()->exportAPGO();
  }
  cell_spikes[BC] = simCore->getMZoneList()[0]->exportAPBC();
  cell_spikes[SC] = simCore->getMZoneList()[0]->exportAPSC();
  cell_spikes[PC] = simCore->getMZoneList()[0]->exportAPPC();
  cell_spikes[IO] = simCore->getMZoneList()[0]->exportAPIO();
  cell_spikes[NC] = simCore->getMZoneList()[0]->exportAPNC();
}

void Control::initialize_spike_sums() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    spike_sums[i].non_cs_spike_sum = 0;
    spike_sums[i].cs_spike_sum = 0;
    spike_sums[i].non_cs_spike_counter =
        (uint32_t *)calloc(rast_cell_nums[i], sizeof(uint32_t));
    spike_sums[i].cs_spike_counter =
        (uint32_t *)calloc(rast_cell_nums[i], sizeof(uint32_t));
  }
  spike_sums_initialized = true;
}

/**
 *  @details The rasters, conceptually, are all 2D arrays of dims
 *  (msMeasure * num_trials, num_cells) EXCEPT for the GR,
 *  which has dims (msMeasure, num_cells) due to sheer number
 *  of GR cells.
 */
void Control::initialize_rasters() {
  uint32_t num_probe_trials = 0;
  for (uint32_t i = 0; i < td.num_trials; i++) {
    if (td.trial_names[i] == "probe_trial")
      num_probe_trials++;
  }
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!rf_names[i].empty() || use_gui) {
      uint32_t row_size =
          (CELL_IDS[i] == "GR") ? msMeasure : msMeasure * num_probe_trials;
      rasters[i] = allocate2DArray<uint8_t>(row_size, rast_cell_nums[i]);
    }
  }

  if (use_gui) {
    pc_vm_raster = allocate2DArray<float>(msMeasure, num_pc);
    nc_vm_raster = allocate2DArray<float>(msMeasure, num_nc);
    io_vm_raster = allocate2DArray<float>(msMeasure, num_io);
  }

  raster_arrays_initialized = true;
}

void Control::initialize_psth_save_funcs() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    psth_save_funcs[i] = [this, i]() {
      if (!pf_names[i].empty()) {
        LOG_DEBUG("Saving %s psth to file...", CELL_IDS[i].c_str());
        write2DArray<uint8_t>(pf_names[i], this->psths[i], this->msMeasure,
                              this->rast_cell_nums[i]);
      }
    };
  }
}

void Control::initialize_raster_save_funcs() {
  uint32_t num_probe_trials = 0;
  for (uint32_t i = 0; i < td.num_trials; i++) {
    if (td.trial_names[i] == "probe_trial")
      num_probe_trials++;
  }
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    rast_save_funcs[i] = [this, i, num_probe_trials]() {
      if (!rf_names[i].empty() && CELL_IDS[i] != "GR") {
        uint32_t row_size = (CELL_IDS[i] == "GR")
                                ? this->msMeasure
                                : this->msMeasure * num_probe_trials;
        LOG_DEBUG("Saving %s raster to file...", CELL_IDS[i].c_str());
        write2DArray<uint8_t>(rf_names[i], this->rasters[i], row_size,
                              this->rast_cell_nums[i]);
      }
    };
  }
}

void Control::initialize_psths() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!pf_names[i].empty() || use_gui)
      // TODO: make data type bigger for psth
      psths[i] = allocate2DArray<uint8_t>(msMeasure, rast_cell_nums[i]);
  }
  psth_arrays_initialized = true;
}

void Control::runSession(struct gui *gui) {
  set_info_file_str_props(BEFORE_RUN, if_data);
  double start, end;
  // int goSpkCounter[num_go];
  if (!use_gui)
    run_state = IN_RUN_NO_PAUSE;
  trial = 0;
  raster_counter = 0;
  // trial loop
  while (trial < td.num_trials && run_state != NOT_IN_RUN) {
    std::string currTrialName = td.trial_names[trial];
    std::string nextTrialName =
        (trial < td.num_trials - 1) ? td.trial_names[trial + 1] : "";
    uint32_t useCS = td.use_css[trial];
    uint32_t onsetCS = pre_collection_ts + td.cs_onsets[trial];
    uint32_t csLength = td.cs_lens[trial];
    // uint32_t percentCS    = td.cs_percents[trial]; // unused for now
    uint32_t useUS = td.use_uss[trial];
    uint32_t onsetUS = pre_collection_ts + td.us_onsets[trial];

    int PSTHCounter = 0;
    // float gGRGO_sum = 0;
    // float gMFGO_sum = 0;

    if (currTrialName == "probe_trial")
      pf_pc_plast = OFF;
    else
      pf_pc_plast = GRADED;

    // memset(goSpkCounter, 0, num_go * sizeof(int));

    LOG_INFO("Trial number: %d", trial + 1);
    start = omp_get_wtime();
    for (uint32_t ts = 0; ts < trialTime; ts++) {
      if (use_gr_act_from_poiss) {
        simCore->calcActivityGRPoiss(pf_pc_plast, ts);
      } else {
        if (useUS == 1 && ts == onsetUS) // deliver the US
        {
          simCore->updateErrDrive(0, 0.3);
        }
        // deliver cs if specified at cmdline and within cs duration
        if (useCS && ts >= onsetCS && ts < onsetCS + csLength) {
          mfAP = mfs->calcPoissActivity(TONIC_CS_A, simCore->getMZoneList());
        } else { // background mf activity
          mfAP = mfs->calcPoissActivity(BKGD, simCore->getMZoneList());
        }

        simCore->updateMFInput(mfAP);
        // this is the main simCore function which computes all cell pops'
        // spikes
        simCore->calcActivity(spillFrac, pf_pc_plast, mf_nc_plast,
                              use_pfpc_weight_mask);
      }
      /* collect conductances used to check tuning */
      // if (ts >= onsetCS && ts < onsetCS + csLength) {
      //   mfgoG = simCore->getInputNet()->exportgSum_MFGO();
      //   grgoG = simCore->getInputNet()->exportgSum_GRGO();
      //   goSpks = simCore->getInputNet()->exportAPGO();

      //  for (int i = 0; i < num_go; i++) {
      //    goSpkCounter[i] += goSpks[i];
      //    gGRGO_sum += grgoG[i];
      //    gMFGO_sum += mfgoG[i];
      //  }
      //}

      ///* upon offset of CS, report averages of above collected conductances */
      // if (ts == onsetCS + csLength) {
      //   countGOSpikes(goSpkCounter);
      //   LOG_DEBUG("Mean gGRGO   = %0.4f", gGRGO_sum / (num_go * csLength));
      //   LOG_DEBUG("Mean gMFGO   = %0.5f", gMFGO_sum / (num_go * csLength));
      //   LOG_DEBUG("GR:MF ratio  = %0.2f", gGRGO_sum / gMFGO_sum);
      // }

      /* data collection */
      // comment out for when collect every time step
      if (currTrialName == "probe_trial") {
        if (ts >= onsetCS - msPreCS && ts < onsetCS + csLength + msPostCS) {
          fill_rasters(raster_counter, PSTHCounter);
          fill_psths(PSTHCounter);
          PSTHCounter++;
          raster_counter++;
        }
      }

      if (use_gui) {
        update_spike_sums(ts, onsetCS, onsetCS + csLength);
        // Update gui main loop if any events are pending.
        if (gtk_events_pending())
          gtk_main_iteration();
      }
    }
    end = omp_get_wtime();
    LOG_INFO("'%s' took %0.2fs", currTrialName.c_str(), end - start);

    if (use_gui) {
      // for now, compute the mean and median firing rates for all cells if win
      // is visible
      if (firing_rates_win_visible(gui)) {
        calculate_firing_rates(onsetCS, onsetCS + csLength);
        gdk_threads_add_idle((GSourceFunc)update_fr_labels, gui);
      }
      if (run_state == IN_RUN_PAUSE) {
        LOG_DEBUG("Simulation is paused at end of trial %d.", trial + 1);
        while (run_state == IN_RUN_PAUSE) {
          if (gtk_events_pending())
            gtk_main_iteration();
        }
        LOG_DEBUG("Continuing...");
      }
      reset_spike_sums();
    }
    //if (data_out_dir_created) {
    //  if (currTrialName != "probe_trial" && nextTrialName == "probe_trial") {
    //    save_pfpc_weights_at_trial_to_file(trial);
    //    std::string weight_steps_ltp_fname =
    //        data_out_path + "/" + data_out_base_name + "_TRIAL_" +
    //        std::to_string(trial) + "_LTP.pfpcpe";
    //    LOG_DEBUG(
    //        "Saving pfpc ltp plasticity events array to file at trial %d...",
    //        trial);
    //    std::fstream out_weight_steps_ltp_buf(weight_steps_ltp_fname.c_str(),
    //                                          std::ios::out | std::ios::binary);
    //    simCore->getMZoneList()[0]->save_weight_steps_ltp_to_file(
    //        out_weight_steps_ltp_buf);
    //    out_weight_steps_ltp_buf.close();
    //
    //    std::string weight_steps_ltd_fname =
    //        data_out_path + "/" + data_out_base_name + "_TRIAL_" +
    //        std::to_string(trial) + "_LTD.pfpcpe";
    //    LOG_DEBUG(
    //        "Saving pfpc ltd plasticity events array to file at trial %d...",
    //        trial);
    //    std::fstream out_weight_steps_ltd_buf(weight_steps_ltd_fname.c_str(),
    //                                          std::ios::out | std::ios::binary);
    //    simCore->getMZoneList()[0]->save_weight_steps_ltd_to_file(
    //        out_weight_steps_ltd_buf);
    //    out_weight_steps_ltd_buf.close();
    //  }
    //  if (currTrialName == "probe_trial" && nextTrialName != "probe_trial") {
    //    simCore->getMZoneList()[0]->reset_weight_steps_ltp();
    //    LOG_DEBUG("Resetting PFPC synapse LTP steps at %d...", trial);
    //    simCore->getMZoneList()[0]->reset_weight_steps_ltd();
    //    LOG_DEBUG("Resetting PFPC synapse LTD steps at %d...", trial);
    //  }
    //}
    trial++;
  }
  if (run_state == NOT_IN_RUN)
    LOG_INFO("Simulation terminated.");
  else if (run_state == IN_RUN_NO_PAUSE)
    LOG_INFO("Simulation Completed.");
  run_state = NOT_IN_RUN;
  set_info_file_str_props(AFTER_RUN, if_data);

  if (!use_gui) { // go ahead and save everything if we're not in the gui.
    save_rasters();
    save_psths();
    // if (data_out_dir_created) {
    //   std::string weight_steps_ltp_fname =
    //       data_out_path + "/" + data_out_base_name + "_TRIAL_" +
    //       std::to_string(trial) + "_LTP.pfpcpe";
    //   LOG_DEBUG(
    //       "Saving pfpc ltp plasticity events array to file at trial %d...",
    //       trial);
    //   std::fstream out_weight_steps_ltp_buf(weight_steps_ltp_fname.c_str(),
    //                                         std::ios::out |
    //                                         std::ios::binary);
    //   simCore->getMZoneList()[0]->save_weight_steps_ltp_to_file(
    //       out_weight_steps_ltp_buf);
    //   out_weight_steps_ltp_buf.close();
    //  std::string weight_steps_ltd_fname =
    //      data_out_path + "/" + data_out_base_name + "_TRIAL_" +
    //      std::to_string(trial) + "_LTD.pfpcpe";
    //  LOG_DEBUG(
    //      "Saving pfpc ltd plasticity events array to file at trial %d...",
    //      trial);
    //  std::fstream out_weight_steps_ltd_buf(weight_steps_ltd_fname.c_str(),
    //                                        std::ios::out | std::ios::binary);
    //  simCore->getMZoneList()[0]->save_weight_steps_ltd_to_file(
    //      out_weight_steps_ltd_buf);
    //  out_weight_steps_ltd_buf.close();
    //}
    save_pfpc_weights_at_trial_to_file(trial);
    save_mfdcn_weights_to_file();
    save_sim_to_file();
    save_info_to_file();
  }
  trial--; // setting so that is valid for drawing go rasters after a sim
}

void Control::reset_spike_sums() {
  for (int i = 0; i < NUM_CELL_TYPES; i++) {
    spike_sums[i].cs_spike_sum = 0;
    spike_sums[i].non_cs_spike_sum = 0;
    memset(spike_sums[i].cs_spike_counter, 0,
           rast_cell_nums[i] * sizeof(uint32_t));
    memset(spike_sums[i].non_cs_spike_counter, 0,
           rast_cell_nums[i] * sizeof(uint32_t));
  }
}

void Control::reset_rasters() {
  uint32_t num_probe_trials = 0;
  for (uint32_t i = 0; i < td.num_trials; i++) {
    if (td.trial_names[i] == "probe_trial")
      num_probe_trials++;
  }
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!rf_names[i].empty() || use_gui) {
      uint32_t row_size =
          (CELL_IDS[i] == "GR") ? msMeasure : msMeasure * num_probe_trials;
      memset(rasters[i][0], '\000',
             row_size * rast_cell_nums[i] * sizeof(uint8_t));
    }
  }
}

void Control::reset_psths() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!pf_names[i].empty()) {
      memset(psths[i][0], '\000',
             rast_cell_nums[i] * msMeasure * sizeof(uint8_t));
    }
  }
}

void Control::save_gr_rasters_at_trial_to_file(uint32_t trial) {
  if (!rf_names[GR].empty()) {
    std::string trial_raster_name = data_out_path + "/" +
                                    get_file_basename(rf_names[GR]) +
                                    "_TRIAL_" + std::to_string(trial) + BIN_EXT;
    LOG_DEBUG("Saving granule raster to file...");
    write2DArray<uint8_t>(trial_raster_name, rasters[GR], num_gr, msMeasure);
  }
}

void Control::save_rasters() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!rf_names[i].empty())
      rast_save_funcs[i]();
  }
}

void Control::save_psths() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!pf_names[i].empty())
      psth_save_funcs[i]();
  }
}

void Control::save_con_arrs() {
  if (con_arrs_filenames_created) {
    for (uint32_t i = 0; i < NUM_SYN_CONS; i++) {
      if (!pre_con_arrs_names[i].empty() && !post_con_arrs_names[i].empty()) {
        LOG_DEBUG("Saving %s connectivity array(s) to file...",
                  SYN_CONS_IDS[i].c_str());
        // annoyingly, the only synapse with no pre-synaptic con arr
        std::fstream pre_con_arrs_file_buf;
        if (SYN_CONS_IDS[i] != "MFNC") {
          pre_con_arrs_file_buf.open(pre_con_arrs_names[i].c_str(),
                                     std::ios::out | std::ios::binary);
        }
        std::fstream post_con_arrs_file_buf(post_con_arrs_names[i].c_str(),
                                            std::ios::out | std::ios::binary);
        if (SYN_CONS_IDS[i] == "MFGR") {
          simState->getInnetConStateInternal()->pMFfromMFtoGRRW(
              pre_con_arrs_file_buf, false);
          simState->getInnetConStateInternal()->pGRfromMFtoGRRW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "GRGO") {
          simState->getInnetConStateInternal()->pGRfromGRtoGORW(
              pre_con_arrs_file_buf, false);
          simState->getInnetConStateInternal()->pGOfromGRtoGORW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "MFGO") {
          simState->getInnetConStateInternal()->pMFfromMFtoGORW(
              pre_con_arrs_file_buf, false);
          simState->getInnetConStateInternal()->pGOfromMFtoGORW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "GOGO") {
          simState->getInnetConStateInternal()->pGOOutfromGOtoGORW(
              pre_con_arrs_file_buf, false);
          simState->getInnetConStateInternal()->pGOInfromGOtoGORW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "GOGR") {
          simState->getInnetConStateInternal()->pGOfromGOtoGRRW(
              pre_con_arrs_file_buf, false);
          simState->getInnetConStateInternal()->pGRfromGOtoGRRW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "BCPC") {
          simState->getMZoneConStateInternal(0)->pBCfromBCtoPCRW(
              pre_con_arrs_file_buf, false);
          simState->getMZoneConStateInternal(0)->pPCfromBCtoPCRW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "SCPC") {
          simState->getMZoneConStateInternal(0)->pSCfromSCtoPCRW(
              pre_con_arrs_file_buf, false);
          simState->getMZoneConStateInternal(0)->pPCfromSCtoPCRW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "PCBC") {
          simState->getMZoneConStateInternal(0)->pPCfromPCtoBCRW(
              pre_con_arrs_file_buf, false);
          simState->getMZoneConStateInternal(0)->pBCfromPCtoBCRW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "PCNC") {
          simState->getMZoneConStateInternal(0)->pPCfromPCtoNCRW(
              pre_con_arrs_file_buf, false);
          simState->getMZoneConStateInternal(0)->pNCfromPCtoNCRW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "IOIO") {
          simState->getMZoneConStateInternal(0)->pIOOutIOIORW(
              pre_con_arrs_file_buf, false);
          simState->getMZoneConStateInternal(0)->pIOInIOIORW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "NCIO") {
          simState->getMZoneConStateInternal(0)->pNCfromNCtoIORW(
              pre_con_arrs_file_buf, false);
          simState->getMZoneConStateInternal(0)->pIOfromNCtoIORW(
              post_con_arrs_file_buf, false);
        } else if (SYN_CONS_IDS[i] == "MFNC") {
          // just doesn't exist ig
          // simState->getMZoneConStateInternal(0)->pMFfromMFtoNCRW(
          //    pre_con_arrs_file_buf, false);
          simState->getMZoneConStateInternal(0)->pNCfromMFtoNCRW(
              post_con_arrs_file_buf, false);
        }
        if (SYN_CONS_IDS[i] != "MFNC") {
          pre_con_arrs_file_buf.close();
        }
        post_con_arrs_file_buf.close();
      }
    }
  }
}

void Control::update_spike_sums(int tts, float onset_cs, float offset_cs) {
  // update cs spikes
  if (tts >= onset_cs && tts < offset_cs) {
    for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
      if (cell_spikes[i]) {
        for (uint32_t j = 0; j < rast_cell_nums[i]; j++) {
          spike_sums[i].cs_spike_sum += cell_spikes[i][j];
          spike_sums[i].cs_spike_counter[j] += cell_spikes[i][j];
        }
      }
    }
  }
  // update non-cs spikes
  else if (tts < onset_cs) {
    for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
      if (cell_spikes[i]) {
        for (uint32_t j = 0; j < rast_cell_nums[i]; j++) {
          spike_sums[i].non_cs_spike_sum += cell_spikes[i][j];
          spike_sums[i].non_cs_spike_counter[j] += cell_spikes[i][j];
        }
      }
    }
  }
}

void Control::calculate_firing_rates(float onset_cs, float offset_cs) {
  float non_cs_time_secs =
      (onset_cs - 1) / 1000.0; // why only pre-cs? (Ask Joe)
  float cs_time_secs = (offset_cs - onset_cs) / 1000.0;

  for (int i = 0; i < NUM_CELL_TYPES; i++) {
    // sort sums for medians
    std::sort(spike_sums[i].cs_spike_counter,
              spike_sums[i].cs_spike_counter + rast_cell_nums[i]);
    std::sort(spike_sums[i].non_cs_spike_counter,
              spike_sums[i].non_cs_spike_counter + rast_cell_nums[i]);

    // calculate medians
    firing_rates[i].non_cs_median_fr =
        (spike_sums[i].non_cs_spike_counter[rast_cell_nums[i] / 2 - 1] +
         spike_sums[i].non_cs_spike_counter[rast_cell_nums[i] / 2]) /
        (2.0 * non_cs_time_secs);
    firing_rates[i].cs_median_fr =
        (spike_sums[i].cs_spike_counter[rast_cell_nums[i] / 2 - 1] +
         spike_sums[i].cs_spike_counter[rast_cell_nums[i] / 2]) /
        (2.0 * cs_time_secs);

    // calculate means
    firing_rates[i].non_cs_mean_fr =
        spike_sums[i].non_cs_spike_sum / (non_cs_time_secs * rast_cell_nums[i]);
    firing_rates[i].cs_mean_fr =
        spike_sums[i].cs_spike_sum / (cs_time_secs * rast_cell_nums[i]);
  }
}

void Control::countGOSpikes(int *goSpkCounter) {
  float isi = (td.us_onsets[0] - td.cs_onsets[0]) / 1000.0;
  std::sort(goSpkCounter, goSpkCounter + num_go);

  float m = (goSpkCounter[num_go / 2 - 1] + goSpkCounter[num_go / 2]) / 2.0;
  float goSpkSum = 0;

  for (int i = 0; i < num_go; i++)
    goSpkSum += goSpkCounter[i];

  LOG_DEBUG("Mean GO Rate: %0.2f", goSpkSum / ((float)num_go * isi));
  LOG_DEBUG("Median GO Rate: %0.1f", m / isi);
}

void Control::fill_rasters(uint32_t raster_counter, uint32_t psth_counter) {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (cell_spikes[i]) {
      uint32_t temp_counter = raster_counter;
      if (!rf_names[i].empty() || use_gui) {
        /* GR spikes are only spikes not saved on host every time step:
         * InNet::exportAPGR makes cudaMemcpy call before returning pointer to
         * mem address */
        if (CELL_IDS[i] == "GR") {
          if (use_gr_act_from_poiss) {
            cell_spikes[i] = simCore->getPoissGrs()->getGRAPs();
          } else {
            cell_spikes[i] = simCore->getInputNet()->exportAPGR();
          }
          temp_counter = psth_counter;
        }
        for (uint32_t j = 0; j < rast_cell_nums[i]; j++) {
          rasters[i][temp_counter][j] = cell_spikes[i][j];
        }
      }
    }
  }

  if (use_gui) { // update the voltage rasters for the PC Window
    const float *vm_pc = simCore->getMZoneList()[0]->exportVmPC();
    for (int i = 0; i < num_pc; i++) {
      pc_vm_raster[psth_counter][i] = vm_pc[i];
    }
    const float *vm_io = simCore->getMZoneList()[0]->exportVmIO();
    for (int i = 0; i < num_io; i++) {
      io_vm_raster[psth_counter][i] = vm_io[i];
    }
    const float *vm_nc = simCore->getMZoneList()[0]->exportVmNC();
    for (int i = 0; i < num_nc; i++) {
      nc_vm_raster[psth_counter][i] = vm_nc[i];
    }
  }
}

void Control::fill_psths(uint32_t psth_counter) {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (cell_spikes[i]) {
      if (!pf_names[i].empty() || use_gui) {
        /* GR spikes are only spikes not saved on host every time step:
         * InNet::exportAPGR makes cudaMemcpy call before returning pointer to
         * mem address */
        if (CELL_IDS[i] == "GR") {
          if (use_gr_act_from_poiss) {
            cell_spikes[i] = simCore->getPoissGrs()->getGRAPs();
          } else {
            cell_spikes[i] = simCore->getInputNet()->exportAPGR();
          }
          for (uint32_t j = 0; j < rast_cell_nums[i]; j++) {
            psths[i][psth_counter][j] += cell_spikes[i][j];
          }
        }
      }
    }
  }
}

void Control::delete_spike_sums() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    free(spike_sums[i].non_cs_spike_counter);
    free(spike_sums[i].cs_spike_counter);
  }
}

void Control::delete_rasters() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!rf_names[i].empty() || use_gui)
      delete2DArray<uint8_t>(rasters[i]);
  }
  if (use_gui) {
    delete2DArray<float>(pc_vm_raster);
    delete2DArray<float>(nc_vm_raster);
    delete2DArray<float>(io_vm_raster);
  }
}

void Control::delete_psths() {
  for (uint32_t i = 0; i < NUM_CELL_TYPES; i++) {
    if (!pf_names[i].empty())
      delete2DArray<uint8_t>(psths[i]);
  }
}
