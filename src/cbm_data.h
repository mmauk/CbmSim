//#ifndef _CBM_DATA_H
//#define _CBM_DATA_H
//
//#include <cstdint>
//#include <functional>
//
//#include "commandline.h"
//
//class CbmData
//{
//	CbmData();
//	CbmData(parsed_commandline &p_cl);
//
//	uint8_t **rasters[NUM_CELL_TYPES];
//	uint8_t **psths[NUM_CELL_TYPES];
//
//	float **pc_vm_raster;
//	float **nc_vm_raster;
//	float **io_vm_raster;
//
//	std::string rf_names[NUM_CELL_TYPES];
//	std::string pf_names[NUM_CELL_TYPES]; 
//
//	std::string pf_pc_weights_file = "";
//	std::string mf_nc_weights_file = "";
//
//	struct cell_spike_sums spike_sums[NUM_CELL_TYPES];
//	struct cell_firing_rates firing_rates[NUM_CELL_TYPES];
//
//	uint32_t rast_sizes[NUM_CELL_TYPES]; 
//
//	std::function<void(std::string)> rast_save_funcs[NUM_CELL_TYPES];
//	std::function<void(std::string)> psth_save_funcs[NUM_CELL_TYPES];
//
//	void save_sim_to_file(std::string outSimFile);
//	void save_pfpc_weights_to_file(std::string out_pfpc_file);
//	void load_pfpc_weights_from_file(std::string in_pfpc_file);
//	void save_mfdcn_weights_to_file(std::string out_mfdcn_file);
//	void load_mfdcn_weights_from_file(std::string in_mfdcn_file);
//	
//	void create_raster_filenames(parsed_commandline &p_cl);
//	void create_psth_filenames(parsed_commandline &p_cl);
//	void create_weights_filenames(parsed_commandline &p_cl);
//	
//	void initialize_rast_cell_nums();
//	
//	void initialize_cell_spikes();
//	void initialize_spike_sums();
//	
//	void initialize_rasters(); 
//	void initialize_psths();
//	
//	void initialize_psth_save_funcs();
//	void initialize_raster_save_funcs();
//	
//	void countGOSpikes(int *goSpkCounter);
//	void update_spike_sums(int tts, float onset_cs, float offset_cs);
//	void calculate_firing_rates(float onset_cs, float offset_cs);
//	void fill_rasters(uint32_t raster_counter, uint32_t psth_counter);
//	void fill_psths(uint32_t psth_counter);
//	void save_rasters();
//	void save_psths();
//	
//	void delete_rasters();
//	void delete_psths(); 
//	void delete_spike_sums();
//};
//
//#endif /*_CBM_DATA_H*/

