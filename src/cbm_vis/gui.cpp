#include <sys/stat.h>

#include "logger.h"
#include "array_util.h"
#include "connectivityparams.h"
#include "commandline.h" // for OUTPUT_DATA_PATH def
#include "gui.h"

std::map<std::string, bool> init_all_rast_or_psth_map {
	{"MF", true}, 
	{"GR", true},
	{"GO", true},
	{"BC", true},
	{"SC", true},
	{"PC", true},
	{"IO", true},
	{"NC", true},
};

std::map<std::string, bool> init_all_weights_map = {
	{ "PFPC", true },
	{ "MFNC", true },
};

// temp function so gtk doesn't whine abt NULL callbacks
static void null_callback(GtkWidget *widget, gpointer data) {}

static bool assert(bool expr, const char *error_string, const char *func = "assert")
{
	if (!expr)
	{
		LOG_FATAL("%s(): %s", func, error_string);
		return false;
	}
	return true;
}

static void load_file(GtkWidget *widget, Control *control, void (Control::*on_file_load_func)(std::string),
	std::string err_msg)
{
	std::string in_file_std_str = "";
	GtkWidget *dialog = gtk_file_chooser_dialog_new(
		  "Open File",
		  NULL, /* no parent window is fine for now */
		  GTK_FILE_CHOOSER_ACTION_OPEN,
		  "Cancel",
		  GTK_RESPONSE_CANCEL,
		  "Open",
		  GTK_RESPONSE_ACCEPT,
		  NULL
	);
	gint response = gtk_dialog_run(GTK_DIALOG(dialog));
	if (response == GTK_RESPONSE_ACCEPT)
	{
		GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
		char *in_file_c_str = gtk_file_chooser_get_filename(chooser);
		// TODO: pop-up warning for invalid file
		in_file_std_str = std::string(in_file_c_str);
		g_free(in_file_c_str);
	}
	else
	{
		LOG_ERROR("%s", err_msg.c_str());
		gtk_widget_destroy(dialog);
		return;
	}
	gtk_widget_destroy(dialog);
	(control->*on_file_load_func)(in_file_std_str);
}

static void save_file(GtkWidget *widget, Control *control, void (Control::*on_file_save_func)(std::string),
	std::string err_msg, const char *default_out_file)
{
	std::string out_file_std_str = "";
	GtkWidget *dialog = gtk_file_chooser_dialog_new
		(
		  "Save File",
		  NULL, /* no parent window is fine for now */
		  GTK_FILE_CHOOSER_ACTION_SAVE,
		  "Cancel",
		  GTK_RESPONSE_CANCEL,
		  "Save",
		  GTK_RESPONSE_ACCEPT,
		  NULL
		);

	GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
	gtk_file_chooser_set_do_overwrite_confirmation(chooser, TRUE); /* huh? */
	gtk_file_chooser_set_current_name(chooser, default_out_file);
	gint response = gtk_dialog_run(GTK_DIALOG(dialog));

	if (response == GTK_RESPONSE_ACCEPT)
	{
		char *out_file_c_str = gtk_file_chooser_get_filename(chooser);
		out_file_std_str = std::string(out_file_c_str);
		g_free(out_file_c_str);
	}
	else
	{
		LOG_ERROR("%s", err_msg.c_str());
		gtk_widget_destroy(dialog);
		return;
	}
	gtk_widget_destroy(dialog);
	(control->*on_file_save_func)(out_file_std_str);
}

// redundant overload..eck
static void save_file(GtkWidget *widget, Control *control, std::function<void(std::string)> *save_func,
	std::string err_msg, const char *default_out_file)
{
	std::string out_file_std_str = "";
	GtkWidget *dialog = gtk_file_chooser_dialog_new
		(
		  "Save File",
		  NULL, /* no parent window is fine for now */
		  GTK_FILE_CHOOSER_ACTION_SAVE,
		  "Cancel",
		  GTK_RESPONSE_CANCEL,
		  "Save",
		  GTK_RESPONSE_ACCEPT,
		  NULL
		);

	GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
	gtk_file_chooser_set_do_overwrite_confirmation(chooser, TRUE); /* huh? */
	gtk_file_chooser_set_current_name(chooser, default_out_file);
	gint response = gtk_dialog_run(GTK_DIALOG(dialog));

	if (response == GTK_RESPONSE_ACCEPT)
	{
		char *out_file_c_str = gtk_file_chooser_get_filename(chooser);
		out_file_std_str = std::string(out_file_c_str);
		g_free(out_file_c_str);
	}
	else
	{
		LOG_ERROR("%s", err_msg.c_str());
		gtk_widget_destroy(dialog);
		return;
	}
	gtk_widget_destroy(dialog);
	(*save_func)(out_file_std_str);
}

static void on_load_session_file(GtkWidget *widget, Control *control)
{
	load_file(widget, control, &Control::initialize_session, "[ERROR]: Could not open session file.");
}

static void on_load_sim_file(GtkWidget *widget, Control *control)
{
	// terribly stupid design atm. act params initialized when session is initialized,
	// so user needs to load session file before initializing simulation. FIX IT
	if (!control->trials_data_initialized)
	{
		std::cout << "[ERROR]: Due to shitty design, you need to load a session file before loading in a simulation file...\n";
		return;
	}
	load_file(widget, control, &Control::init_sim, "[ERROR]: Could not open simulation file.");
}

// FIXME: initialize filenames from output_sim_name, coming from either:
// a) commandline
static void on_save_file(GtkWidget *widget, save_data *data)
{
	if (data->opt == data->opt % (2 * NUM_CELL_TYPES))
	{
		if (data->opt == data->opt % NUM_CELL_TYPES)
		{
			if (!data->ctrl_ptr->raster_filenames_created)
			{
				data->ctrl_ptr->create_raster_filenames(init_all_rast_or_psth_map);
			}
			data->ctrl_ptr->rast_save_funcs[data->opt]();
		}
		else
		{
			if (!data->ctrl_ptr->psth_filenames_created)
			{
				data->ctrl_ptr->create_psth_filenames(init_all_rast_or_psth_map);
			}
			// FIXME: place all psth and rast save funcs in a general save func so that
			// we don't have to do this hacky modding out just to get the correct index
			data->ctrl_ptr->psth_save_funcs[data->opt % NUM_CELL_TYPES]();
		}
	}
	else if (data->opt == PFPC)
	{
		if (!data->ctrl_ptr->pfpc_weights_filenames_created)
		{
			data->ctrl_ptr->create_weights_filenames(init_all_weights_map);
		}
		data->ctrl_ptr->save_pfpc_weights_to_file();
	}
	else if (data->opt == MFNC)
	{
		if (!data->ctrl_ptr->mfnc_weights_filenames_created)
		{
			data->ctrl_ptr->create_weights_filenames(init_all_weights_map);
		}
		data->ctrl_ptr->save_mfdcn_weights_to_file();
	}
	else
	{
		if (!data->ctrl_ptr->out_sim_filename_created)
		{
			data->ctrl_ptr->create_out_sim_filename();
		}
		data->ctrl_ptr->save_sim_to_file();
	}
}

static void on_load_pfpc_weights(GtkWidget *widget, Control *control)
{
	load_file(widget, control, &Control::load_pfpc_weights_from_file, "[ERROR]: Could not open pf-pc weights file.");
}

static void on_load_mfdcn_weights(GtkWidget *widget, Control *control)
{
	load_file(widget, control, &Control::load_mfdcn_weights_from_file, "[ERROR]: Could not open mf-dcn weights file.");
}

static bool is_valid_dir_name(const char *in_str)
{
	int len = snprintf(NULL, 0, "%s", in_str);
	if (len == 0 || len > 255) return false; // str length must be in the range (0,255]
	if (len == 1 && in_str[0] == '/') return false; // str cannot be just the path separator character
	return true;
}

static bool output_dir_exists(const char *in_str)
{
	int len = snprintf(NULL, 0, "%s", in_str);
	char *full_path = (char *)malloc(OUTPUT_DATA_PATH.length() + len + 1);
	strcpy(full_path, OUTPUT_DATA_PATH.c_str());
	strcat(full_path, in_str);
	struct stat sb;
	bool return_val = (stat(full_path, &sb) == 0) ? true : false;
	free(full_path);
	return return_val;
}

// performs a concatenation of the two given strings, checks whether the
// path separator token exists after the base path
// caller owns the data that this function returns
char *create_dir_name_from(const char *base_path, const char *base_name)
{
	int base_path_len = snprintf(NULL, 0, "%s", base_path);
	int base_name_len = snprintf(NULL, 0, "%s", base_name);
	char *full_path = (char *)malloc(base_path_len + base_name_len + 1 + 1);
	strcpy(full_path, base_path);
	if (base_path[base_path_len-1] != '/')  // add in extra path separator token if not present in base_path
		strcat(full_path, "/");
	strcat(full_path, base_name);
	return full_path;
}

// create directory from a base path concatenated with base_name
// assume user has validated the name of the directory to be made,
// and that the full path (base_path + base_name) does not already exist
static int create_dir_from(const char *base_path, const char *base_name, bool overwrite = false)
{
	char *full_path = create_dir_name_from(base_path, base_name);
	int status = 0;
	if (overwrite)
	{
		LOG_INFO("Deleting existing directory '%s'...", full_path);
		int full_path_len = snprintf(NULL, 0, "%s", full_path);
		char *command = (char *)malloc(7 + full_path_len + 1);
		sprintf(command, "rm -rf %s", full_path);
		status += system(command);
		free(command);
	}
	status += mkdir(full_path, 0775);
	free(full_path);
	return status;
}

static void create_dir_seq_on_valid_name(GtkWidget *parent_win, const char *dir_base_name, bool overwrite = false)
{
	LOG_INFO("Making directory...");
	GtkDialogFlags flags = GtkDialogFlags(GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT);
	char *status_str;
	int status_str_len;
	if (create_dir_from(OUTPUT_DATA_PATH.c_str(), dir_base_name, overwrite) != 0)
	{
		status_str_len = snprintf(NULL, 0, "Could not create directory, '%s'", dir_base_name);
		status_str = (char *)malloc(status_str_len + 1);
		sprintf(status_str, "Could not create directory '%s'", dir_base_name);
		LOG_ERROR("Could not create directory, '%s'", dir_base_name);
	}
	else
	{
		status_str_len = snprintf(NULL, 0, "Directory successfully created!");
		status_str = (char *)malloc(status_str_len + 1);
		sprintf(status_str, "Directory successfully created!");
		LOG_INFO("Directory successfully created!");
	}
	GtkWidget *msg_dialog = gtk_message_dialog_new(GTK_WINDOW(parent_win),
															  flags,
															  GTK_MESSAGE_INFO,
															  GTK_BUTTONS_NONE,
															  status_str);
	gtk_widget_show_all(msg_dialog);
	gtk_dialog_run(GTK_DIALOG(msg_dialog));
	gtk_widget_destroy(msg_dialog);
	free(status_str);
}

static void on_create_dir(GtkWidget *widget, struct gui *gui)
{
	GtkDialogFlags flags = GtkDialogFlags(GTK_DIALOG_MODAL | GTK_DIALOG_DESTROY_WITH_PARENT);
	GtkWidget *dialog = gtk_dialog_new_with_buttons("Choose an Output Directory",
													 GTK_WINDOW(gui->window),
													 flags,
													 "Cancel",
													 GTK_RESPONSE_CANCEL,
													 "OK",
													 GTK_RESPONSE_OK,
													 NULL);
	GtkWidget *entry_label = gtk_label_new("Enter a directory name");
	GtkWidget *entry = gtk_entry_new();
	GtkWidget *content_area = gtk_dialog_get_content_area(GTK_DIALOG(dialog));
	
	gtk_container_add(GTK_CONTAINER(content_area), entry_label);
	gtk_container_add(GTK_CONTAINER(content_area), entry);
	gtk_widget_show_all(dialog);
	bool open = true;
	while (open)
	{
		int result = gtk_dialog_run(GTK_DIALOG(dialog));
		switch (result)
		{
			case GTK_RESPONSE_OK:
				if (!is_valid_dir_name(gtk_entry_get_text(GTK_ENTRY(entry))))
				{
					LOG_WARN("Directory name entered was invalid!");
					GtkWidget *msg_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
																   flags,
																   GTK_MESSAGE_INFO,
																   GTK_BUTTONS_NONE,
																   "Directory with name '%s' cannot be made.",
																   gtk_entry_get_text(GTK_ENTRY(entry)));
					gtk_widget_show_all(msg_dialog);
					gtk_dialog_run(GTK_DIALOG(msg_dialog));
					gtk_widget_destroy(msg_dialog);
					gtk_widget_grab_focus(entry);
				}
				else if (output_dir_exists(gtk_entry_get_text(GTK_ENTRY(entry))))
				{
					LOG_WARN("file already exists.");
					GtkWidget *msg_dialog = gtk_message_dialog_new(GTK_WINDOW(dialog),
																   flags,
																   GTK_MESSAGE_INFO,
																   GTK_BUTTONS_YES_NO,
																   "This directory already exists: would you like to overwrite it?\n"
																   "(Doing so will erase the current directory and all of its content.)");
					gtk_widget_show_all(msg_dialog);
					int overwrite = gtk_dialog_run(GTK_DIALOG(msg_dialog));
					switch (overwrite)
					{
						case GTK_RESPONSE_YES:
							// FIXME: reduce redundancy between this branch and the following else branch
							create_dir_seq_on_valid_name(dialog, gtk_entry_get_text(GTK_ENTRY(entry)), true);
							if (gui->ctrl_ptr != NULL)
							{
								char *full_path = create_dir_name_from(OUTPUT_DATA_PATH.c_str(), gtk_entry_get_text(GTK_ENTRY(entry))); 
								gui->ctrl_ptr->data_out_path = std::string(full_path); // data is safely copied to data_out_path
								gui->ctrl_ptr->data_out_base_name = std::string(gtk_entry_get_text(GTK_ENTRY(entry)));
								gui->ctrl_ptr->data_out_dir_created = true;
								free(full_path); // free current allocated memory
							}
							open = false;
							break;
						case GTK_RESPONSE_NO:
							gtk_widget_grab_focus(entry);
							break;
					}
					gtk_widget_destroy(msg_dialog);
				}
				else
				{
					create_dir_seq_on_valid_name(dialog, gtk_entry_get_text(GTK_ENTRY(entry)));
					if (gui->ctrl_ptr != NULL)
					{
						char *full_path = create_dir_name_from(OUTPUT_DATA_PATH.c_str(), gtk_entry_get_text(GTK_ENTRY(entry))); 
						gui->ctrl_ptr->data_out_path = std::string(full_path); // data is safely copied to data_out_path
						gui->ctrl_ptr->data_out_base_name = std::string(gtk_entry_get_text(GTK_ENTRY(entry)));
						gui->ctrl_ptr->data_out_dir_created = true;
						free(full_path); // free current allocated memory
					}
					open = false;
					break;
				}
				break;
			default:
				open = false;
				break;
		}
	}
	gtk_widget_destroy(dialog);
} 

//FIXME: below is a stop-gap solution. should be more careful with destroy callback
gboolean firing_rates_win_visible(struct gui *gui)
{
	return gui->frw.window != NULL
	   && GTK_IS_WIDGET(gui->frw.window)
	   && gtk_widget_get_realized(gui->frw.window)
	   && gtk_widget_get_visible(gui->frw.window);
}

gboolean update_fr_labels(struct gui *gui)
{
	const struct cell_firing_rates *fr = gui->ctrl_ptr->firing_rates;
	int cell_index = 0;
	FOREACH(gui->frw.cell_labels, clp)
	{
		int fr_index = 0;
		FOREACH(*clp, rp)
		{
			if (fr_index > 0)
			{
				int len = snprintf(NULL, 0, "%.2f", ((float *)(&fr[cell_index]))[fr_index-1]);
				char *result = (char *)malloc(len+1);
				snprintf(result, len+1, "%.2f", ((float *)(&fr[cell_index]))[fr_index-1]);
				gtk_label_set_text(GTK_LABEL(rp->label), result);
				free(result);
			}
			fr_index++;
		}
		cell_index++;
	}
	return false;
}

static void on_firing_rates_window(GtkWidget *widget, struct gui *gui)
{
	gui->frw.window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	gui->frw.grid = gtk_grid_new();

	// set window props
	gtk_window_set_title(GTK_WINDOW(gui->frw.window), "Firing Rates");
	gtk_window_set_default_size(GTK_WINDOW(gui->frw.window),
								DEFAULT_FIRING_RATE_WINDOW_WIDTH,
								DEFAULT_FIRING_RATE_WINDOW_HEIGHT);
	gtk_window_set_position(GTK_WINDOW(gui->frw.window), GTK_WIN_POS_CENTER);
	gtk_window_set_resizable(GTK_WINDOW(gui->frw.window), TRUE);
	gtk_container_set_border_width(GTK_CONTAINER(gui->frw.window), 5);

	//gtk_widget_add_events(gui->frw.window, GDK_DELETE);
	//g_signal_connect(gui->frw.window, "destroy", G_CALLBACK(gtk_widget_destroy), NULL);

	//set grid props
	gtk_grid_set_column_spacing(GTK_GRID(gui->frw.grid), 3);
	gtk_grid_set_row_spacing(GTK_GRID(gui->frw.grid), 3);

	// header label props
	FOREACH(gui->frw.headers, hp)
	{
		hp->label = gtk_label_new(hp->string);
		gtk_widget_set_hexpand(hp->label, true);
		gtk_widget_set_vexpand(hp->label, true);
		gtk_grid_attach(GTK_GRID(gui->frw.grid),
						hp->label,
						hp->col,
						hp->row, 1, 1);
	}

	// row entry props
	FOREACH(gui->frw.cell_labels, clp)
	{
		FOREACH(*clp, rp)
		{
			rp->label = gtk_label_new(rp->string);
			gtk_widget_set_hexpand(rp->label, true);
			gtk_widget_set_vexpand(rp->label, true);
			gtk_grid_attach(GTK_GRID(gui->frw.grid),
							rp->label,
							rp->col,
							rp->row, 1, 1);
		}
	}
	gtk_container_add(GTK_CONTAINER(gui->frw.window), gui->frw.grid);
	gtk_widget_show_all(gui->frw.window);
}

static void on_update_weight(GtkWidget *spin_button, float *weight)
{
	*weight = gtk_spin_button_get_value(GTK_SPIN_BUTTON(spin_button));
}

static void on_tuning_window(GtkWidget *widget, struct gui *gui)
{
	struct tuning_window tw = {
		.window = gtk_window_new(GTK_WINDOW_TOPLEVEL),
		.grid = gtk_grid_new(),
		.tuning_buttons = {
			{
				gtk_adjustment_new(gIncDirectMFtoGR, 0.0, 1.0, 0.0001, 0.1, 0.0),
				NULL, 1, 0, 4,
				{
					NULL, "MF-GR", 0, 0
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncDirectMFtoGR,
					false
				}
			},
			{
				gtk_adjustment_new(mfgoW, 0.0, 1.0, 0.0001, 0.1, 0.0),
				NULL, 1, 1, 4, 
				{
					NULL, "MF-GO", 0, 1
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&mfgoW,
					false
				}
			},
			{
				gtk_adjustment_new(grgoW, 0.0, 1.0, 0.0001, 0.1, 0.0),
				NULL, 1, 2, 4,
				{
					NULL, "GR-GO", 0, 2
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&grgoW,
					false
				}
			},
			{
				gtk_adjustment_new(gogrW, 0.0, 1.0, 0.01, 0.1, 0.0),
				NULL, 1, 3, 2, 
				{
					NULL, "GO-GR", 0, 3
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gogrW,
					false
				}
			},
			{
				gtk_adjustment_new(gGABAIncGOtoGO, 0.0, 1.0, 0.01, 0.1, 0.0),
				NULL, 3, 0, 2,
				{
					NULL, "GO-GO", 2, 0
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gGABAIncGOtoGO,
					false
				}
			},
			{
				gtk_adjustment_new(gIncGRtoPC, 0.0, 1.0, 0.000001, 0.1, 0.0),
				NULL, 3, 1, 6,
				{
					NULL, "GR-PC", 2, 1
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncGRtoPC,
					false
				}
			},
			{
				gtk_adjustment_new(gIncGRtoSC, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 3, 2, 3,
				{
					NULL, "GR-SC", 2, 2
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncGRtoSC,
					false
				}
			},
			{
				gtk_adjustment_new(gIncGRtoBC, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 3, 3, 3,
				{
					NULL, "GR-BC", 2, 3
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncGRtoBC,
					false
				}
			},
			{
				gtk_adjustment_new(gIncSCtoPC, 0.0, 1.0, 0.00001, 0.1, 0.0),
				NULL, 5, 0, 5,
				{
					NULL, "SC-PC", 4, 0
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncSCtoPC,
					false
				}
			},
			{
				gtk_adjustment_new(gIncBCtoPC, 0.0, 1.0, 0.00001, 0.1, 0.0),
				NULL, 5, 1, 5,
				{
					NULL, "BC-PC", 4, 1
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncBCtoPC,
					false
				}
			},
			{
				gtk_adjustment_new(gIncPCtoBC, 0.0, 1.0, 0.01, 0.1, 0.0),
				NULL, 5, 2, 2,
				{
					NULL, "PC-BC", 4, 2
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncPCtoBC,
					false
				}
			},
			{
				gtk_adjustment_new(gIncAvgPCtoNC, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 5, 3, 3,
				{
					NULL, "PC-DCN", 4, 3
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncAvgPCtoNC,
					false
				}
			},
			{
				gtk_adjustment_new(gAMPAIncMFtoNC, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 7, 0, 3,
				{
					NULL, "MF-DCN", 6, 0
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gAMPAIncMFtoNC,
					false
				}
			},
			{
				gtk_adjustment_new(gIncNCtoIO, 0.0, 1.0, 0.0001, 0.1, 0.0),
				NULL, 7, 1, 4,
				{
					NULL, "DCN-IO", 6, 1
				},
				{
					"activate",
					G_CALLBACK(on_update_weight),
					&gIncNCtoIO,
					false
				}
			}
		}
	};

	// set window props
	gtk_window_set_title(GTK_WINDOW(tw.window), "Tuning");
	gtk_window_set_default_size(GTK_WINDOW(tw.window),
								DEFAULT_TUNING_WINDOW_WIDTH,
								DEFAULT_TUNING_WINDOW_HEIGHT);
	gtk_window_set_position(GTK_WINDOW(tw.window), GTK_WIN_POS_CENTER);
	gtk_window_set_resizable(GTK_WINDOW(tw.window), FALSE);
	gtk_container_set_border_width(GTK_CONTAINER(tw.window), 5);

	//set grid props
	gtk_grid_set_column_spacing(GTK_GRID(tw.grid), 3);
	gtk_grid_set_row_spacing(GTK_GRID(tw.grid), 3);
	
	//set button props
	FOREACH(tw.tuning_buttons, b)
	{
		// set button label
		b->label.label = gtk_label_new(b->label.string);
		gtk_grid_attach(GTK_GRID(tw.grid),
						b->label.label,
						b->label.col,
						b->label.row, 1, 1);

		// set button
		b->widget = gtk_spin_button_new(b->adjustment, 0.01, b->digits_to_display);
		gtk_widget_set_hexpand(b->widget, true);
		gtk_widget_set_vexpand(b->widget, true);
		gtk_grid_attach(GTK_GRID(tw.grid), b->widget, b->col, b->row, 1, 1);
		g_signal_connect(b->widget, b->signal.signal, b->signal.handler, b->signal.data);
	}

	gtk_container_add(GTK_CONTAINER(tw.window), tw.grid);
	gtk_widget_show_all(tw.window);
}

static void on_toggle_run(GtkWidget *widget, struct gui *gui)
{
	if (gui->ctrl_ptr->simState && gui->ctrl_ptr->simCore && gui->ctrl_ptr->mfFreq && gui->ctrl_ptr->mfs)
	{
		switch (gui->ctrl_ptr->run_state)
		{
			case NOT_IN_RUN:
				// TODO: start sim in a new thread
				gtk_button_set_label(GTK_BUTTON(widget), "Pause");
				gui->ctrl_ptr->run_state = IN_RUN_NO_PAUSE;
				gtk_widget_show(gui->normal_buttons[1].widget);
				// FIXME: thats a segfault: where is the object this guy is called upon???
				//g_thread_new("sim_thread", (GThreadFunc)&Control::runSession, gui);
				gui->ctrl_ptr->runSession(gui);
				gtk_button_set_label(GTK_BUTTON(widget), "Run");
				gtk_widget_hide(gui->normal_buttons[1].widget);
				break;
			case IN_RUN_NO_PAUSE:
				gtk_button_set_label(GTK_BUTTON(widget), "Continue");
				gui->ctrl_ptr->run_state = IN_RUN_PAUSE;
				break;
			case IN_RUN_PAUSE:
				gtk_button_set_label(GTK_BUTTON(widget), "Pause");
				gui->ctrl_ptr->run_state = IN_RUN_NO_PAUSE;
				break;
		}
	}
	else
	{
		LOG_WARN("trying to run an uninitialized simulation.");
		LOG_WARN("(Hint: initialize the simulation before running it.)");
	}
}

static void on_exit_sim(GtkWidget *widget, struct gui *gui)
{
	gtk_button_set_label(GTK_BUTTON(gui->normal_buttons[0].widget), "Run");
	gui->ctrl_ptr->run_state = NOT_IN_RUN;
	gtk_widget_hide(widget);
}

static void draw_raster(GtkWidget *drawing_area, cairo_t *cr, uint32_t trial, uint32_t num_cells,
	  uint32_t num_row, uint8_t **raster_data)
{
	// background color setup
	cairo_set_source_rgb(cr, 0, 0, 0);
	cairo_paint(cr);

	/* GtkDrawingArea size */
	GdkRectangle da;            
	GdkWindow *window = gtk_widget_get_window(GTK_WIDGET(drawing_area));
	
	/* Determine GtkDrawingArea dimensions */
	gdk_window_get_geometry(window,
	        &da.x,
	        &da.y,
	        &da.width,
	        &da.height);

	// raster sample size == 4096
	// TODO: add pixel_height_per_cell var
	float raster_to_pixel_scale_y = da.height / (float)num_cells;
	float raster_to_pixel_scale_x = da.width / (float)num_row;

	cairo_translate(cr, 0, da.height);
	cairo_scale(cr, raster_to_pixel_scale_x, -raster_to_pixel_scale_y);

	// point color
	cairo_set_source_rgb(cr, 0.0, 1.0, 0.0);

	uint32_t trial_start = trial * num_row;
	uint32_t trial_end = trial_start + num_row;

	for (uint32_t i = 0; i < num_cells; i++)
	{
		for (uint32_t j = trial_start; j < trial_end; j++)
		{
			if (raster_data[j][i])
			{
				cairo_rectangle(cr, j - trial_start, i, 2, 2);
				cairo_fill(cr);
			}
		}
	}
} 

static void draw_spatial_activity(GtkWidget *drawing_area, cairo_t *cr, Control *control)
{
	/*
	 * the plan: 
	 *     for cell coordinates x and y (in the square: [0, num_cell) x [0, num_cell)), if cell at (x, y)
	 *     fires at time step t_i, draw a dot with alpha 1. for each time step after a spike, decrease alpha
	 *     by a fixed amount.
	 *
	 *     for now, call either g_threads_add_idle or directly emit signal to re-draw every time step during a run.
	 *     this allows for the view to animate at a reasonable pace (e.g. 60 fps) when it is up, and then run the 
	 *     simulation at full speed when it is not. Useful for when you want to check network activity after a 
	 *     certain number of training or forgetting trials
	 *
	 */
}

static void draw_gr_raster(GtkWidget *drawing_area, cairo_t *cr, Control *control)
{
	/* 4096 gr raster sample size */
	draw_raster(drawing_area, cr, 0, 4096, control->msMeasure, control->rasters[GR]);
}

static void draw_go_raster(GtkWidget *drawing_area, cairo_t *cr, Control *control)
{
	draw_raster(drawing_area, cr, control->trial, num_go, control->msMeasure, control->rasters[GO]);
}

/* weights plot */
static void draw_pf_pc_plot(GtkWidget *drawing_area, cairo_t *cr, Control *control)
{
	const float *pfpc_weights = control->simCore->getMZoneList()[0]->exportPFPCWeights();
	for (int i = 0; i < 4096; i++)
	{
		control->sample_pfpc_syn_weights[i] = pfpc_weights[i];
	}

	// background color setup
	cairo_set_source_rgb(cr, 0, 0, 0);
	cairo_paint(cr);

	/* GtkDrawingArea size */
	GdkRectangle da;            
	GdkWindow *window = gtk_widget_get_window(GTK_WIDGET(drawing_area));
	
	/* Determine GtkDrawingArea dimensions */
	gdk_window_get_geometry(window,
	        &da.x,
	        &da.y,
	        &da.width,
	        &da.height);

	float pfpc_w_to_pixel_scale_y = 1.0; /* weights bounded within [0, 1] */
	float pfpc_w_to_pixel_scale_x = da.width / (float)4096; /* gr sample size is 4096 */
 
	cairo_translate(cr, 0, da.height);
	cairo_scale(cr, pfpc_w_to_pixel_scale_x, -pfpc_w_to_pixel_scale_y);

	// point color
	cairo_set_source_rgb(cr, 0.0, 0.0, 1.0);

	// TODO: place scaling in above scale
	for (int i = 0; i < 4096; i++)
	{
		cairo_rectangle(cr, i, (int)(da.height * control->sample_pfpc_syn_weights[i]), 2, 2);
		cairo_fill(cr);
	}
}

// TODO: continue development and adjusting scale params
static void draw_pc_plot(GtkWidget *drawing_area, cairo_t *cr, Control *control)
{
	// background color setup
	cairo_set_source_rgb(cr, 0, 0, 0);
	cairo_paint(cr);

	/* GtkDrawingArea size */
	GdkRectangle da;            
	GdkWindow *window = gtk_widget_get_window(GTK_WIDGET(drawing_area));

	gdk_window_get_geometry(window,
	        &da.x,
	        &da.y,
	        &da.width,
	        &da.height);

	float len_scale_y = threshRestPC - threshMaxPC;
	float pc_w_to_pixel_scale_y = -da.height / (9.5 * len_scale_y);
	float pc_w_to_pixel_scale_x = da.width / (float)control->msMeasure;

	cairo_scale(cr, pc_w_to_pixel_scale_x, -pc_w_to_pixel_scale_y);
	cairo_translate(cr, 0, 17.0);

	cairo_set_source_rgb(cr, 0.075, 0.075, 0.075);
	cairo_rectangle(cr, control->msPreCS,
		-17,
		control->td.cs_lens[0],
		9.5 * len_scale_y);
	cairo_fill(cr);

	// pc point color
	cairo_set_source_rgb(cr, 1.0, 1.0, 1.0);
	unsigned int trial_start = control->trial * control->msMeasure;
	uint32_t k = trial_start;

	// FIXME: not drawing on second trial
	for (uint32_t i = 0; i < control->msMeasure; i++)
	{
		int alternator = 1;
		for (int j = 0; j < num_pc; j++)
		{
			float vm_ij = control->pc_vm_raster[i][j] + alternator * 0.2 * ceil(j/2.0) * len_scale_y; 
			cairo_rectangle(cr, i, vm_ij, 1.0, 0.05);
			if (control->rasters[PC][k][j])
			{
				cairo_rectangle(cr, i, vm_ij, 1.0, 1.0);
			}
			alternator *= -1;
		}
		cairo_fill(cr);
		k++;
	}

	// nc point color
	cairo_set_source_rgb(cr, 0.0, 1.0, 0.0);
	
	float nc_offset = -72.0;
	float nc_scale  = 0.55;

	k = trial_start;
	for (int i = 0; i < control->msMeasure; i++)
	{
		int alternator = 1;
		for (int j = 0; j < num_nc; j++)
		{
			float vm_ij = nc_scale * control->nc_vm_raster[i][j] + nc_offset + alternator * 0.2 * ceil(j/2.0) * len_scale_y; 
			cairo_rectangle(cr, i, vm_ij, 1.0, 0.05);
			if (control->rasters[NC][k][j])
			{
				cairo_rectangle(cr, i, vm_ij, 1.0, 1.0);
			}
			alternator *= -1;
		}
		cairo_fill(cr);
		k++;
	}

	// io point color
	cairo_set_source_rgb(cr, 0.0, 1.0, 1.0);
	
	float io_offset = -124.0;
	float io_scale  = 0.1;

	k = trial_start;
	for (int i = 0; i < control->msMeasure; i++)
	{
		int alternator = 1;
		for (int j = 0; j < num_io; j++)
		{
			float vm_ij = io_scale * control->io_vm_raster[i][j] + io_offset + alternator * 0.2 * ceil(j/2.0) * len_scale_y; 
			cairo_rectangle(cr, i, vm_ij, 1.0, 0.05);
			if (control->rasters[IO][k][j])
			{
				cairo_rectangle(cr, i, vm_ij, 1.0, 1.0);
			}
			alternator *= -1;
		}
		cairo_fill(cr);
		k++;
	}
} 

static void generate_plot(GtkWidget *widget, void (* draw_func)(GtkWidget *, cairo_t *, Control *),
	Control *control, const gchar *title, gint width, gint height)
{
	if (!control->sim_initialized)
	{
		std::cout << "[ERROR]: Simulation not initialized. Nothing to show...\n";
		return;
	}
	GtkWidget *child_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	
	gtk_window_set_title(GTK_WINDOW(child_window), title);
	gtk_window_set_default_size(GTK_WINDOW(child_window), width, height);
	gtk_window_set_resizable(GTK_WINDOW(child_window), FALSE);

	GtkWidget *drawing_area = gtk_drawing_area_new();
	gtk_widget_set_size_request(drawing_area, width, height);
	gtk_container_add(GTK_CONTAINER(child_window), drawing_area);
	g_signal_connect(G_OBJECT(drawing_area), "draw", G_CALLBACK(draw_func), control);
	gtk_widget_show_all(child_window);
}

static void on_quit(GtkWidget *widget, Control *control)
{
	control->run_state = NOT_IN_RUN;
	gtk_main_quit();
}

static void on_gr_raster(GtkWidget *widget, Control *control)
{
	generate_plot(widget, draw_gr_raster, control, "GR Rasters",
		DEFAULT_RASTER_WINDOW_WIDTH, DEFAULT_RASTER_WINDOW_HEIGHT);
}

static void on_go_raster(GtkWidget *widget, Control *control)
{
	generate_plot(widget, draw_go_raster, control, "GO Rasters",
		DEFAULT_RASTER_WINDOW_WIDTH, DEFAULT_RASTER_WINDOW_HEIGHT);
}

static void on_pfpc_window(GtkWidget *widget, Control *control)
{
	generate_plot(widget, draw_pf_pc_plot, control, "PF-PC Weights",
		DEFAULT_PFPC_WINDOW_WIDTH, DEFAULT_PFPC_WINDOW_HEIGHT);
}

static void on_pc_window(GtkWidget *widget, Control *control)
{
	generate_plot(widget, draw_pc_plot, control, "PC Window",
		DEFAULT_PC_WINDOW_WIDTH, DEFAULT_PC_WINDOW_HEIGHT);
}

static bool on_parameters(GtkWidget *widget, gpointer data)
{
	return assert(false, "[DEBUG]: Not implemented", __func__);
}

static void on_dcn_plast(GtkWidget *widget, Control *control)
{
	if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)))
	{
		LOG_INFO("mf-nc plasticity set to mode: Graded");
		control->mf_nc_plast = GRADED; 
	}
	else
	{
		LOG_INFO("mf-nc plasticity set to mode: Off");
		control->mf_nc_plast = OFF; 
	}
}

static void on_radio(GtkWidget *widget, Control *control)
{
	if (gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(widget)))
	{
		const gchar *this_rad_label = gtk_button_get_label(GTK_BUTTON(widget));
		LOG_INFO("pf-pc plasticity set to mode: %s", this_rad_label);
		if (this_rad_label == "Graded")
			control->pf_pc_plast = GRADED;
		else if (this_rad_label == "Binary")
			control->pf_pc_plast = BINARY;
		else if (this_rad_label == "Cascade")
			control->pf_pc_plast = CASCADE;
		else if (this_rad_label == "Off")
			control->pf_pc_plast = OFF; 
	}
}

static void set_gui_window_attribs(struct gui *gui)
{
	gtk_window_set_title(GTK_WINDOW(gui->window), "Main Window");
	gtk_window_set_default_size(GTK_WINDOW(gui->window), MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT);
	gtk_window_set_position(GTK_WINDOW(gui->window), GTK_WIN_POS_CENTER);

	gtk_widget_add_events(gui->window, GDK_DELETE);
	g_signal_connect(gui->window, "destroy", G_CALLBACK(on_quit), gui->ctrl_ptr);
}

static void set_gui_grid_attribs(struct gui *gui)
{
	gtk_widget_set_name(gui->grid, "grid");
}

static void set_gui_normal_button_attribs(struct gui *gui)
{
	FOREACH(gui->normal_buttons, b)
	{
		gtk_button_set_label(GTK_BUTTON(b->widget), b->label);
		gtk_widget_set_hexpand(b->widget, true);
		gtk_widget_set_vexpand(b->widget, true);
		gtk_grid_attach(GTK_GRID(gui->grid), b->widget, b->col, b->row, 1, 1);
		g_signal_connect(b->widget, b->signal.signal, b->signal.handler, b->signal.data);
	}
}

static void set_gui_dcn_plast_button_attribs(struct gui *gui)
{
	gtk_button_set_label(GTK_BUTTON(gui->dcn_plast_button.widget), gui->dcn_plast_button.label);
	gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gui->dcn_plast_button.widget), true);
	gtk_widget_set_hexpand(gui->dcn_plast_button.widget, true);
	gtk_widget_set_vexpand(gui->dcn_plast_button.widget, true);
	gtk_grid_attach(GTK_GRID(gui->grid),
		  			gui->dcn_plast_button.widget,
		  			gui->dcn_plast_button.col, 
		  			gui->dcn_plast_button.row,
		  			1,
		  			1);
	g_signal_connect(gui->dcn_plast_button.widget,
					 gui->dcn_plast_button.signal.signal,
					 gui->dcn_plast_button.signal.handler,
					 gui->dcn_plast_button.signal.data);
}

static void set_gui_radio_button_attribs(struct gui *gui)
{
	gtk_grid_attach(GTK_GRID(gui->grid), gui->plast_radio_label, 0, 2, 1, 1); /* hard coded col, row nums for now */
	GtkWidget *group = NULL;
	gint radio_mask = 0;
	FOREACH(gui->plasticity_radios, r)
	{
		gtk_button_set_label(GTK_BUTTON(r->widget), r->label);
		gtk_widget_set_hexpand(r->widget, true);
		gtk_widget_set_vexpand(r->widget, true);
		gtk_radio_button_join_group(GTK_RADIO_BUTTON(r->widget), GTK_RADIO_BUTTON(group));
		if (r->label == "Graded") group = r->widget; /* hack to ensure first radio group NULL, rest are first radio */
		gtk_grid_attach(GTK_GRID(gui->grid), r->widget, r->col, r->row, 1, 1);
		g_signal_connect(r->widget, r->signal.signal, r->signal.handler, r->signal.data);
		radio_mask++;
	}
}

static void set_gui_menu_item_helper(struct menu_item *menu_item); /* forward declare */

static void set_gui_menu_helper(struct menu *menu)
{
   FOREACH_NELEM(menu->menu_items, menu->num_menu_items, mi)
   {
		if (mi) /* recursion always terminates on sub_menus, so somewhat unnecessary */
		{
			if (mi->label != "") /* by extension, every menu_item has a label */
			{
				gtk_menu_item_set_label(GTK_MENU_ITEM(mi->menu_item), mi->label);
				// careful: when mi->sub_menu.menu is NULL, removes this mi's submenu,
				// though since we are initializing, this should effectively do nothing
				gtk_menu_item_set_submenu(GTK_MENU_ITEM(mi->menu_item), mi->sub_menu.menu); 
			}
			if (mi->signal.signal) /* little hack for checking whether mi->signal was non empty initialized */
			{
				if (mi->signal.swapped)
				{
					g_signal_connect_swapped(mi->menu_item, mi->signal.signal, mi->signal.handler, mi->signal.data);
				}
				else
				{
					g_signal_connect(mi->menu_item, mi->signal.signal, mi->signal.handler, mi->signal.data);
				}
			}
			gtk_menu_shell_append(GTK_MENU_SHELL(menu->menu), mi->menu_item);
			set_gui_menu_item_helper(mi);
		}
   }
}

static void set_gui_menu_item_helper(struct menu_item *menu_item)
{
	if (!menu_item->sub_menu.menu) return; // little hack for checking whether sub_menu was empty initialized
	set_gui_menu_helper(&menu_item->sub_menu);
}

static void set_gui_menu_attribs(struct gui *gui)
{
	set_gui_menu_helper(&gui->menu_bar);
}

static void free_gui_menu_helper(struct menu *menu)
{
	FOREACH_NELEM(menu->menu_items, menu->num_menu_items, mi)
	{
		if (mi->sub_menu.menu) free_gui_menu_helper(&mi->sub_menu);
	}
	delete[] menu->menu_items;
}

static void free_gui_menus(struct gui *gui)
{
	free_gui_menu_helper(&gui->menu_bar);
} 

int gui_init_and_run(int *argc, char ***argv, Control *control)
{
	if (!gtk_init_check(argc, argv))
	{
		LOG_FATAL("Could not initialize GTK");
		return 1;
	}

	save_data save_data[NUM_SAVE_OPTS] = {
		{control, MF_RAST}, {control, GR_RAST}, {control, GO_RAST}, {control, BC_RAST},
		{control, SC_RAST}, {control, PC_RAST}, {control, IO_RAST}, {control, NC_RAST},
		{control, MF_PSTH}, {control, GR_PSTH}, {control, GO_PSTH}, {control, BC_PSTH},
		{control, SC_PSTH}, {control, PC_PSTH}, {control, IO_PSTH}, {control, NC_PSTH},
		{control, PFPC}, {control, MFNC}, {control, SIM}
	};

	struct gui gui = {
		.window = gtk_window_new(GTK_WINDOW_TOPLEVEL),
		.grid = gtk_grid_new(),
		.normal_buttons = {
			{"Run", gtk_button_new(), 0, 0,
				{ "clicked", G_CALLBACK(on_toggle_run), &gui, false }
			},
			{"Exit Sim", gtk_button_new(), 0, 1,
				{ "clicked", G_CALLBACK(on_exit_sim), &gui, false }
			},
			{"GR Raster", gtk_button_new(), 1, 0,
				{ "clicked", G_CALLBACK(on_gr_raster), control, false }
			},
			{"GO Raster", gtk_button_new(), 1, 1,
				{ "clicked", G_CALLBACK(on_go_raster), control, false }
			},
			{"PC Window", gtk_button_new(), 1, 2,
				{ "clicked", G_CALLBACK(on_pc_window), control, false }
			},
			{"Parameters", gtk_button_new(), 1, 3,
				{ "clicked", G_CALLBACK(on_parameters), NULL, false }
			},
		},
		.dcn_plast_button = {
			"DCN Plasticity", gtk_check_button_new(), 1, 4,
			{ "clicked", G_CALLBACK(on_dcn_plast), control, false }
		},
		.plast_radio_label = gtk_label_new("Plasticity"),
		.plasticity_radios = {
			{
				"Graded", gtk_radio_button_new(NULL), 0, 3,
				{ "toggled", G_CALLBACK(on_radio), control, false }
			},
			{
				"Binary", gtk_radio_button_new(NULL), 0, 4,
				{ "toggled", G_CALLBACK(on_radio), control, false }
			}, 
			{
				"Cascade", gtk_radio_button_new(NULL), 0, 5,
				{ "toggled", G_CALLBACK(on_radio), control, false }
			},
			{
				"Off", gtk_radio_button_new(NULL), 0, 6,
				{ "toggled", G_CALLBACK(on_radio), control, false }
			}
		},
		.menu_bar = {
			.menu = gtk_menu_bar_new(),
			.num_menu_items = NUM_SUB_MENU_ITEMS,
			.menu_items = new menu_item[NUM_SUB_MENU_ITEMS]
			{
				{"File", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_FILE_MENU_ITEMS, new menu_item[NUM_FILE_MENU_ITEMS]
						{
							{"Load...", gtk_menu_item_new(), {},
								{gtk_menu_new(), NUM_FILE_SUB_MENU_ITEMS, new menu_item[NUM_FILE_SUB_MENU_ITEMS]
									{
										{"Session File", gtk_menu_item_new(),
											{ "activate", G_CALLBACK(on_load_session_file), control, false },
											{}
										},
										{"Simulation File", gtk_menu_item_new(),
											{ "activate", G_CALLBACK(on_load_sim_file), control, false },
											{}
										}
									}
								}
							},
							{"Save Sim", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[SIM], false },
								{}
							},
							{"Create Output Dir", gtk_menu_item_new(),
								{"activate", G_CALLBACK(on_create_dir), &gui, false}, {}},
							{"", gtk_separator_menu_item_new(), {}, {}},
							{"Quit", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_quit), control, false },
								{}
							},
						}
					}
				},
				{"Weights", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_WEIGHTS_MENU_ITEMS, new menu_item[NUM_WEIGHTS_MENU_ITEMS]
						{
							{"Save PF-PC", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[PFPC], false },
								{}
							},
							{"Load PF-PC", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_load_pfpc_weights), control, false },
								{}
							},
							{"Save MF-DN", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[MFNC], false },
								{}
							},
							{"Load MF-DN", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_load_mfdcn_weights), control, false },
								{}
							}
						}
					}
				},
				{"Raster", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_RASTER_MENU_ITEMS, new menu_item[NUM_RASTER_MENU_ITEMS]
						{
							{"Save GR", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[GR_RAST], false },
								{}
							},
							{"Save GO", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[GO_RAST], false },
								{}
							},
							{"Save PC", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[PC_RAST], false },
								{}
							},
							{"Save DCN", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[NC_RAST], false },
								{}
							},
							{"Save IO", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[IO_RAST], false },
								{}
							},
							{"Save BC", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[BC_RAST], false },
								{}
							},
							{"Save SC", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[SC_RAST], false },
								{}
							},
							{"Save MF", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[MF_RAST], false },
								{}
							},
						}
					}
				},
				{"PSTH", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_RASTER_MENU_ITEMS, new menu_item[NUM_RASTER_MENU_ITEMS]
						{
							{"Save GR", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[GR_PSTH], false },
								{}
							},
							{"Save GO", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[GO_PSTH], false },
								{}
							},
							{"Save PC", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[PC_PSTH], false },
								{}
							},
							{"Save DCN", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[NC_PSTH], false },
								{}
							},
							{"Save IO", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[IO_PSTH], false },
								{}
							},
							{"Save BC", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[BC_PSTH], false },
								{}
							},
							{"Save SC", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[SC_PSTH], false },
								{}
							},
							{"Save MF", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_save_file), &save_data[MF_PSTH], false },
								{}
							},
						}
					}
				},
				{"Analysis", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_ANALYSIS_MENU_ITEMS, new menu_item[NUM_ANALYSIS_MENU_ITEMS]
						{
							{"Firing Rates", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_firing_rates_window), &gui, false },
								{}
							},
						}
					}
				},
				{"Tuning", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_TUNING_MENU_ITEMS, new menu_item[NUM_TUNING_MENU_ITEMS]
						{
							{"Tuning", gtk_menu_item_new(),
								{ "activate", G_CALLBACK(on_tuning_window), &gui, false },
								{}
							},
						}
					}
				}
			}
		},
		.frw = {
			.window = NULL,
			.grid = NULL,
			.headers = {
				{ NULL,            "Cell", 0, 0 },
				{ NULL,   "Non-CS r_mean", 1, 0 },
				{ NULL, "Non-CS r_median", 2, 0 },
				{ NULL,       "CS r_mean", 3, 0 },
				{ NULL,     "CS r_median", 4, 0 },
			},
			.cell_labels = {
				{
					{ NULL,   "MF", 0, 1 },
					{ NULL, "0.00", 1, 1 },
					{ NULL, "0.00", 2, 1 },
					{ NULL, "0.00", 3, 1 },
					{ NULL, "0.00", 4, 1 },
				},
				{
					{ NULL,   "GR", 0, 2 },
					{ NULL, "0.00", 1, 2 },
					{ NULL, "0.00", 2, 2 },
					{ NULL, "0.00", 3, 2 },
					{ NULL, "0.00", 4, 2 },
				},
				{
					{ NULL,   "GO", 0, 3 },
					{ NULL, "0.00", 1, 3 },
					{ NULL, "0.00", 2, 3 },
					{ NULL, "0.00", 3, 3 },
					{ NULL, "0.00", 4, 3 },
				},
				{
					{ NULL,   "BC", 0, 4 },
					{ NULL, "0.00", 1, 4 },
					{ NULL, "0.00", 2, 4 },
					{ NULL, "0.00", 3, 4 },
					{ NULL, "0.00", 4, 4 },
				},
				{
					{ NULL,   "SC", 0, 5 },
					{ NULL, "0.00", 1, 5 },
					{ NULL, "0.00", 2, 5 },
					{ NULL, "0.00", 3, 5 },
					{ NULL, "0.00", 4, 5 },
				},
				{
					{ NULL,   "PC", 0, 6 },
					{ NULL, "0.00", 1, 6 },
					{ NULL, "0.00", 2, 6 },
					{ NULL, "0.00", 3, 6 },
					{ NULL, "0.00", 4, 6 },
				},
				{
					{ NULL,   "IO", 0, 7 },
					{ NULL, "0.00", 1, 7 },
					{ NULL, "0.00", 2, 7 },
					{ NULL, "0.00", 3, 7 },
					{ NULL, "0.00", 4, 7 },
				},
				{
					{ NULL,  "DCN", 0, 8 },
					{ NULL, "0.00", 1, 8 },
					{ NULL, "0.00", 2, 8 },
					{ NULL, "0.00", 3, 8 },
					{ NULL, "0.00", 4, 8 },
				}
			}
		},
		.ctrl_ptr = control /* big yikes move */
	}; 

	// set all attribs
	set_gui_window_attribs(&gui);
	set_gui_grid_attribs(&gui);
	set_gui_normal_button_attribs(&gui);
	set_gui_dcn_plast_button_attribs(&gui);
	set_gui_radio_button_attribs(&gui);
	set_gui_menu_attribs(&gui);

	// organize menu bar and grid vertically
	GtkWidget *v_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	gtk_container_add(GTK_CONTAINER(gui.window), v_box);
	gtk_box_pack_start(GTK_BOX(v_box), gui.menu_bar.menu, FALSE, TRUE, 0);
	gtk_box_pack_start(GTK_BOX(v_box), gui.grid, FALSE, TRUE, 0);

	//GtkDialogFlags flags = GTK_DIALOG_DESTROY_WITH_PARENT;
	//GtkWidget *dialog = gtk_message_dialog_new (GTK_WINDOW(gui.window),
	//								 GTK_DIALOG_MODAL,
	//								 GTK_MESSAGE_INFO,
	//								 GTK_BUTTONS_NONE,
	//								 "CbmSim - A Cerebellar Simulator\n\n"
	//								 "Welcome to the Cerebellar Simulator! The goal of this software is to\n"
	//								 "simulate dynamics of the micro-circuitry within the cerebellar cortex.\n"
	//								 "See the \"help\" menu to get started and for further information on the\n"
	//								 "various features included in this program. Also, see the \"about\" menu\n"
	//								 "for more information about the history of this project's development");
	//GtkWidget *msg_area = gtk_message_dialog_get_message_area(GTK_MESSAGE_DIALOG(dialog));
	//GtkWidget *check_btn_label = gtk_label_new("hello world");
	//GtkWidget *dialog_check_btn = gtk_check_button_new_with_label("Don't show me this again.");
	//gtk_box_pack_start(GTK_BOX(msg_area), check_btn_label, FALSE, TRUE, 0);
	//gtk_box_pack_start(GTK_BOX(msg_area), dialog_check_btn, FALSE, TRUE, 0);

	// show main widnow 
	gtk_widget_show_all(gui.window);
	gtk_widget_hide(gui.normal_buttons[1].widget);

	// present the dialog 
	//gtk_dialog_run (GTK_DIALOG(dialog));
	//gtk_widget_destroy (dialog);

	// run main event loop after dialog is exited out of 
	gtk_main();

	// manually delete objects we created
	free_gui_menus(&gui);

	return 0;
}

