#include "array_util.h"
#include "params/connectivityparams.h"
#include "gui.h"

// temp function so gtk doesn't whine abt NULL callbacks
static void null_callback(GtkWidget *widget, gpointer data) {}

static bool assert(bool expr, const char *error_string, const char *func = "assert")
{
	if (!expr)
	{
		fprintf(stderr, "%s(): %s\n", func, error_string);
		return false;
	}
	return true;
}


static void load_file(GtkWidget *widget, Control *control, std::string &out_file_name)
{
	GtkWidget *dialog = gtk_file_chooser_dialog_new
		(
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
		char *sim_file = gtk_file_chooser_get_filename(chooser);
		// TODO: pop-up warning for invalid file
		out_file_name = std::string(sim_file);
		g_free(sim_file);
	}

	gtk_widget_destroy(dialog);

}

static void on_load_experiment_file(GtkWidget *widget, Control *control)
{
		std::string expt_filename = "";
		load_file(widget, control, expt_filename);
		if (expt_filename == "")
		{
			fprintf(stderr, "[ERROR]: Could not open experiment file.\n");
			return;
		}
		control->init_experiment(expt_filename);
}

static void on_load_sim_file(GtkWidget *widget, Control *control)
{
		std::string in_sim_filename = "";
		load_file(widget, control, in_sim_filename);
		if (in_sim_filename == "")
		{
			fprintf(stderr, "[ERROR]: Could not open simulation file.\n");
			return;
		}
		// set input state file name for saving state after each trial
		control->inSimFileName = in_sim_filename;
		control->init_sim(in_sim_filename);
}

static void on_save_state(GtkWidget *widget, Control *control)
{
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

	gtk_file_chooser_set_current_name(chooser, DEFAULT_STATE_FILE_NAME);

	gint response = gtk_dialog_run(GTK_DIALOG(dialog));

	if (response == GTK_RESPONSE_ACCEPT)
	{
		char *sim_state_file = gtk_file_chooser_get_filename(chooser);
		// TODO: pop-up warning for invalid file
		control->save_sim_state_to_file(std::string(sim_state_file));
		g_free(sim_state_file);
	}
	gtk_widget_destroy(dialog);
}

static void on_save_pfpc_weights(GtkWidget *widget, Control *control)
{
	// before we make the file chooser dialog, should determine whether we can even do this operation
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

	gtk_file_chooser_set_current_name(chooser, DEFAULT_PFPC_WEIGHT_NAME);

	gint response = gtk_dialog_run(GTK_DIALOG(dialog));

	if (response == GTK_RESPONSE_ACCEPT)
	{
		char *pfpc_weights_file = gtk_file_chooser_get_filename(chooser);
		// TODO: pop-up warning for invalid file
		control->save_pfpc_weights_to_file(std::string(pfpc_weights_file));
		g_free(pfpc_weights_file);
	}
	gtk_widget_destroy(dialog);
}

static void on_save_mfdcn_weights(GtkWidget *widget, Control *control)
{
	// before we make the file chooser dialog, should determine whether we can even do this operation
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

	gtk_file_chooser_set_current_name(chooser, DEFAULT_MFDCN_WEIGHT_NAME);

	gint response = gtk_dialog_run(GTK_DIALOG(dialog));

	if (response == GTK_RESPONSE_ACCEPT)
	{
		char *mfdcn_weights_file = gtk_file_chooser_get_filename(chooser);
		// TODO: pop-up warning for invalid file
		control->save_mfdcn_weights_to_file(std::string(mfdcn_weights_file));
		g_free(mfdcn_weights_file);
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
	gui->frw.window = gtk_window_new(GTK_WINDOW_TOPLEVEL),
	gui->frw.grid = gtk_grid_new(),

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

void update_weight(GtkWidget *spin_button, float *weight)
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
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 1, 0,
				{
					NULL, "MF-GR", 0, 0
				},
				{
					"activate",
					G_CALLBACK(update_weight),
					&gIncDirectMFtoGR,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 1, 1,
				{
					NULL, "MF-GO", 0, 1
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 1, 2,
				{
					NULL, "GR-GO", 0, 2
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 1, 3,
				{
					NULL, "GO-GR", 0, 3
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 3, 0,
				{
					NULL, "GO-GO", 2, 0
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 3, 1,
				{
					NULL, "GR-PC", 2, 1
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 3, 2,
				{
					NULL, "GR-SC", 2, 2
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 3, 3,
				{
					NULL, "GR-BC", 2, 3
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 5, 0,
				{
					NULL, "SC-PC", 4, 0
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 5, 1,
				{
					NULL, "BC-PC", 4, 1
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 5, 2,
				{
					NULL, "PC-BC", 4, 2
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 5, 3,
				{
					NULL, "PC-DCN", 4, 3
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 7, 0,
				{
					NULL, "MF-DCN", 6, 0
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
					false
				}
			},
			{
				gtk_adjustment_new(0.035, 0.0, 1.0, 0.001, 0.1, 0.0),
				NULL, 7, 1,
				{
					NULL, "DCN-IO", 6, 1
				},
				{
					"activate",
					G_CALLBACK(null_callback),
					NULL,
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
		b->widget = gtk_spin_button_new(b->adjustment, 0.001, 3);
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
				gui->ctrl_ptr->runTrials(0, 0, 0, 0, gui);
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
		//if (gui->ctrl_ptr->terminate)
		//{
		//	std::cout << "[INFO] Finished running trials in "
		//	          << (float) time / CLOCKS_PER_SEC << "s." << std::endl;
		//   gtk_main_quit();
		//}
		//else
		//{
		//	std::cout << "[INFO] Finished running all trials in "
		//	          << (float) time / CLOCKS_PER_SEC << "s." << std::endl;
		//}
	}
	else
	{
		fprintf(stderr, "[ERROR]: trying to run an uninitialized simulation.\n");
		fprintf(stderr, "[ERROR]: (Hint: initialize the simulation before running it.)\n");
	}
}

static void on_exit_sim(GtkWidget *widget, struct gui *gui)
{
	// TODO: change label of run button back to run
	gtk_button_set_label(GTK_BUTTON(gui->normal_buttons[0].widget), "Run");
	gui->ctrl_ptr->run_state = NOT_IN_RUN;
	gtk_widget_hide(widget);
}

static void draw_raster(GtkWidget *drawing_area, cairo_t *cr, ct_uint32_t trial, ct_uint32_t num_cells,
	  ct_uint32_t num_col, ct_uint8_t **raster_data)
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
	float raster_to_pixel_scale_x = da.width / (float)num_col;

	cairo_translate(cr, 0, da.height);
	cairo_scale(cr, raster_to_pixel_scale_x, -raster_to_pixel_scale_y);

	// point color
	cairo_set_source_rgb(cr, 0.0, 1.0, 0.0);

	for (int i = 0; i < num_cells; i++)
	{
		for (int j = 0; j < num_col; j++)
		{
			if (raster_data[i][j])
			{
				cairo_rectangle(cr, j, i, 2, 2);
				cairo_fill(cr);
			}
		}
	}
} 

static void draw_gr_raster(GtkWidget *drawing_area, cairo_t *cr, Control *control)
{
	/* 4096 gr raster sample size */
	draw_raster(drawing_area, cr, control->trial, 4096, control->PSTHColSize, control->sample_gr_rast_internal);
}

static void draw_go_raster(GtkWidget *drawing_area, cairo_t *cr, Control *control)
{
	draw_raster(drawing_area, cr, control->trial, num_go, control->PSTHColSize, control->all_go_rast_internal);
}

static void draw_pf_pc_plot(GtkWidget *drawing_area, cairo_t *cr, Control *control)
{
	const float *pfpc_weights = control->simCore->getMZoneList()[0]->exportPFPCWeights();
	for (int i = 0; i < 4096; i++)
	{
		control->sample_pfpc_syn_weights[i] = pfpc_weights[control->gr_indices[i]];
	}
	//std::sort(control->sample_pfpc_syn_weights, control->sample_pfpc_syn_weights + 4096,
	//	  [](float a, float b){return abs(a-0.5) < abs(b-0.5);});

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

static void generate_raster_plot(GtkWidget *widget,
	  void (* draw_func)(GtkWidget *, cairo_t *, Control *), Control *control)
{
	GtkWidget *child_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	
	gtk_window_set_title(GTK_WINDOW(child_window), "Raster Plot");
	gtk_window_set_default_size(GTK_WINDOW(child_window),
								DEFAULT_RASTER_WINDOW_WIDTH,
								DEFAULT_RASTER_WINDOW_HEIGHT);
	gtk_window_set_resizable(GTK_WINDOW(child_window), FALSE);

	GtkWidget *drawing_area = gtk_drawing_area_new();
	gtk_widget_set_size_request(drawing_area,
								DEFAULT_RASTER_WINDOW_WIDTH,
								DEFAULT_RASTER_WINDOW_HEIGHT);
	gtk_container_add(GTK_CONTAINER(child_window), drawing_area);
	g_signal_connect(G_OBJECT(drawing_area), "draw", G_CALLBACK(draw_func), control);
	gtk_widget_show_all(child_window);
}

static void generate_pfpc_plot(GtkWidget *widget,
	  void (* draw_func)(GtkWidget *, cairo_t *, Control *), Control *control)
{
	GtkWidget *child_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
	
	gtk_window_set_title(GTK_WINDOW(child_window), "PFPC Weights");
	gtk_window_set_default_size(GTK_WINDOW(child_window),
								DEFAULT_PFPC_WINDOW_WIDTH,
								DEFAULT_PFPC_WINDOW_HEIGHT);
	gtk_window_set_resizable(GTK_WINDOW(child_window), FALSE);

	GtkWidget *drawing_area = gtk_drawing_area_new();
	gtk_widget_set_size_request(drawing_area,
								DEFAULT_PFPC_WINDOW_WIDTH,
								DEFAULT_PFPC_WINDOW_HEIGHT);
	gtk_container_add(GTK_CONTAINER(child_window), drawing_area);
	g_signal_connect(G_OBJECT(drawing_area), "draw", G_CALLBACK(draw_func), control);
	gtk_widget_show_all(child_window);

}

static void on_quit(GtkWidget *widget, Control *control)
{
	if (control->in_run)
	{
		control->in_run = false;
		control->terminate = true;
	}
	else gtk_main_quit();
}


static void on_gr_raster(GtkWidget *widget, Control *control)
{
	generate_raster_plot(widget, draw_gr_raster, control);
}

static void on_go_raster(GtkWidget *widget, Control *control)
{
	generate_raster_plot(widget, draw_go_raster, control);
}

static void on_pc_window(GtkWidget *widget, Control *control)
{
	generate_pfpc_plot(widget, draw_pf_pc_plot, control);
}

static bool on_parameters(GtkWidget *widget, gpointer data)
{
	return assert(false, "Not implemented", __func__);
}

static bool on_dcn_plast(GtkWidget *widget, gpointer data)
{
	return assert(false, "Not implemented", __func__);
}

static bool on_radio(GtkWidget *widget, gpointer data)
{
	return assert(false, "Not implemented", __func__);
}

// TODO: make width and height params to this functn
static void set_gui_window_attribs(struct gui *gui)
{
	gtk_window_set_title(GTK_WINDOW(gui->window), "Main Window");
	gtk_window_set_default_size(GTK_WINDOW(gui->window), MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT);
	gtk_window_set_position(GTK_WINDOW(gui->window), GTK_WIN_POS_CENTER);

	gtk_widget_add_events(gui->window, GDK_DELETE);
	g_signal_connect(gui->window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
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
		//g_signal_connect(r->widget, "toggled", r->handler, (gpointer) &radio_mask); // IFFY!
		radio_mask++;
	}
}

static void set_gui_menu_item_helper(struct menu_item *menu_item); /* forward declare */

// TODO: debug
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
		fprintf(stderr, "Could not initialize GTK\n");
		return 1;
	}

	struct gui gui = {
		.window = gtk_window_new(GTK_WINDOW_TOPLEVEL),
		.grid = gtk_grid_new(),
		.normal_buttons = {
			{"Run", gtk_button_new(), 0, 0,
				{
					"clicked",
					G_CALLBACK(on_toggle_run),
					&gui,
					false
				}
			},
			{"Exit Sim", gtk_button_new(), 0, 1,
				{
					"clicked",
					G_CALLBACK(on_exit_sim),
					&gui,
					false
				}
			},
			{"GR Raster", gtk_button_new(), 1, 0,
				{
				   "clicked",
				   G_CALLBACK(on_gr_raster),
				   control,
				   false
				}
			},
			{"GO Raster", gtk_button_new(), 1, 1,
				{
				   "clicked",
				   G_CALLBACK(on_go_raster),
				   control,
				   false
				}
			},
			{"PC Window", gtk_button_new(), 1, 2,
				{
				   "clicked",
				   G_CALLBACK(on_pc_window),
				   control,
				   false
				}
			},
			{"Parameters", gtk_button_new(), 1, 3,
				{
				   "clicked",
				   G_CALLBACK(on_parameters),
				   NULL,
				   false
				}
			},
		},
		.dcn_plast_button = {
			"DCN Plasticity",
			gtk_check_button_new(),
			1,
			4,
			{
				"clicked",
				G_CALLBACK(on_dcn_plast),
				NULL,
				false
			}
		},
		.plast_radio_label = gtk_label_new("Plasticity"),
		.plasticity_radios = {
			{"Graded" , gtk_radio_button_new(NULL), 0, 3, {}},
			{"Binary" , gtk_radio_button_new(NULL), 0, 4, {}}, 
			{"Cascade", gtk_radio_button_new(NULL), 0, 5, {}},
			{"Off"    , gtk_radio_button_new(NULL), 0, 6, {}}
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
										{"Experiment File", gtk_menu_item_new(),
											{
												"activate",
												G_CALLBACK(on_load_experiment_file),
												control,
												false
											},
											{}
										},
										{"Simulation File", gtk_menu_item_new(),
											{
												"activate",
												G_CALLBACK(on_load_sim_file),
												control,
												false
											},
											{}
										}
									}
								}
							},
							{"Save State", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(on_save_state),
									control,
									false
								},
								{}
							},
							{"", gtk_separator_menu_item_new(), {}, {}},
							{"Quit", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(on_quit),
									control,
									false
								},  
								{}
							},
						}
					}
				},
				{"Weights", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_WEIGHTS_MENU_ITEMS, new menu_item[NUM_WEIGHTS_MENU_ITEMS]
						{
							{"Save PF-PC", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(on_save_pfpc_weights),
									control,
									false
								},
								{}
							},
							{"Load PF-PC", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Save MF-DN", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(on_save_mfdcn_weights),
									control,
									false
								},
								{}
							},
							{"Load MF-DN", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							}
						}
					}
				},
				{"PSTH", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_PSTH_MENU_ITEMS, new menu_item[NUM_PSTH_MENU_ITEMS]
						{
							{"Save GR", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Save GO", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Save PC", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Save DN", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Save CF", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Save BC", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Save SC", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Save MF", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
						}
					}
				},
				{"Analysis", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_ANALYSIS_MENU_ITEMS, new menu_item[NUM_ANALYSIS_MENU_ITEMS]
						{
							{"Firing Rates", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(on_firing_rates_window),
									&gui,
									false
								},
								{}
							},
						}
					}
				},
				{"Tuning", gtk_menu_item_new(), {},
					{gtk_menu_new(), NUM_TUNING_MENU_ITEMS, new menu_item[NUM_TUNING_MENU_ITEMS]
						{
							{"Tuning", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(on_tuning_window),
									&gui,
									false
								},
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
				{
					NULL, "Cell", 0, 0
				},
				{
					NULL, "Non-CS r_mean", 1, 0
				},
				{
					NULL, "Non-CS r_median", 2, 0
				},
				{
					NULL, "CS r_mean", 3, 0
				},
				{
					NULL, "CS r_median", 4, 0
				}
			},
			.cell_labels = {
				{
					{
						NULL, "MF", 0, 1
					},
					{
						NULL, "0.00", 1, 1
					},
					{
						NULL, "0.00", 2, 1
					},
					{
						NULL, "0.00", 3, 1
					},
					{
						NULL, "0.00", 4, 1
					}
				},
				{
					{
						NULL, "GR", 0, 2
					},
					{
						NULL, "0.00", 1, 2
					},
					{
						NULL, "0.00", 2, 2
					},
					{
						NULL, "0.00", 3, 2
					},
					{
						NULL, "0.00", 4, 2
					}
				},
				{
					{
						NULL, "GO", 0, 3
					},
					{
						NULL, "0.00", 1, 3
					},
					{
						NULL, "0.00", 2, 3
					},
					{
						NULL, "0.00", 3, 3
					},
					{
						NULL, "0.00", 4, 3
					}
				},
				{
					{
						NULL, "BC", 0, 4
					},
					{
						NULL, "0.00", 1, 4
					},
					{
						NULL, "0.00", 2, 4
					},
					{
						NULL, "0.00", 3, 4
					},
					{
						NULL, "0.00", 4, 4
					}
				},
				{
					{
						NULL, "SC", 0, 5
					},
					{
						NULL, "0.00", 1, 5
					},
					{
						NULL, "0.00", 2, 5
					},
					{
						NULL, "0.00", 3, 5
					},
					{
						NULL, "0.00", 4, 5
					}
				},
				{
					{
						NULL, "PC", 0, 6
					},
					{
						NULL, "0.00", 1, 6
					},
					{
						NULL, "0.00", 2, 6
					},
					{
						NULL, "0.00", 3, 6
					},
					{
						NULL, "0.00", 4, 6
					}
				},
				{
					{
						NULL, "IO", 0, 7
					},
					{
						NULL, "0.00", 1, 7
					},
					{
						NULL, "0.00", 2, 7
					},
					{
						NULL, "0.00", 3, 7
					},
					{
						NULL, "0.00", 4, 7
					}
				},
				{
					{
						NULL, "DCN", 0, 8
					},
					{
						NULL, "0.00", 1, 8
					},
					{
						NULL, "0.00", 2, 8
					},
					{
						NULL, "0.00", 3, 8
					},
					{
						NULL, "0.00", 4, 8
					}
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

	// show and run
	gtk_widget_show_all(gui.window);
	gtk_widget_hide(gui.normal_buttons[1].widget);
	gtk_main();

	// manually delete objects we created
	free_gui_menus(&gui);

	return 0;
}

