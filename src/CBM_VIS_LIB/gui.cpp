#include "array_util.h"
#include "control.h"
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

// for now we load in the activity params file and init the sim with a separate button
static void on_load_activity_param_file(GtkWidget *widget, Control *control)
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
		char *activity_file = gtk_file_chooser_get_filename(chooser);
		// TODO: pop-up warning for invalid file
		control->init_activity_params(std::string(activity_file));
		g_free(activity_file);
	}

	gtk_widget_destroy(dialog);
}

static void on_load_connectivity_state_file(GtkWidget *widget, Control *control)
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
		char *sim_state_file = gtk_file_chooser_get_filename(chooser);
		// TODO: pop-up warning for invalid file
		std::string sim_state_file_std_str = std::string(sim_state_file);
		//GThread *init_sim_worker = g_thread_new(
		//	  "init_state",
		//	  (* GThreadFunc)control->init_sim_state,
		//	  (gpointer)&sim_state_file_std_str);
		//control->init_sim_state(std::string(sim_state_file));
		g_free(sim_state_file);
	}

	gtk_widget_destroy(dialog);

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
		control->save_sim_state(std::string(sim_state_file));
		g_free(sim_state_file);
	}

	gtk_widget_destroy(dialog);

}

// NOTE: Assumes that activity params have been loaded!!!!
static void on_init_sim(GtkWidget *widget, Control *control)
{
	if (control->ap) control->construct_control();
	else 
	{
		fprintf(stderr, "[ERROR]: Trying to initialize a simulation without loading a file.\n");
		fprintf(stderr, "[ERROR]: (Hint: Load an activity parameter file first then initialize the simulation.)\n");
	}
}

static void on_run(GtkWidget *widget, Control *control)
{
	if (control->simState && control->simCore && control->mfFreq && control->mfs)
	{

		float mfW = 0.0035; // mf weights (to what?)
		float ws = 0.3275; // weight scale
		float gogr = 0.0105; // gogr weights

		float grW[8] = { 0.00056, 0.0007, 0.000933, 0.0014, 0.0028, 0.0056, 0.0112, 0.0224 };
		int grWLength = sizeof(grW) / sizeof(grW[0]);

		std::cout << "[INFO]: Running all simulations..." << std::endl;
		clock_t time = clock();
		for (int goRecipParamNum = 0; goRecipParamNum < 1; goRecipParamNum++)
		{
			float GRGO = grW[goRecipParamNum] * ws; // scaled grgo weights
			float MFGO = mfW * ws; // scaled mfgo weights
			float GOGR = gogr; // gogr weights, unchanged
			for (int simNum = 0; simNum < 1; simNum++)
			{
				std::cout << "[INFO]: Running simulation #" << (simNum + 1) << std::endl;
				control->runTrials(simNum, GOGR, GRGO, MFGO);
				// TODO: put in output file dir to save to!
				//control.saveOutputArraysToFile(goRecipParamNum, simNum);
			}
		}
		time = clock() - time;
		std::cout << "[INFO] All simulations finished in "
		          << (float) time / CLOCKS_PER_SEC << "s." << std::endl;
	}
	else
	{
		fprintf(stderr, "[ERROR]: trying to run an uninitialized simulation.\n");
		fprintf(stderr, "[ERROR]: (Hint: initialize the simulation before running it.)\n");
	}
}

static void on_pause(GtkWidget *widget, Control *control)
{
	control->sim_is_paused = true;
}

static void on_continue(GtkWidget *widget, Control *control)
{
	control->sim_is_paused = false;
}

static bool on_gr_raster(GtkWidget *widget, gpointer data)
{
	return assert(false, "Not implemented", __func__);
}

static bool on_go_raster(GtkWidget *widget, gpointer data)
{
	return assert(false, "Not implemented", __func__);
}

static bool on_pc_window(GtkWidget *widget, gpointer data)
{
	return assert(false, "Not implemented", __func__);
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
	gtk_window_set_default_size(GTK_WINDOW(gui->window), 600, 400);

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

int gui_init_and_run(int *argc, char ***argv)
{
	if (!gtk_init_check(argc, argv))
	{
		fprintf(stderr, "Could not initialize GTK\n");
		return 1;
	}

	Control *control = new Control(); 

	struct gui gui = {
		.window = gtk_window_new(GTK_WINDOW_TOPLEVEL),
		.grid = gtk_grid_new(),
		.normal_buttons = {
			{"Initialize Sim", gtk_button_new(), 0, 0,
				{
					"clicked",
					G_CALLBACK(on_init_sim),
					control,
					false
				}
			},
			{"Run", gtk_button_new(), 1, 0,
				{
					"clicked",
					G_CALLBACK(on_run),
					control,
					false
				}
			},
			{"Pause", gtk_button_new(), 0, 1,
				{
					"clicked",
					G_CALLBACK(on_pause),
					control,
					false
				}
			},
			{"Continue", gtk_button_new(), 1, 1,
				{
					"clicked",
					G_CALLBACK(on_continue),
					control,
					false
				}
			},
			{"GR Raster", gtk_button_new(), 3, 0,
				{
				   "clicked",
				   G_CALLBACK(on_gr_raster),
				   NULL,
				   false
				}
			},
			{"GO Raster", gtk_button_new(), 3, 1,
				{
				   "clicked",
				   G_CALLBACK(on_go_raster),
				   NULL,
				   false
				}
			},
			{"PC Window", gtk_button_new(), 3, 2,
				{
				   "clicked",
				   G_CALLBACK(on_pc_window),
				   NULL,
				   false
				}
			},
			{"Parameters", gtk_button_new(), 3, 3,
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
										{"Activity Parameter File", gtk_menu_item_new(),
											{
												"activate",
												G_CALLBACK(on_load_activity_param_file),
												control,
												false
											},
											{}
										},
										{"Connectivity State File", gtk_menu_item_new(),
											{
												"activate",
												G_CALLBACK(on_load_connectivity_state_file),
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
									G_CALLBACK(gtk_main_quit),
									NULL,
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
							{"Save Weights", gtk_menu_item_new(),
								{
									"activate",
									G_CALLBACK(null_callback),
									NULL,
									false
								},
								{}
							},
							{"Load Weights", gtk_menu_item_new(),
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
									G_CALLBACK(null_callback),
									NULL,
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
							{"Analysis", gtk_menu_item_new(),
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
				}
			}
		}
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

	// show, run, and free
	gtk_widget_show_all(gui.window);
	gtk_main();

	// manually delete objects we created
	free_gui_menus(&gui);
	delete control;

	return 0;
}

