#include "array_util.h"
#include "control.h"
#include "gui.h"

static bool assert(bool expr, const char *error_string, const char *func = "assert")
{
	if (!expr)
	{
		fprintf(stderr, "%s(): %s\n", func, error_string);
		return false;
	}
	return true;
}

static bool on_run(GtkWidget *widget, gpointer data)
{
	float mfW = 0.0035; // mf weights (to what?)
	float ws = 0.3275; // weight scale
	float gogr = 0.0105; // gogr weights

	float grW[8] = { 0.00056, 0.0007, 0.000933, 0.0014, 0.0028, 0.0056, 0.0112, 0.0224 };
	int grWLength = sizeof(grW) / sizeof(grW[0]);

	std::cout << "[INFO]: Running all simulations..." << std::endl;
	clock_t time = clock();
	//for (int goRecipParamNum = 0; goRecipParamNum < 1; goRecipParamNum++)
	//{
	//	float GRGO = grW[goRecipParamNum] * ws; // scaled grgo weights
	//	float MFGO = mfW * ws; // scaled mfgo weights
	//	float GOGR = gogr; // gogr weights, unchanged
	//	for (int simNum = 0; simNum < 1; simNum++)
	//	{
	//		std::cout << "[INFO]: Running simulation #" << (simNum + 1) << std::endl;
	//		Control control(actParamFile);
	//		control.runTrials(simNum, GOGR, GRGO, MFGO);
	//		// TODO: put in output file dir to save to!
	//		control.saveOutputArraysToFile(goRecipParamNum, simNum);
	//	}
	//}
	time = clock() - time;
	std::cout << "[INFO] All simulations finished in "
	          << (float) time / CLOCKS_PER_SEC << "s." << std::endl;
	return true;
}

static bool on_pause(GtkWidget *widget, gpointer data)
{
	return assert(false, "Not implemented", __func__);
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
	gtk_window_set_default_size(GTK_WINDOW(gui->window), 1280, 720);

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
		g_signal_connect(b->widget, "clicked", G_CALLBACK(b->handler), NULL);
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
		  			 "clicked",
					 gui->dcn_plast_button.handler,
					 NULL);
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
		g_signal_connect(r->widget, "toggled", r->handler, (gpointer) &radio_mask); // IFFY!
		radio_mask++;
	}
}

// TODO: make recursive
static void set_gui_menu_attribs(struct gui *gui)
{
	/* iterate through each sub menu and append all menu items to given sub menu */
	const int *sm_num_ptr = num_item_per_sub_menu;
	FOREACH_NELEM(gui->menu_bar.menu_items, NUM_SUB_MENUS, mi)
	{
		gtk_menu_item_set_label(GTK_MENU_ITEM(mi->menu_item), mi->label);
		gtk_menu_item_set_submenu(GTK_MENU_ITEM(mi->menu_item), mi->sub_menu.menu);
		FOREACH_NELEM(mi->sub_menu.menu_items, *sm_num_ptr, smi)
		{
			gtk_menu_item_set_label(GTK_MENU_ITEM(smi->menu_item), smi->label);
			gtk_menu_shell_append(GTK_MENU_SHELL(mi->sub_menu.menu), smi->menu_item);
		}
		gtk_menu_shell_append(GTK_MENU_SHELL(gui->menu_bar.menu), mi->menu_item); /* add each menu item to the menu_bar */
		sm_num_ptr++;
	}
}

// TODO: make recursive
static void free_gui_objs(struct gui *gui)
{
	FOREACH(gui->menu_bar.menu_items, mi)
	{
		delete[] mi->sub_menu.menu_items;
	}
	delete[] gui->menu_bar.menu_items;
} 

int gui_init_and_run(int *argc, char ***argv)
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
			{"Run"       , gtk_button_new(), G_CALLBACK(on_run), 0, 0},
			{"Pause"     , gtk_button_new(), G_CALLBACK(on_pause), 1, 0},
			{"GR Raster" , gtk_button_new(), G_CALLBACK(on_gr_raster), 3, 0},
			{"GO Raster" , gtk_button_new(), G_CALLBACK(on_go_raster), 3, 1},
			{"PC Window" , gtk_button_new(), G_CALLBACK(on_pc_window), 3, 2},
			{"Parameters", gtk_button_new(), G_CALLBACK(on_parameters), 3, 3}
		},
		.dcn_plast_button = {
			"DCN Plasticity",
			gtk_check_button_new(),
			G_CALLBACK(on_dcn_plast),
			1,
			4
		},
		.plast_radio_label = gtk_label_new("Plasticity"),
		.plasticity_radios = {
			{"Graded" , gtk_radio_button_new(NULL), G_CALLBACK(on_radio), 0, 3},
			{"Binary" , gtk_radio_button_new(NULL), G_CALLBACK(on_radio), 0, 4}, 
			{"Cascade", gtk_radio_button_new(NULL), G_CALLBACK(on_radio), 0, 5},
			{"Off"    , gtk_radio_button_new(NULL), G_CALLBACK(on_radio), 0, 6}
		},
		.menu_bar = {
			.menu = gtk_menu_bar_new(),
			.menu_items = new menu_item[NUM_SUB_MENUS]
			{
				{"File", gtk_menu_item_new(), 
					{gtk_menu_new(), new menu_item[NUM_FILE_MENU_ITEMS]
						{
							{"Save Sim", gtk_menu_item_new(), {}},
							{"Load Sim", gtk_menu_item_new(), {}},
						}
					}
				},
				{"Weights", gtk_menu_item_new(),
					{gtk_menu_new(), new menu_item[NUM_WEIGHTS_MENU_ITEMS]
						{
							{"Save Weights", gtk_menu_item_new(), {}},
							{"Load Weights", gtk_menu_item_new(), {}},
							{"Save MF-DN",   gtk_menu_item_new(), {}},
							{"Load MF-DN",   gtk_menu_item_new(), {}}
						}
					}
				},
				{"PSTH", gtk_menu_item_new(),
					{gtk_menu_new(), new menu_item[NUM_PSTH_MENU_ITEMS]
						{
							{"Save GR", gtk_menu_item_new(), {}},
							{"Save GO", gtk_menu_item_new(), {}},
							{"Save PC", gtk_menu_item_new(), {}},
							{"Save DN", gtk_menu_item_new(), {}},
							{"Save CF", gtk_menu_item_new(), {}},
							{"Save BC", gtk_menu_item_new(), {}},
							{"Save SC", gtk_menu_item_new(), {}},
							{"Save MF", gtk_menu_item_new(), {}}
						}
					}
				},
				{"Analysis", gtk_menu_item_new(),
					{gtk_menu_new(), new menu_item[NUM_ANALYSIS_MENU_ITEMS]
						{
							{"Analysis", gtk_menu_item_new(), {}}
						}
					}
				}
			}
		}
	}; 

	// set all attribs
	set_gui_window_attribs(&gui);
	
	GtkWidget *v_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
	gtk_container_add(GTK_CONTAINER(gui.window), v_box);

	set_gui_grid_attribs(&gui);
	set_gui_normal_button_attribs(&gui);
	set_gui_dcn_plast_button_attribs(&gui);
	set_gui_radio_button_attribs(&gui);
	set_gui_menu_attribs(&gui);

	gtk_box_pack_start(GTK_BOX(v_box), gui.menu_bar.menu, FALSE, TRUE, 0);
	gtk_box_pack_start(GTK_BOX(v_box), gui.grid, FALSE, TRUE, 0);
	

// ========================== purely declarative below =================================

	//GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

	//GtkWidget *v_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);

	//GtkWidget *menu_bar = gtk_menu_bar_new();

	//// file menu init
	//GtkWidget *file_menu_item = gtk_menu_item_new();
	//GtkWidget *file_menu = gtk_menu_new();

	//GtkWidget *open_file_menu_item = gtk_menu_item_new();
	//GtkWidget *open_file_menu = gtk_menu_new();

	//GtkWidget *save_file_menu_item = gtk_menu_item_new();
	//GtkWidget *save_file_menu = gtk_menu_new();

	//// weights menu init
	//GtkWidget *weights_menu_item = gtk_menu_item_new();
	//GtkWidget *weights_menu = gtk_menu_new();

	//GtkWidget *open_weights_menu_item = gtk_menu_item_new();
	//GtkWidget *open_weights_file_menu = gtk_menu_new();

	//GtkWidget *save_weights_menu_item = gtk_menu_item_new();
	//GtkWidget *save_weights_menu = gtk_menu_new();
	//
	//gtk_container_add(GTK_CONTAINER(window), v_box);

	//// set file menu relations
	//gtk_menu_item_set_label(GTK_MENU_ITEM(file_menu_item), "File");
	//gtk_menu_item_set_submenu(GTK_MENU_ITEM(file_menu_item), file_menu);

	//gtk_menu_item_set_label(GTK_MENU_ITEM(open_file_menu_item), "Open File");
	//gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), open_file_menu_item);

	//gtk_menu_item_set_label(GTK_MENU_ITEM(save_file_menu_item), "Save File");
	//gtk_menu_shell_append(GTK_MENU_SHELL(file_menu), save_file_menu_item);

	//gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), file_menu_item);

	//// set weight menu relations
	//gtk_menu_item_set_label(GTK_MENU_ITEM(weights_menu_item), "Weights");
	//gtk_menu_item_set_submenu(GTK_MENU_ITEM(weights_menu_item), weights_menu);

	//gtk_menu_item_set_label(GTK_MENU_ITEM(open_weights_menu_item), "Open weights");
	//gtk_menu_shell_append(GTK_MENU_SHELL(weights_menu), open_weights_menu_item);

	//gtk_menu_item_set_label(GTK_MENU_ITEM(save_weights_menu_item), "Save weights");
	//gtk_menu_shell_append(GTK_MENU_SHELL(weights_menu), save_weights_menu_item);

	//gtk_menu_shell_append(GTK_MENU_SHELL(menu_bar), weights_menu_item);

	//gtk_box_pack_start(GTK_BOX(v_box), menu_bar, FALSE, FALSE, 0);

	gtk_widget_show_all(gui.window);
	gtk_main();
	//free_gui_objs(&gui);
	return 0;
}

