#include "array_util.h"
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
	return assert(false, "Not implemented", __func__);
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
}

static void set_gui_grid_attribs(struct gui *gui)
{
	gtk_widget_set_name(gui->grid, "grid");
	gtk_container_add(GTK_CONTAINER(gui->window), gui->grid);
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

static void connect_gui_window_signals(struct gui *gui)
{
	gtk_widget_add_events(gui->window, GDK_DELETE);
	g_signal_connect(gui->window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
}

bool gui_init_and_run(int *argc, char ***argv)
{
	if (!gtk_init_check(argc, argv))
	{
		fprintf(stderr, "Could not initialize GTK\n");
		return false;
	}

	struct gui gui = {
		gtk_window_new(GTK_WINDOW_TOPLEVEL),
		gtk_grid_new(),
		{
			{"Run"       , gtk_button_new(), G_CALLBACK(on_run), 0, 0},
			{"Pause"     , gtk_button_new(), G_CALLBACK(on_pause), 1, 0},
			{"GR Raster" , gtk_button_new(), G_CALLBACK(on_gr_raster), 3, 0},
			{"GO Raster" , gtk_button_new(), G_CALLBACK(on_go_raster), 3, 1},
			{"PC Window" , gtk_button_new(), G_CALLBACK(on_pc_window), 3, 2},
			{"Parameters", gtk_button_new(), G_CALLBACK(on_parameters), 3, 3}
		},
		{
			"DCN Plasticity",
			gtk_check_button_new(),
			G_CALLBACK(on_dcn_plast),
			1,
			4
		},
		gtk_label_new("Plasticity"),
		{
			{"Graded" , gtk_radio_button_new(NULL), G_CALLBACK(on_radio), 0, 3},
			{"Binary" , gtk_radio_button_new(NULL), G_CALLBACK(on_radio), 0, 4}, 
			{"Cascade", gtk_radio_button_new(NULL), G_CALLBACK(on_radio), 0, 5},
			{"Off"    , gtk_radio_button_new(NULL), G_CALLBACK(on_radio), 0, 6}
		},
		g_menu_new(),
	
		//.sub_menus = {
		//	{g_menu_new()},
		//	{g_menu_new()},
		//	{g_menu_new()},
		//	{g_menu_new()},
		//}
	}; 

	// set all attribs
	set_gui_window_attribs(&gui);
	set_gui_grid_attribs(&gui);
	set_gui_normal_button_attribs(&gui);
	set_gui_dcn_plast_button_attribs(&gui);
	set_gui_radio_button_attribs(&gui);
	// TODO: initialize the submenus

	connect_gui_window_signals(&gui);

	gtk_widget_show_all(gui.window);
	gtk_main();
	return true;
}

