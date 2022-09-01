#ifndef GUI_H_
#define GUI_H_

#include <stdio.h>
#include <gtk/gtk.h>
#include "control.h"

/* main window constants */
#define MAIN_WINDOW_WIDTH 300
#define MAIN_WINDOW_HEIGHT 300

#define NUM_NORMAL_BUTTONS 6
#define NUM_PLASTICITY_RADIOS 4

#define NUM_SUB_MENU_ITEMS 5

#define NUM_FILE_MENU_ITEMS 4
#define NUM_WEIGHTS_MENU_ITEMS 4
#define NUM_PSTH_MENU_ITEMS 8
#define NUM_ANALYSIS_MENU_ITEMS 1
#define NUM_TUNING_MENU_ITEMS 1
#define NUM_FILE_SUB_MENU_ITEMS 2

/* analysis window constants */
#define NUM_FIRING_RATE_TABLE_HEADERS 5
#define NUM_CELL_TYPES 8
#define NUM_FIRING_RATE_DATA_COLS 5
#define DEFAULT_FIRING_RATE_WINDOW_WIDTH 600
#define DEFAULT_FIRING_RATE_WINDOW_HEIGHT 160

/* tuning window constants */
#define NUM_TUNING_BUTTONS 14
#define DEFAULT_TUNING_WINDOW_WIDTH 600
#define DEFAULT_TUNING_WINDOW_HEIGHT 160

/* raster window constants */
#define DEFAULT_RASTER_WINDOW_WIDTH 540
#define DEFAULT_RASTER_WINDOW_HEIGHT 960

/* pfpc weight window constants */
#define DEFAULT_PFPC_WINDOW_WIDTH 1024
#define DEFAULT_PFPC_WINDOW_HEIGHT 700

/* pc window constants */
#define DEFAULT_PC_WINDOW_WIDTH 1024
#define DEFAULT_PC_WINDOW_HEIGHT 1000

/* file dialog constants */
#define DEFAULT_STATE_FILE_NAME "connectivity_state"
#define DEFAULT_PFPC_WEIGHT_NAME "pfpc_weights.bin"
#define DEFAULT_MFDCN_WEIGHT_NAME "mfdcn_weights.bin"
#define DEFAULT_GR_PSTH_FILE_NAME "gr_psth.bin"
#define DEFAULT_GO_PSTH_FILE_NAME "go_psth.bin"
#define DEFAULT_PC_PSTH_FILE_NAME "pc_psth.bin"
#define DEFAULT_NC_PSTH_FILE_NAME "dcn_psth.bin"
#define DEFAULT_IO_PSTH_FILE_NAME "cf_psth.bin"
#define DEFAULT_BC_PSTH_FILE_NAME "bc_psth.bin"
#define DEFAULT_SC_PSTH_FILE_NAME "sc_psth.bin"
#define DEFAULT_MF_PSTH_FILE_NAME "mf_psth.bin"

struct signal
{
	const gchar *signal;
	GCallback handler;
	gpointer data;
	bool swapped; /* may find this to be unnecessary... */
};

struct button
{
	const gchar *label;
	GtkWidget *widget;
	gint col;
	gint row;
	struct signal signal;
};

struct menu
{
	GtkWidget *menu;
	int num_menu_items;
	struct menu_item *menu_items;
};

struct menu_item
{
	const gchar *label;
	GtkWidget *menu_item;
	struct signal signal;
	struct menu sub_menu; /* single sub menu for every menu item */
};

struct tuning_button_label
{
	GtkWidget *label;
	const gchar *string;
	gint col;
	gint row;
};

struct tuning_button
{
	GtkAdjustment *adjustment;
	GtkWidget *widget;
	gint col;
	gint row;
	struct tuning_button_label label;
	struct signal signal;

};

struct tuning_window
{
	GtkWidget *window;
	GtkWidget *grid;
	struct tuning_button tuning_buttons[NUM_TUNING_BUTTONS];
};

// TODO: combine with tuning button label
struct firing_rate_label
{
	GtkWidget *label;
	const gchar *string;
	gint col;
	gint row;
};

struct firing_rate_window
{
	GtkWidget *window;
	GtkWidget *grid;
	struct firing_rate_label headers[NUM_FIRING_RATE_TABLE_HEADERS];
	struct firing_rate_label cell_labels[NUM_CELL_TYPES][NUM_FIRING_RATE_DATA_COLS];
};

struct gui
{
	GtkWidget *window;
	GtkWidget *grid;
	struct button normal_buttons[NUM_NORMAL_BUTTONS];
	struct button dcn_plast_button;
	GtkWidget *plast_radio_label;
	struct button plasticity_radios[NUM_PLASTICITY_RADIOS];
	struct menu menu_bar;
	struct firing_rate_window frw;
	Control *ctrl_ptr;
	
};

gboolean firing_rates_win_visible(struct gui *gui);
gboolean update_fr_labels(struct gui *gui); /* here because use in control */
int gui_init_and_run(int *argc, char ***argv, Control *control);

#endif /* GUI_H_ */

