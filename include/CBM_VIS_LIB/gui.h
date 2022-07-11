#ifndef GUI_H_
#define GUI_H_

#include <stdio.h>
#include <gtk/gtk.h>

#define NUM_NORMAL_BUTTONS 8
#define NUM_PLASTICITY_RADIOS 4
#define NUM_SUB_MENU_ITEMS 4

#define NUM_FILE_MENU_ITEMS 4
#define NUM_WEIGHTS_MENU_ITEMS 4
#define NUM_PSTH_MENU_ITEMS 8
#define NUM_ANALYSIS_MENU_ITEMS 1

#define NUM_FILE_SUB_MENU_ITEMS 2

//const int num_item_per_sub_menu[NUM_SUB_MENUS] = {
//	NUM_FILE_MENU_ITEMS,
//	NUM_WEIGHTS_MENU_ITEMS,
//	NUM_PSTH_MENU_ITEMS,
//	NUM_ANALYSIS_MENU_ITEMS
//};

struct signal
{
	const gchar *signal;
	GCallback handler;
	gpointer data;
	bool swapped; /* may find this to be unnecessary... */
	//GdkEventMask mask;
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
	//GCallback handler;
	GtkWidget *menu_item;
	struct signal signal;
	struct menu sub_menu; /* single sub menu for every menu item */
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
};

int gui_init_and_run(int *argc, char ***argv);

#endif /* GUI_H_ */

