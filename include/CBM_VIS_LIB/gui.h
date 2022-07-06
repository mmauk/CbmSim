#ifndef GUI_H_
#define GUI_H_

#include <stdio.h>
#include <gtk/gtk.h>

struct signal
{
	const gchar *signal;
	GCallback handler;
	GdkEventMask mask;
};

struct button
{
	gchar *label;
	GtkWidget *widget;
	GCallback handler;
	gint col;
	gint row;
};

struct menu
{
	GMenu *menu;
};

struct gui
{
	GtkWidget *window;
	GtkWidget *grid;

	struct button normal_buttons[6];

	struct button dcn_plast_button;

	GtkWidget *plast_radio_label;

	struct button plasticity_radios[4];

	GMenu *main_menu_bar;

	//struct menu sub_menus[4];
		//GMenu *fileMenu,
		//GMenu *weightMenu,
		//GMenu *psthMenu,
		//GMenu analysisMenu
};

bool gui_init_and_run(int *argc, char ***argv);

#endif /* GUI_H_ */

