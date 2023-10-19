/** @file main.cpp
 *  @brief this is the main entry point to the program.
 *
 *  It parses commandline arguments using functions from commandline.h,
 *  and it either builds a simulation or runs trials by calling methods
 *  from control.h.
 *
 *  @author Sean Gallogly (sean.gallo@austin.utexas.edu)
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <time.h>

#include "commandline.h"
#include "control.h"
#include "file_parse.h"
#include "gui.h"
#include "logger.h"

int main(int argc, char **argv) {
  logger_initConsoleLogger(stderr);
// for now, set the log level dependent on whether
// we are compiling for debug target or release target
#ifdef DEBUG
  logger_setLevel(LogLevel_DEBUG);
#else
  logger_setLevel(LogLevel_INFO);
#endif
  parsed_commandline p_cl = {};
  parse_and_validate_parsed_commandline(&argc, &argv, p_cl);

  Control *control = new Control(p_cl);
  int exit_status = 0;

  omp_set_num_threads(1); /* for 4 gpus, 8 is the sweet spot. Unsure for 2. */

  if (p_cl.vis_mode == "TUI") {
    if (!p_cl.build_file.empty()) {
      control->build_sim();
      control->save_sim_to_file();
    } else if (!p_cl.session_file.empty()) {
      control->runSession(NULL); // saving is done at the end of runSession.
    }
    control->save_con_arrs(); // save conn arrs regardless route took
  } else if (p_cl.vis_mode == "GUI") {
    exit_status = gui_init_and_run(&argc, &argv, control);
  }
  delete control;
  return exit_status;
}
