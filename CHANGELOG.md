 # 0.1.0 (01-18-2023)

This is a moderately-sized release with changes mainly targeting the automation of output file production,
saving granule rasters and weights every trial, and more.

Here is a non-exhaustive summary of the changes made.

Breaking Changes:

- output 2D arrays (rasters and psths) are the transpose from what they were prior.
  ([3eafb41](https://github.com/gawdSim/CbmSim/commit/3eafb41), [644204e](https://github.com/gawdSim/CbmSim/commit/644204e))   
- Raster and PSTH array saving is now conducted with lambda functions ([a3d616c](https://github.com/gawdSim/CbmSim/commit/a3d616c),
  [7d3cc14](https://github.com/gawdSim/CbmSim/commit/7d3cc14))
- added logging framework, all messages to console go through logger now ([e564bae](https://github.com/gawdSim/CbmSim/commit/e564bae),
  [2fbf139](https://github.com/gawdSim/CbmSim/commit/2fbf139))
- cell connectivity parameters are in build files, and activity parameters are split between build and session files 
  ([#2](https://github.com/gawdSim/CbmSim/pull/2), [#5](https://github.com/gawdSim/CbmSim/pull/5))

Features:

- output filenames are automatically generated from output commandline arguments and organized in required output option
  ([2f6c3ff](https://github.com/gawdSim/CbmSim/commit/2f6c3ff), [58daf5e](https://github.com/gawdSim/CbmSim/commit/58daf5e),
   [d3a748f](https://github.com/gawdSim/CbmSim/commit/d3a748f))
- gui includes ability to turn on/off cerebellar cortex or deep cerebellar nucleus plasticity (types of cortex plasticity
  are templated, but for now only graded plasticity is implemented) ([964eeb9](https://github.com/gawdSim/CbmSim/commit/964eeb9),
  [da594ea](https://github.com/gawdSim/CbmSim/commit/da594ea), [a3d616c](https://github.com/gawdSim/CbmSim/commit/a3d616c))
- weights can be loaded or saved at the end of a trial in the gui (see `gui.cpp:on_load_pfpc_weights`, `gui.cpp:on_load_mfdcn_weights`,
  and `gui.cpp:on_save_file`)
- session file can be loaded and saved in gui (see `gui.cpp:on_load_session_file`, `gui.cpp:on_save_file`)
- simulation file can be loaded and saved in gui (see `gui.cpp:on_load_sim_file`, `gui.cpp:on_save_file`)
- gui includes views into golgi and granule rasters at the end of a trial ([c3f4ef3](https://github.com/gawdSim/CbmSim/commit/c3f4ef3),
  [1579347](https://github.com/gawdSim/CbmSim/commit/1579347))
- gui includes views on purkinje, deep nucleus, and inferior olive cell voltages and spikes after a trial
  ([50d34d1](https://github.com/gawdSim/CbmSim/commit/50d34d1))
- gui menubar includes tuning window pop-up, allowing synaptic weights to be altered during runtime
  ([0379e62](https://github.com/gawdSim/CbmSim/commit/0379e62))
- gui menubar includes firing rate window, showing CS and non-CS mean and median firing rates of all cell populations
  ([0379e62](https://github.com/gawdSim/CbmSim/commit/0379e62))
- gui includes output directory creation workflow ([524ba02](https://github.com/gawdSim/CbmSim/commit/524ba02),
  [af98846](https://github.com/gawdSim/CbmSim/commit/af98846), [7ea0ddb](https://github.com/gawdSim/CbmSim/commit/7ea0ddb))
- clicking on a given cell or synapse type under save context menu automatically names saved file
  ([87172bf](https://github.com/gawdSim/CbmSim/commit/87172bf))
- user can test whether any changes they make alters the commandline behaviour with scripts/tests/sh
  ([66d2f84](https://github.com/gawdSim/CbmSim/commit/66d2f84))
- granule cell rasters and weights are now saved once every trial ([#9](https://github.com/gawdSim/CbmSim/pull/9),
  [#12](https://github.com/gawdSim/CbmSim/pull/12))
- a pre-collection time period of 2000 simulation steps (2000ms) was added ([#10](https://github.com/gawdSim/CbmSim/pull/10),
  [#12](https://github.com/gawdSim/CbmSim/pull/12))

Bug-Fixes:

- go-raster window in GUI accurately updates per trial ([9cfa1bb](https://github.com/gawdSim/CbmSim/commit/9cfa1bb),
  [90f931d](https://github.com/gawdSim/CbmSim/commit/90f931d))
- various commandline edge case fixes ([584d5be](https://github.com/gawdSim/CbmSim/commit/584d5be),
  [2fdb100](https://github.com/gawdSim/CbmSim/commit/2fdb100), [effeb16](https://github.com/gawdSim/CbmSim/commit/effeb16))
- integer overflow fix when creating very large 2d arrays ([644204e](https://github.com/gawdSim/CbmSim/commit/644204e))

Internal Changes:

- bc cell activity is now computed within mzone.cpp ([fc72ef7](https://github.com/gawdSim/CbmSim/commit/fc72ef7),
  [1fd1cb5](https://github.com/gawdSim/CbmSim/commit/1fd1cb5))
- build and session files are parsed in a multi-step (tokenize->lex->parse) process (see `file_parse.h` and `file_parse.cpp`)
- commandline is parsed in a multistep (char** -> vector<string> -> parsed_commandline) process, then validated separately
  (see `commandline.h` and `commandline.cpp`)
- all control over the simulation and various modules is conducted in `control.h` and `control.cpp`. `set_sim` module was deprecated
- `main.cpp` is vastly simplified and calls only a few functions from `control` module
