/*
 * File: build_file.h
 * Author: Michael Mauk
 * Created on: circa 07/01/2022
 * Modified by: Sean Gallogly
 * Modified first on: circa 08/02/2022
 *
 * Description:
 *     This is the interface file for reading in an experiment file. It includes the 
 *     structures and constants necessary to parse an experiment file into a final
 *     data structure. 
 *
 */
#ifndef TRIAL_FILE_H_
#define TRIAL_FILE_H_
#include <string>

#define MAX_NUM_TRIALS 1500

struct trial /* data structure that holds info for up to 1000 trials */
{
	std::string TrialName;
	int CSuse;
	int CSonset;
	int CSoffset;
	int CSpercent;
	int USuse;
	int USonset;
};

struct experiment_info /* this struct is only used in the function parse_trials */
{
	int numTrialTypes;          // number of different trial types     
	int TrialBegin[32];         // line each trial list begins in the file
	int TrialEnd[32];           // line each trial list ends
	int numBlockTypes;          // number of different block types
	int BlockBegin[100];        // line each block begins
	int BlockEnd[100];          // and where it ends
	int numSessionTypes;        // number of different session types
	int numSessions;            // number of sessions listed in the experiment
	int SessionBegin[32];       // line each session begins
	int SessionEnd[32];         // and ends
	int ExpBegin;               // line the experiment list begins
	int ExpEnd;                 // and ends 
	std::string SessionList[100];    // list of sessions in the experiment
	std::string Sessions[32][32];    // list of blocks in each session type [session][list of blocks] [i][0] is name of block
	std::string Blocks[32][32];      // list of trials in each block
	int BlockTags[32][32];      // list of trial tags in each block
	int SessionTags[32][32];    // list of block tags in each session
	int ExpTags[100];           // list of session tags in each experiment
	int SessionBlockCount[32];  // number of blocks in each session
	int BlockTrialCount[32];    // number of trials in each block
};

typedef struct /* a structure which includes the number of trials, convenient for later processing */
{
	int num_trials;
	trial trials[MAX_NUM_TRIALS];
}
experiment;

/*
 * Description:
 *     resets all trial attribs to their value-initialized values and sets the number of expt
 *     trials to zero.
 *
 */
void reset_experiment(experiment &experiment);

/*
 * Description:
 *     the main parsing function. Prints out progress along the way. Takes in a trial file, as
 *     represented by trial_file_name and populates the experiment referenced by experiment.
 *
 */
void parse_experiment_file(std::string trial_file_name, experiment &experiment);

/*
 * Description:
 *     returns a string representation of the trial referred to by ti. The string has no
 *     EOL character at the very end.
 *
 */
std::string trial_to_string(trial &ti);

/*
 * Description:
 *     returns a string representation of the experiment referred to by ei. The string has no
 *     EOL character at the very end.
 *
 */
std::string experiment_to_string(experiment &ei);

/*
 * Description:
 *     overloads the stream insertion operator (<<) to insert a trial into the output stream
 *     referred to by os.
 *
 */
std::ostream & operator<<(std::ostream &os, trial &ti);

/*
 * Description:
 *     overloads the stream insertion operator (<<) to insert a experiment into the output stream
 *     referred to by os.
 *
 */
std::ostream & operator<<(std::ostream &os, experiment &ei);

#endif /* TRIAL_FILE_H_ */

