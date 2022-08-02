#include <iostream>
#include <fstream>
#include <sstream>
#include "fileIO/experiment_file.h"

void parse_experiment_file(std::string trial_file_name, experiment &experiment)
{
	int i,j,k;
	int counter;
	std::string data[1000];
	int lineCounter = -1;
	experiment_info e;
	std::string previous;
	std::fstream inFile;     // buffer to open file

	e.numTrialTypes = 0;
	e.numBlockTypes = 0;
	inFile.open(trial_file_name, std::ios::in); //open file to perform read operation using file object

	if (inFile.is_open())  //checking whether the file is open
	{  
	   std::string fileline;    // a string to hold the line read from the file
		while(getline(inFile, fileline))  //read data from file object and put it into string.
		{ 
										  // *** best if I can make this a litle more forgiving
										  // in terms of capitalization, etc.  
			if (!fileline.empty())
			{
				// ignore first two lines for purposes of parsing experiment file
				if (fileline != "#Begin filetype experiment"
				 && fileline != "#VIS TUI"
				 && fileline != "#VIS GUI")
				{
					lineCounter++;
					data[lineCounter] = fileline;
				}
				int x = fileline.find_first_not_of(' ');
				if (isdigit(fileline[x]) == 0)
				{
					if (fileline == "#Begin Trial")
					{
						e.TrialBegin[e.numTrialTypes] = lineCounter;
						e.numTrialTypes++; 
					}
					else if (fileline == "#End Trial")
					{
						e.TrialEnd[e.numTrialTypes - 1] = lineCounter; 
					}
					else if (fileline == "#Begin Block")
					{
						e.BlockBegin[e.numBlockTypes] = lineCounter;
						e.numBlockTypes++; 
					}
					else if (fileline == "#End Block")
					{
						e.BlockEnd[e.numBlockTypes-1] = lineCounter; 
					}
					else if (fileline == "#Begin Session")
					{
						e.SessionBegin[e.numSessionTypes] = lineCounter;
						e.numSessionTypes++; 
					}
					else if (fileline == "#End Session")
					{
						e.SessionEnd[e.numSessionTypes - 1] = lineCounter; 
					}
					else if (fileline == "#Begin Experiment")
					{
						e.ExpBegin = lineCounter;
					}
					else if (fileline == "#End Experiment")
					{
						e.ExpEnd = lineCounter; 
					}
				}
				else
				{
					std::stringstream stream(fileline);
					stream >> x;
					data[lineCounter] = previous;
					for (int i2 = 0; i2 < x - 2; i2++)
					{
						lineCounter++;
						data[lineCounter] = previous;
					}
					std::cout << "found one that starts with digit: " << x << std::endl; 
					// figure out what the number is and add that number of previous filelines
				}
			}
			previous = fileline;
		}
		inFile.close(); //close the file object.   
	}

	// read trial info and store in trials arrays
	trial temptrials[100];
	
	for (i = 0; i < e.numTrialTypes; i++)  // The names of each trial go into TrialNames
	{    
		std::cout << e.TrialBegin[i] + 1 << std::endl;
		std::string full = data[e.TrialBegin[i] + 1];

		std::string comment_delim;
		std::stringstream stream(full); // parse the line

		stream >> temptrials[i].TrialName;
		stream >> comment_delim; // bunk extraction as we expect a '#' character here
		stream >> temptrials[i].CSuse;
		stream >> temptrials[i].CSonset;
		stream >> temptrials[i].CSoffset;
		stream >> temptrials[i].CSpercent;
		stream >> temptrials[i].USuse;
		stream >> temptrials[i].USonset;

		std::cout << temptrials[i].TrialName << " "
			 << temptrials[i].CSuse     << " " 
			 << temptrials[i].CSonset   << " " 
			 << temptrials[i].CSoffset  << " "
			 << temptrials[i].CSpercent << " "
			 << temptrials[i].USuse     << " " 
			 << temptrials[i].USonset   << std::endl;
	}
	
	// now find lists of sessions and blocks
	int ans;
	int countTrials;
	for (i = 0; i < e.numBlockTypes + 1; i++) // for each block find the list of trials
	{  
		e.Blocks[i][0] = data[e.BlockBegin[i] + 1];   // This is the name of the block
		counter = 1;
		countTrials = 0;
		for (j = e.BlockBegin[i] + 2; j < e.BlockEnd[i]; j++)  // this finds the trials listed for each block
		{   
			e.Blocks[i][counter] = data[j];
			countTrials++;
			for (k = 0; k < e.numTrialTypes; k++)  // this is the list of block as numeric "tags" so they can be indexed later
			{     
				ans = data[j].find(temptrials[k].TrialName); // bugfix: find in data, as data includes more spaces...
				std::cout << data[j] << " " << temptrials[k].TrialName << " " << ans << std::endl;
				if (ans >= 0)  // give it this tag
				{                                
					e.BlockTags[i][counter] = k;
					break;
				}
			}
			counter++;
		}
		e.BlockTrialCount[i] = countTrials;  // stores the number of trials in each block
	}

	std::cout << std::endl;

	for ( i = 0; i < e.numBlockTypes; i++)
	{
		counter = 1;
		std::cout << e.Blocks[i][0] << std::endl; 
		for ( j = e.BlockBegin[i] + 2; j < e.BlockEnd[i]; j++)
		{
		   std::cout << e.Blocks[i][counter] << "   " << e.BlockTags[i][counter] << std::endl;
			counter++;
		}
	}
	
	for (i = 0; i < e.numSessionTypes + 1; i++)  // for each session find the list of blocks
	{ 
		e.Sessions[i][0] = data[e.SessionBegin[i] + 1];
		counter = 1;
		countTrials = 0;  // reusing this variable from above
		for (j = e.SessionBegin[i] + 2; j < e.SessionEnd[i]; j++)
		{
			e.Sessions[i][counter] = data[j];
			countTrials++;
			for (k = 0; k < e.numBlockTypes; k++)
			{
				ans = data[j].find(e.Blocks[k][0]);
				std::cout << ans << "*" << data[j] << "  *" << e.Blocks[k][0] << std::endl;
				if (ans >= 0)
				{
					e.SessionTags[i][counter] = k;
					break;
				}
			}
			counter++;
		}
		e.SessionBlockCount[i] = countTrials;
	}

	std::cout << std::endl;

	for (i = 0; i < e.numSessionTypes; i++)
	{
		counter = 1;
		std::cout << e.Sessions[i][0] << std::endl; 
		for (j = e.SessionBegin[i] + 2; j < e.SessionEnd[i]; j++)
		{
		   std::cout << e.Sessions[i][counter] << "   " << e.SessionTags[i][counter] << std::endl;
			counter++;
		}
	}

	std::cout << std::endl;

	e.numSessions = 0;
	for (i = e.ExpBegin + 1; i < e.ExpEnd; i++)
	{
	   std::string l = data[i];
		e.SessionList[e.numSessions]=l;
		e.numSessions++;
	}

	for (i = 0; i < e.numSessions; i++)
	{
		for (j = 0;j < e.numSessionTypes; j++)
		{
			ans = e.SessionList[i].find(e.Sessions[j][0]);
			if (ans > 0)
			{
				e.ExpTags[i] = j;
				break;
			}
		}
		std::cout << e.SessionList[i] << "  " << e.ExpTags[i] << std::endl;
	}
	
	// now assemble the full list of trials from the list of sessions and blocks and trials 
	int numtrials = 0;
	for (int s = 0; s < e.numSessions; s++)
	{ 
		int currentSession = e.ExpTags[s];
		for (int b = 0; b < e.SessionBlockCount[currentSession]; b++)
		{
			int currentBlock = e.SessionTags[currentSession][b + 1];
			for (int t1 = 0; t1 < e.BlockTrialCount[currentBlock]; t1++)
			{
				int currentTrial            = e.BlockTags[currentBlock][t1 + 1];
				experiment.trials[numtrials].TrialName = temptrials[currentTrial].TrialName;
				experiment.trials[numtrials].CSuse     = temptrials[currentTrial].CSuse;
				experiment.trials[numtrials].CSonset   = temptrials[currentTrial].CSonset;
				experiment.trials[numtrials].CSoffset  = temptrials[currentTrial].CSoffset;
				experiment.trials[numtrials].CSpercent = temptrials[currentTrial].CSpercent;
				experiment.trials[numtrials].USuse     = temptrials[currentTrial].USuse;
				experiment.trials[numtrials].USonset   = temptrials[currentTrial].USonset;
				std::cout << numtrials << "  " << experiment.trials[numtrials].TrialName << std::endl;
				numtrials++;
			}
		}
	}
	experiment.num_trials = numtrials;
}

std::string trial_to_string(trial &ti)
{
	std::ostringstream out_string_buf;

	out_string_buf << "[ TrialName: ";
	out_string_buf << ti.TrialName;
	out_string_buf << " ]\n";

	out_string_buf << "[ CSuse: ";
	out_string_buf << std::to_string(ti.CSuse);
	out_string_buf << " ]\n";

	out_string_buf << "[ CSonset: ";
	out_string_buf << std::to_string(ti.CSonset);
	out_string_buf << " ]\n";

	out_string_buf << "[ CSoffset: ";
	out_string_buf << std::to_string(ti.CSoffset);
	out_string_buf << " ]\n";

	out_string_buf << "[ CSpercent: ";
	out_string_buf << std::to_string(ti.CSpercent);
	out_string_buf << " ]\n";

	out_string_buf << "[ USuse: ";
	out_string_buf << std::to_string(ti.USuse);
	out_string_buf << " ]\n";

	out_string_buf << "[ USonset: ";
	out_string_buf << std::to_string(ti.USonset);
	out_string_buf << " ]";
	
	return out_string_buf.str();
}

std::string experiment_to_string(experiment &ei)
{
	std::ostringstream out_string_buf;
	for (int i = 0; i < ei.num_trials; i++)
	{
		out_string_buf << trial_to_string(ei.trials[i]) << "\n";
	}
	return out_string_buf.str();
}

std::ostream & operator<<(std::ostream &os, trial &ti)
{
	return os << trial_to_string(ti);
}

std::ostream & operator<<(std::ostream &os, experiment &ei)
{
	return os << experiment_to_string(ei);
}

