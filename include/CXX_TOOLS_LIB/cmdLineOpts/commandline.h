#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_
#include <unistd.h> /* getopt */
#include <string>
#include <algorithm>


/*
 * function: getCmdOption
 *
 * args: char** begin 		  - ptr to first char array in a range
 * 		 char** end 		  - ptr to last char array in a range
 * 		 const string& option - the command line opt as a string that 
 * 		 						we want to find
 *
 * return val: the command opt as a char ptr upon success, else 0 (NULL)
 */
char* getCmdOption(char** begin, char** end, const std::string& option)
{
	char** itr = std::find(begin, end, option);
	if (itr != end && ++itr !=end)
	{
		return *itr;
	}
	return 0;
}

/*
 * function: cmdOptionExists
 * 
 * args: char** begin		  - ptr to first char array in a range
 * 		 char** end			  - ptr to last char array in a range
 * 		 const string& option - the command line opt as a string
 *
 * return val: a boolean, true if the option was given, false if not
 */
bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end; 
}


#endif /* COMMAND_LINE_H_ */

