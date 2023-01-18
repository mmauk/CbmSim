#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "file_utility.h"
#include "logger.h"

/*
 * Implementation Notes:
 *     uses STD library functions to first find the last of the string "/"
 *     in the input string. If the separator isn't found in the input string,
 *     the empty string is returned. Otherwise, the substring representing just the filename
 *     is returned.
 *
 */
std::string strip_file_path(std::string full_file_path)
{
	size_t sep = full_file_path.find_last_of("\\/");
	return (sep != std::string::npos) ? full_file_path.substr(sep + 1, full_file_path.size() - sep - 1) : "";
}

/*
 * Implementation Notes:
 *    Applies the same logic as strip_file_path, except the absence of the "/" separator
 *    is allowed. However, if there is no extension (as evidenced by the presence of the "." substring),
 *    then the filename as-given as input is returned.
 */
std::string get_file_basename(std::string full_file_path)
{
	size_t sep = full_file_path.find_last_of("\\/");
	if (sep != std::string::npos)
		full_file_path = full_file_path.substr(sep + 1, full_file_path.size() - sep - 1);
	size_t dot = full_file_path.find_last_of(".");
	return (dot != std::string::npos) ? full_file_path.substr(0, dot) : full_file_path;
}

/*
 * Implementation Notes:
 *     Uses a stringstream and the library function std::put_time to convert the time obtained
 *     from the calls to std::time and std::localtime into the requested format.
 */
std::string get_current_time_as_string(std::string time_format)
{
	std::time_t t = std::time(0);
	std::tm *now = std::localtime(&t);
	std::stringstream formatter;
	formatter << std::put_time(now, time_format.c_str());
	return formatter.str();
}

void rawBytesRW(char *arr, size_t byteLen, bool read, std::fstream &file)
{
	if (read) file.read(arr, byteLen);
	else file.write(arr, byteLen);
}

/*
 * Implementation Notes
 *     is meant to be called from the main function "file_exists". uses the posix library function opendir
 *     and the returned pointer to determine whether the path can be opened as a directory. If not,
 *     then the directory to search is expected to be a file, and its name is compared to the search string (filename).
 *     True or false is returned upon a match or no match, respectively.
 *
 *     Within the recursive section, a boolean accumulator is defined such that in the following loop over the 
 *     children of the current directory, the accumulator remains false until a match is found. the accumulator
 *     is checked for its truth-value such that we break out without performing a full recursive search beyond the
 *     first match.
 */
bool file_exists_recurse(std::string dir_to_search, std::string &filename, std::string &resolved_fullpath)
{
	struct dirent *dp;
	DIR *dir = opendir(dir_to_search.c_str());
	if (!dir)
	{
		if (strip_file_path(dir_to_search) == filename)
		{
			resolved_fullpath = dir_to_search;
			return true;
		}
		return false;
	}

	bool accum = false;
	while ((dp = readdir(dir)) != NULL)
	{
		if (strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0)
		{
			std::string new_path;
			if (dir_to_search[dir_to_search.length()-1] == '/')
				new_path = dir_to_search + std::string(dp->d_name);
			else
				new_path = dir_to_search + "/" + std::string(dp->d_name);
			accum = accum || file_exists_recurse(new_path, filename, resolved_fullpath);
			if (accum) break;
		}
	}
	return accum;
}

/*
 * Implementation Notes
 *     is entry point for recursive directory search for the search string (filename).
 *     The directory to search is converted into a full path via the posix library function
 *     realpath. The recursive helper function is then called on the returned absolute path.
 *
 */
bool file_exists(std::string dir_to_search, std::string &filename, std::string &resolved_fullpath)
{
	// PATH_MAX set to 4096
	char search_path_real[PATH_MAX];
	realpath(dir_to_search.c_str(), search_path_real);
	return file_exists_recurse(std::string(search_path_real), filename, resolved_fullpath);
}

