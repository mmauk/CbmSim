#include <sstream>
#include <unistd.h>
#include <dirent.h>

#include "file_utility.h"
#include "logger.h"

std::string strip_file_path(std::string full_file_path)
{
	size_t sep = full_file_path.find_last_of("\\/");
	return (sep != std::string::npos) ? full_file_path.substr(sep + 1, full_file_path.size() - sep - 1) : "";
}

std::string get_file_basename(std::string full_file_path)
{
	size_t sep = full_file_path.find_last_of("\\/");
	if (sep != std::string::npos)
		full_file_path = full_file_path.substr(sep + 1, full_file_path.size() - sep - 1);
	size_t dot = full_file_path.find_last_of(".");
	if (dot != std::string::npos) std::string name = full_file_path.substr(0, dot);
	else std::string name = full_file_path;
	return (dot != std::string::npos) ? full_file_path.substr(0, dot) : full_file_path;
}

std::string get_current_time_as_string()
{
	std::stringstream time_stream;

	std::time_t t = std::time(0);
	std::tm *now = std::localtime(&t);

	if (now->tm_mon + 1 < 10) time_stream << "0";
	time_stream << (now->tm_mon+1);
	if (now->tm_mday < 10) time_stream << "0";
	time_stream << (now->tm_mday) << (now->tm_year + 1900);
	return time_stream.str();
}

void rawBytesRW(char *arr, size_t byteLen, bool read, std::fstream &file)
{
	if (read) file.read(arr, byteLen);
	else file.write(arr, byteLen);
}

bool file_exists_recurse(std::string dir_to_search, std::string &filename)
{
	struct dirent *dp;
	DIR *dir = opendir(dir_to_search.c_str());
	if (!dir)
		return (strip_file_path(dir_to_search) == filename) ? true : false;

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
			accum = accum || file_exists_recurse(new_path, filename);
			if (accum) break;
		}
	}
	return accum;
}

bool file_exists(std::string dir_to_search, std::string &filename)
{
	char cwd[PATH_MAX];
	if (getcwd(cwd, sizeof(cwd)) == NULL)
	{
		LOG_FATAL("Couldn't determine current working directory. Exiting");
		exit(1);
	}
	return file_exists_recurse(std::string(cwd)+"/"+dir_to_search, filename);
}

