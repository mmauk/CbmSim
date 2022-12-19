#include <sstream>
#include "file_utility.h"

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

