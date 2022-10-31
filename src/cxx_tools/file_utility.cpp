#include "file_utility.h"

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

void rawBytesRW(char *arr, unsigned long byteLen, bool read, std::fstream &file)
{
	if (read) file.read(arr, byteLen);
	else file.write(arr, byteLen);
}

