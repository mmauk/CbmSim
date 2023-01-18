/*
 * file: serialize.h
 * author: Sean Gallogly
 * created on: 07/28/2022
 * 
 * Description:
 *     Provides functions for serializing data to and from file. Serializiation is
 *     the process of sending and retrieving binary data to and from file via a 
 *     data structure of the programmer's choice. For now we implement a 
 *     serialization interface for std::maps, but others can be added easily in the
 *     future by utilizing the basic concepts which can be found in the implementation
 *     file.
 *
 */
#ifndef FILE_UTILITY_H_
#define FILE_UTILITY_H_

#include <algorithm>
#include <string>
#include <cstring>
#include <fstream>
#include <map>
#include <ctime>

const std::string DEFAULT_DATE_FORMAT = "(%m_%d_%Y)";
const std::string DEFAULT_TIME_FORMAT = "(%H:%M:%S)";
const std::string DEFAULT_LOCALE_FORMAT = "(%z %Z)";

const std::string TXT_EXT = ".txt";
const std::string BIN_EXT = ".bin";
const std::string SIM_EXT = ".sim";

/* the debug executable is contained within {PROJECT_ROOT}build/debug,
 * so the data folder is two directories up, rather than one directory
 * up for the release executable
 */
#ifdef DEBUG
	const std::string INPUT_DATA_PATH  = "../../data/inputs/";
	const std::string OUTPUT_DATA_PATH = "../../data/outputs/";
#else
	const std::string INPUT_DATA_PATH  = "../data/inputs/";
	const std::string OUTPUT_DATA_PATH = "../data/outputs/";
#endif

/*
 * Description:
 *     takes a reference to a string and a character and trims the character off of
 *     the string from the left. The trimming is done in-place.
 */
inline void ltrim(std::string &s, unsigned char ch)
{
	s.erase(s.begin(), std::find_if(s.begin(), s.end(), [ch](unsigned char compare)
	{
		return compare != ch;
	}));
}

/*
 * Description:
 *     takes a reference to a string and a character and trims the character off of
 *     the string from the right. The trimming is done in-place.
 */
inline void rtrim(std::string &s, unsigned char ch) {
	s.erase(std::find_if(s.rbegin(), s.rend(), [ch](unsigned char compare) {
		return compare != ch;
	}).base(), s.end());
}

/*
 * Description:
 *     takes in a string representation of a file-path and returns the filename,
 *     stripped of its path. If just the filename is given, it is returned as-is.
 *     For example, if the file-path "/home/user/a_file" is given, then 
 *     "a_file" is returned as a std::string.
 */
std::string strip_file_path(std::string full_file_path);

/*
 * Description:
 *     takes in a string representation of a file-path and returns the filename,
 *     stripped of its path and its extension. For example, if the file-path
 *     "/home/user/some_info.txt" is given, then "some_info" is returned as a std::string.
 */
std::string get_file_basename(std::string full_file_path);

/*
 * Description:
 *     takes in a format string and returns the current time. The function is as flexible
 *     as the time format allows, ie the time can be represented in seconds, day/month/year,
 *     hour/minute/second, and so on. 
 *
 *     For a detailed description of valid format strings, see en.cppreference.com/w/cpp/io/manip/put_time 
 */
std::string get_current_time_as_string(std::string time_format);

/*
 * Description:
 *     takes in a directory to search, a filename, and searches recursively for the given filename.
 *     If a match is found, the full path of the filename is saved in the reference resolved_fullpath
 *     and the function returns true. Otherwise, the resolved_fullpath is set to the empty string and the
 *     function returns false.
 */
bool file_exists(std::string dir_to_search, std::string &filename, std::string &resolved_fullpath);

/*
 * Description:
 *     takes in a pointer (arr) and a given number of bytes (byteLen) and reads from or writes to
 *     the fstream (file) if the given bool (read) is true or false respectively.
 */
void rawBytesRW(char *arr, unsigned long byteLen, bool read, std::fstream &file);

/*
 * Description:
 *     takes in a map with template key type key_t and value type val_t and writes
 *     the underlying bytes to file. The order is dictated by the implementation of std::map.
 *     the first sizeof(size_t) bytes of the written file represent the number of key-value pairs
 *     within the map.
 */
template<typename key_t, typename val_t>
void serialize_map_to_file(std::map<key_t, val_t> &map, std::fstream &file_buf)
{
	size_t map_size = map.size();
	char *map_size_bytes = (char *)calloc(1, sizeof(size_t));

	memcpy(map_size_bytes, &map_size, sizeof(size_t));
	file_buf.write(map_size_bytes, sizeof(size_t));

	// serialize key value params in the order that they are given within the map
	for (auto iter = map.begin(); iter != map.end(); iter++)
	{
		size_t key_len_as_size_t = iter->first.length() + 1;

		char *key_len = (char *)calloc(1, sizeof(size_t));
		memcpy(key_len, &key_len_as_size_t, sizeof(size_t));

		char *key = (char *)calloc(key_len_as_size_t, sizeof(char));
		strcpy(key, iter->first.c_str());
		
		char *val = (char *)calloc(1, sizeof(val_t));
		memcpy(val, &(iter->second), sizeof(val_t));

		file_buf.write(key_len, sizeof(size_t));
		file_buf.write(key, key_len_as_size_t * sizeof(char));
		file_buf.write(val, sizeof(val_t));
	
		free(key_len);
		free(key);
		free(val);
	}
	free(map_size_bytes);
}

/*
 * Description:
 *     takes in a file and reads its contents into a map with template key type key_t
 *     and value type val_t. The order is dictated by the implementation of std::map.
 *     it is assumed that the data contained within the file is formatted as 
 *     key_t key and val_t val pairs, and can be cast into the types of key_t
 *     and val_t respectively. It is also assumed that the first sizeof(size_t) bytes
 *     of the file can be interpreted as the number of key-pair values contained
 *     within the file.
 */
template<typename key_t, typename val_t>
void unserialize_map_from_file(std::map<key_t, val_t> &map, std::fstream &file_buf)
{
	size_t int_params_size;
	char *int_params_size_arr = (char *)calloc(1, sizeof(size_t));

	file_buf.read(int_params_size_arr, sizeof(size_t));
	memcpy(&int_params_size, int_params_size_arr, sizeof(size_t));

	for (int i = 0; i < int_params_size; i++)
	{
		char *key_len_bytes = (char *)calloc(1, sizeof(size_t));
		file_buf.read(key_len_bytes, sizeof(size_t));

		size_t key_len;
		memcpy(&key_len, key_len_bytes, sizeof(size_t));

		char *key_bytes = (char *)calloc(key_len, sizeof(char));
		file_buf.read(key_bytes, key_len);

		char *val_bytes = (char *)calloc(1, sizeof(val_t));
		file_buf.read(val_bytes, sizeof(val_t));
		
		val_t val;
		memcpy(&val, val_bytes, sizeof(val_t));

		map[std::string(key_bytes)] = val;
		
		free(key_len_bytes);
		free(key_bytes);
		free(val_bytes);
	}
	free(int_params_size_arr);
}

#endif /* FILE_UTILITY_H_ */

