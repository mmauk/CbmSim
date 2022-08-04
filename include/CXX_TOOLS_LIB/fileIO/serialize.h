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
#include <fstream>
#include <unordered_map>
#include <string>
#include <cstring>

template<typename key_t, typename val_t>
void serialize_map_to_file(std::unordered_map<key_t, val_t> &map, std::fstream &file_buf)
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

template<typename key_t, typename val_t>
void unserialize_map_from_file(std::unordered_map<key_t, val_t> &map, std::fstream &file_buf)
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

