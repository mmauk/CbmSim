/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include <fstream>
#include <cstring>
#include "params/connectivityparams.h"

// FIXME: these constants and serial struct are in two places: should change location

const size_t key_int_val_size = sizeof(std::string) + sizeof(int);
const size_t key_float_val_size = sizeof(std::string) + sizeof(float);

template<size_t N, typename val_type>
struct serial
{
	char bin_arr[N];
	serial(std::string key, val_type val)
	{
		memcpy(bin_arr, &key, sizeof(std::string));
		memcpy(bin_arr + sizeof(std::string), &val, sizeof(val_type));
	}
	void read(std::fstream &in_param_buf) {in_param_buf.read(bin_arr, N);}
	void write(std::fstream &out_param_buf) {out_param_buf.write(bin_arr, N);}
};

ConnectivityParams::ConnectivityParams() {}

ConnectivityParams::ConnectivityParams(parsed_file &p_file)
{
	for (auto iter = p_file.parsed_sections["connectivity"].param_map.begin();
			  iter != p_file.parsed_sections["connectivity"].param_map.end();
			  iter++)
	{
		if (iter->second.type_name == "int")
		{
			int_params[iter->first] = std::stoi(iter->second.value);
		}
		else if (iter->second.type_name == "float")
		{
			float_params[iter->first] = std::stof(iter->second.value);
		}
	}
}

std::string ConnectivityParams::toString()
{
	std::string out_string = "[\n";
	for (auto iter = int_params.begin(); iter != int_params.end(); iter++)
	{
		out_string += "[ '" + iter->first + "', '"
							+ std::to_string(iter->second)
							+ "' ]\n";
	}

	for (auto iter = float_params.begin(); iter != float_params.end(); iter++)
	{
		out_string += "[ '" + iter->first + "', '"
							+ std::to_string(iter->second)
							+ "' ]\n";
	}
	out_string += "]";
	return out_string;
}

void ConnectivityParams::readParams(std::fstream &inParamBuf)
{
	// TODO: finish writing
}

void ConnectivityParams::writeParams(std::fstream &outParamBuf)
{
	std::cout << "[INFO]: Writing connectivity params to file..." << std::endl;
	for (auto iter = int_params.begin(); iter != int_params.end(); iter++)
	{
		serial<key_int_val_size, int> data(iter->first, iter->second);
		data.write(outParamBuf);
	}
	for (auto iter = float_params.begin(); iter != float_params.end(); iter++)
	{
		serial<key_float_val_size, float>data(iter->first, iter->second);
		data.write(outParamBuf);
	}
	std::cout << "[INFO]: Finished writing connectivity params to file." << std::endl;
}

std::ostream &operator<<(std::ostream &os, ConnectivityParams &cp)
{
	return os << cp.toString();
}

