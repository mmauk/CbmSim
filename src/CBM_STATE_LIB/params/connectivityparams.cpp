/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#include "fileIO/serialize.h"
#include "params/connectivityparams.h"

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
	// TODO: need addtl checks on whether param maps are initialized or not
	if (!(int_params.size() == 0 && float_params.size() == 0))
	{
		int_params.clear();
		float_params.clear();
	}
	std::cout << "[INFO]: Reading connectivity params from file..." << std::endl;
	unserialize_map_from_file<std::string, int>(int_params, inParamBuf);
	unserialize_map_from_file<std::string, float>(float_params, inParamBuf);
	std::cout << "[INFO]: Finished reading connectivity params from file." << std::endl;
}

void ConnectivityParams::writeParams(std::fstream &outParamBuf)
{
	std::cout << "[INFO]: Writing connectivity params to file..." << std::endl;
	serialize_map_to_file<std::string, float>(float_params, outParamBuf);
	serialize_map_to_file<std::string, int>(int_params, outParamBuf);
	std::cout << "[INFO]: Finished writing connectivity params to file." << std::endl;
}

std::ostream &operator<<(std::ostream &os, ConnectivityParams &cp)
{
	return os << cp.toString();
}

