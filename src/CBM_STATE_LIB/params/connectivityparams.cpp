/*
 * activityparams.cpp
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

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

void ConnectivityParams::writeParams(std::string o_file)
{
// TODO: generate a build_file???
}

std::ostream &operator<<(std::ostream &os, ConnectivityParams &cp)
{
	return os << cp.toString();
}

