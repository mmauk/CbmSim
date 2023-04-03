/*
 * File: info_file.cpp
 * Author: Sean Gallogly
 * Created on: 12/16/2022
 * 
 * Description:
 *     This file implements the prototypes given in info_file.h. In doing so it also defines
 *     utility functions that are private only to this file.
 */
#include <unistd.h> // geteuid
#include <pwd.h> // for getpwuid 
#include <ctime> 
#include <sstream> // string stream
#include <iomanip> // put_time

#include "logger.h"
#include "info_file.h"

void cp_to_info_file_data(parsed_commandline &p_cl, parsed_sess_file &s_file, info_file_data &if_data)
{
	cp_parsed_commandline(p_cl, if_data.p_cl);
	cp_parsed_sess_file(s_file, if_data.s_file);
}

/*
 * Description:
 *     Obtains the username of the user who is running the generated executable
 *     
 *     Uses the unix function 'geteuid' to query the effective user ID of the calling process.    
 */
static const char *get_user_name()
{
	uid_t uid = geteuid();
	struct passwd *pw = getpwuid(uid);
	if (pw)
	{
		return pw->pw_name;
	}
	
	return "";
}

/*
 * Implementation Notes:
 *     uses a string stream and std::put_time to format the current time according to the default
 *     formats. The default formats include extra wrapping-parentheses due to the ease with which
 *     this form can be handled when actually writing to the info file. This choice may be subject
 *     to change.
 *
 *     The only failure state for this function is if the username could not be obtained from
 *     get_user_name, for whatever reason.
 *
 */
void set_info_file_str_props(enum when when, info_file_data &if_data)
{
	std::time_t t = std::time(0);
	std::tm *now = std::localtime(&t);
	std::stringstream formatter;
	if (when == BEFORE_RUN)
	{
		formatter << std::put_time(now, DEFAULT_DATE_FORMAT.c_str());
		if_data.start_date = formatter.str();
		ltrim(if_data.start_date, '(');
		rtrim(if_data.start_date, ')');
		std::stringstream().swap(formatter);
		formatter << std::put_time(now, DEFAULT_LOCALE_FORMAT.c_str());
		if_data.locale = formatter.str();
		std::stringstream().swap(formatter);
		formatter << std::put_time(now, DEFAULT_TIME_FORMAT.c_str());
		if_data.start_time = formatter.str();
		ltrim(if_data.start_time, '(');
		rtrim(if_data.start_time, ')');
		std::stringstream().swap(formatter);
		if_data.sim_version = SIM_VERSION;
		std::string user_name = std::string(get_user_name());
		if (user_name.empty())
		{
			LOG_ERROR("Could not obtain username when setting information file properties."); 
			LOG_ERROR("Username will be omitted from generated info file");
			return;
		}
		if_data.username = user_name;
	}
	else
	{
		formatter << std::put_time(now, DEFAULT_DATE_FORMAT.c_str());
		if_data.end_date = formatter.str();
		ltrim(if_data.end_date, '(');
		rtrim(if_data.end_date, ')');
		std::stringstream().swap(formatter);
		formatter << std::put_time(now, DEFAULT_TIME_FORMAT.c_str());
		if_data.end_time = formatter.str();
		ltrim(if_data.end_time, '(');
		rtrim(if_data.end_time, ')');
		std::stringstream().swap(formatter);
	}
}

/*
 * Implementation Notes:
 *     essentially implements the most brute-force search for the largest element in a linear data
 *     structure. Unfortunately, this function has O(N) time complexity, but the input map is not 
 *     expected to exceed 10 elements, so the penalties at higher N will most likely never be seen.
 */
uint32_t get_max_key_len(std::map<std::string, std::string> &map)
{
	uint32_t max_len = 0;
	uint32_t test_len;
	for (auto pair : map)
	{
		test_len = pair.first.length();
		if (test_len > max_len) max_len = test_len;
	}
	return max_len;
}

/*
 * Implementation Notes:
 *     also implements the most brute-force search for the largest element in a linear data
 *     structure. Unfortunately, this function has O(N) time complexity, but the input map is not 
 *     expected to exceed 10 elements, so the penalties at higher N will most likely never be seen.
 */
uint32_t get_max_first_len(std::vector<std::pair<std::string, std::string>> &vec)
{
	uint32_t max_len = 0;
	uint32_t test_len = 0;
	for (auto pair: vec)
	{
		test_len = pair.first.length();
		if (test_len > max_len) max_len = test_len;
	}
	return max_len;
}

