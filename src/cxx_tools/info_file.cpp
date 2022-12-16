#include <unistd.h> // geteuid
#include <pwd.h> // for getpwuid 
#include <ctime> 
#include <sstream> // string stream
#include <iomanip> // put_time

#include "logger.h"
#include "info_file.h"

const uint32_t INFO_FILE_COL_WIDTH = 79;
const std::string SIM_VERSION = "0.0.1";

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

void cp_to_info_file_data(parsed_commandline &p_cl, parsed_sess_file &s_file, info_file_data &if_data)
{
	cp_parsed_commandline(p_cl, if_data.p_cl);
	cp_parsed_sess_file(s_file, if_data.s_file);
}

const char *get_user_name()
{
	uid_t uid = geteuid();
	struct passwd *pw = getpwuid(uid);
	if (pw)
	{
		return pw->pw_name;
	}
	
	return "";
}

void set_info_file_str_props(enum when when, info_file_data &if_data)
{
	std::time_t t = std::time(0);
	std::tm *now = std::localtime(&t);
	std::stringstream formatter;
	if (when == BEFORE_RUN)
	{
		formatter << std::put_time(now, DEFAULT_DATE_FORMAT.c_str());
		if_data.start_date = formatter.str();
		std::stringstream().swap(formatter);
		formatter << std::put_time(now, DEFAULT_LOCALE_FORMAT.c_str());
		if_data.locale = formatter.str();
		std::stringstream().swap(formatter);
		formatter << std::put_time(now, DEFAULT_TIME_FORMAT.c_str());
		if_data.start_time = formatter.str();
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
		std::stringstream().swap(formatter);
		formatter << std::put_time(now, DEFAULT_TIME_FORMAT.c_str());
		if_data.end_time = formatter.str();
		std::stringstream().swap(formatter);
	}
}


