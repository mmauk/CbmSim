#include <stdlib.h>
#include "logger.h"

static void assert(bool expr, const char *error_string, const char *func = "assert")
{
	if (!expr)
	{
		LOG_FATAL("%s(): %s", func, error_string);
		exit(1);
	}
}

