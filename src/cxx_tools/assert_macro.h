#include <stdlib.h>
#include "logger.h"

#define ASSERT(predicate, err_str, func_name) \
  if (!(predicate)) { \
    LOG_FATAL("%s(): %s", (func_name), (err_str)); \
    exit(1); \
  }

