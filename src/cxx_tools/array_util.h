#ifndef ARRAY_UTIL_H_
#define ARRAY_UTIL_H_

// get number of elements in an array
#define NELEM(array) (sizeof(array) / sizeof(*(array)))

// loop over an array of given size:
#define FOREACH_NELEM(array, nelem, iter)                                      \
  for (__typeof__(*(array)) *iter = (array); iter < (array) + (nelem); iter++)

// loop over an array of known size
#define FOREACH(array, iter) FOREACH_NELEM(array, NELEM(array), iter)

#include "sfmt.h"
#include <time.h>

// from a response in: ...where?

/* Arrange the N elements of array in random order, using fisher-yates method */
template <typename T> void fisher_yates_shuffle(T *array, size_t N) {
  CRandomSFMT0 randGen(time(0));
  for (size_t i = 0; i < N; i++) {
    size_t j = i + randGen.IRandom(0, N - i - 1); // TRIPLE CHECK!
    T t = array[j];
    array[j] = array[i];
    array[i] = t;
  }
}

#endif /* ARRAY_UTIL_H_ */
