#ifndef ARRAY_UTIL_H_
#define ARRAY_UTIL_H_

// get number of elements in an array
#define NELEM(array) (sizeof(array) / sizeof(*(array)))

// loop over an array of given size:
#define FOREACH_NELEM(array, nelem, iter) \
	for (__typeof__(*(array)) *iter = (array); \
		iter < (array) + (nelem); \
		iter++)

// loop over an array of known size
#define FOREACH(array, iter) \
   FOREACH_NELEM(array, NELEM(array), iter)

#define ARR_FILLED_WITH(array, nelem, val) \
   FOREACH_NELEM((array), (nelem), (a)) { \
	  if (*(a) != 

#include <stdint.h>
#include <limits>
#include <cmath>

template <typename T>
bool nearly_equal(T a, T b)
{
	return std::nextafter(a, std::numeric_limits<T>::lowest()) <= b
		&& std::nextafter(a, std::numeric_limits<T>::max()) >= b;
}

template <typename T>
bool nearly_equal(T a, T b, int factor /* a factor of epsilon */)
{
	T min_a = a - (a - std::nextafter(a, std::numeric_limits<T>::lowest())) * factor;
	T max_a = a + (std::nextafter(a, std::numeric_limits<T>::max()) - a) * factor;

	return min_a <= b && max_a >= b;
}

template <typename T>
bool arr_filled_with_int_t(T *arr, uint64_t len, T val)
{
	for (size_t i = 0; i < len; i++)
	{
		if (arr[i] != val) return false;
	}
	return true;
}

//TODO: TEST TEST TEST
template <typename T>
bool arr_filled_with_float_t(T *arr, uint64_t len, T val)
{
	for (size_t i = 0; i < len; i++)
	{
		if (!nearly_equal(arr[i], val)) return false;
	}
	return true;
}

#endif /* ARRAY_UTIL_H_ */

