/*
 * dynamic2darray.h
 *
 *  Created on: Oct 8, 2012
 *      Author: consciousness
 */
#ifndef _DYNAMIC2DARRAY_H
#define _DYNAMIC2DARRAY_H

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "file_utility.h"
#include "arrayvalidate.h"

template<typename Type>
Type** allocate2DArray(unsigned long long numRows, unsigned long long numCols)
{
	Type** retArr = (Type **)calloc(numRows, sizeof(Type *));
	retArr[0] = (Type *)calloc(numRows * numCols, sizeof(Type));

	for (unsigned long long i = 1; i < numRows; i++)
	{
		retArr[i] = &(retArr[0][i * numCols]);
	}

	return retArr;
}

template <typename Type>
void write2DArray(std::string out_file_name, Type **inArr,
	unsigned long long num_row, unsigned long long num_col, bool append = false)
{
	std::ios_base::openmode app_opt = (append) ? std::ios_base::app : (std::ios_base::openmode)0;
	std::fstream out_file_buf(out_file_name.c_str(), std::ios::out | std::ios::binary | app_opt);

	if (!out_file_buf.is_open())
	{
		fprintf(stderr, "[INFO]: Couldn't open '%s' for writing. Exiting...\n", out_file_name.c_str());
		exit(-1);
	}
	rawBytesRW((char *)inArr[0], num_row * num_col * sizeof(Type), false, out_file_buf);
	out_file_buf.close();
}

template<typename Type>
void delete2DArray(Type** array)
{
	free(array[0]);
	free(array);
}

template <typename Type>
inline bool validate2DArray(Type **array, unsigned long long numElements)
{
	return validateFloatArray(array[0], numElements);
}

#endif /* _DYNAMIC2DARRAY_H */

