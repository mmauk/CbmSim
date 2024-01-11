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
#include <omp.h>

#include "logger.h"
#include "sfmt.h"
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

// potentially dangerous: does not check the diminesions of the new array
// also: caller owns the memory!
template <typename Type>
Type ** transpose2DArray(Type **in, uint64_t num_rows_old, uint64_t num_cols_old)
{
	double start, stop; 
	Type ** result = allocate2DArray<Type>(num_cols_old, num_rows_old);
	start = omp_get_wtime();
	#pragma omp parallel for
	for (size_t i = 0; i < num_rows_old; i++)
	{
		for (size_t j = 0; j < num_cols_old; j++)
		{
			result[j][i] = in[i][j];
		}
	}
	stop = omp_get_wtime();
	LOG_INFO("transpose took %0.2fs", stop - start);
	return result;
}

// IMPORTANT: SHUFFLES IN-PLACE
template <typename Type>
void shuffle_along_axis(Type **in_arr, uint64_t num_rows, uint64_t num_cols, uint32_t axis = 0)
{
	CRandomSFMT0 randGen(time(0));
	if (axis == 0)
	{
		Type *shared_row_buf = (Type *)calloc(num_cols, sizeof(Type));
		for (size_t i = 0; i < num_rows; i++)
		{
			size_t j = i + randGen.IRandom(0, num_rows - i - 1);
			memcpy(shared_row_buf, in_arr[j], num_cols * sizeof(Type));
			memcpy(in_arr[j], in_arr[i], num_cols * sizeof(Type));
			memcpy(in_arr[i], shared_row_buf, num_cols * sizeof(Type));
		}
		free(shared_row_buf);
	}
	else
	{
		Type *shared_col_buf = (Type *)calloc(num_rows, sizeof(Type));
		for (size_t i = 0; i < num_cols; i++)
		{
			size_t j = i + randGen.IRandom(0, num_cols - i - 1);
			for (size_t k = 0; k < num_rows; k++)
			{
				shared_col_buf[k] = in_arr[k][i]; // painful if columns are loooooong
				in_arr[k][j] = in_arr[k][i];
				in_arr[k][i] = shared_col_buf[k];
			}
		}
		free(shared_col_buf);
	}
}

template <typename Type>
void write2DArray(std::string out_file_name, Type **inArr,
	unsigned long long num_row, unsigned long long num_col, bool append = false)
{
	std::ios_base::openmode app_opt = (append) ? std::ios_base::app : (std::ios_base::openmode)0;
	std::fstream out_file_buf(out_file_name.c_str(), std::ios::out | std::ios::binary | app_opt);

	if (!out_file_buf.is_open())
	{
		fprintf(stderr, "[ERROR]: Couldn't open '%s' for writing. Exiting...\n", out_file_name.c_str());
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

