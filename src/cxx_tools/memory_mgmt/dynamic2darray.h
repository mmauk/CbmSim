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

#include "arrayvalidate.h"

template<typename Type> Type** allocate2DArray(unsigned int numRows, unsigned int numCols)
{
	Type** retArr = (Type **)calloc(numRows, sizeof(Type *));
	retArr[0] = (Type *)calloc(numRows * numCols, sizeof(Type));

	for (size_t i = 1; i < numRows; i++)
	{
		retArr[i] = &(retArr[0][i * numCols]);
	}

	return retArr;
}

template<typename Type> void delete2DArray(Type** array)
{
	free(array[0]);
	free(array);
}

inline bool validate2DfloatArray(float **array, unsigned int numElements)
{
	return validateFloatArray(array[0], numElements);
}

inline bool validate2DdoubleArray(double **array, unsigned int numElements)
{
	return validateDoubleArray(array[0], numElements);
}

#endif /* _DYNAMIC2DARRAY_H */

