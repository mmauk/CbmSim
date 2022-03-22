/*
 * arrayvalidate.cpp
 *
 *  Created on: Mar 9, 2013
 *      Author: consciousness
 */

#include "memoryMgmt/arrayvalidate.h"

bool validateFloatArray(float *array, unsigned int numElem)
{
	bool valid = true;

	for (size_t i = 0; i < numElem; i++)
	{
		if (isnanf(array[i]) || isinff(array[i])
		{
			valid = false;
			break;
		}
	}

	return valid;
}

bool validateDoubleArray(double *array, unsigned int numElem)
{
	bool valid = true;

	for (size_t i = 0; i < numElem; i++)
	{
		if (isnan(array[i] || isinf(array[i])
		{
			valid=false;
			break;
		}
	}

	return valid;
}

