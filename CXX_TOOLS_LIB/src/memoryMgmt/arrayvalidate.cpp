/*
 * arrayvalidate.cpp
 *
 *  Created on: Mar 9, 2013
 *      Author: consciousness
 */

#include "memoryMgmt/arrayvalidate.h"

bool validateFloatArray(float *array, unsigned int numElements)
{
	bool valid;

	valid=true;

	for(int i=0; i<numElements; i++)
	{
		if(isnanf(array[i]) || (isinff(array[i])!=0))
		{
			valid=false;
		}
	}

	return valid;
}

bool validateDoubleArray(double *array, unsigned int numElements)
{
	bool valid;

	valid=true;

	for(int i=0; i<numElements; i++)
	{
		if(isnan(array[i]) || (isinf(array[i])!=0))
		{
			valid=false;
		}
	}

	return valid;
}
