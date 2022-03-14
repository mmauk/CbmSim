/*
 * dynamic2darray.h
 *
 *  Created on: Oct 8, 2012
 *      Author: consciousness
 */

#ifndef DYNAMIC2DARRAY_H_
#define DYNAMIC2DARRAY_H_

template<typename Type> Type** allocate2DArray(unsigned int numRows, unsigned int numCols)
{
	Type** retArr;

	retArr=new Type*[numRows];
	retArr[0]=new Type[numRows*numCols];

	for(int i=1; i<numRows; i++)
	{
		retArr[i]=&(retArr[0][i*numCols]);
	}

	return retArr;
}

template<typename Type> void delete2DArray(Type** array)
{
	delete[] array[0];
	delete[] array;
}

bool validate2DfloatArray(float ** array, unsigned int numElements);

bool validate2DdoubleArray(double **array, unsigned int numElements);

#endif /* DYNAMIC2DARRAY_H_ */
