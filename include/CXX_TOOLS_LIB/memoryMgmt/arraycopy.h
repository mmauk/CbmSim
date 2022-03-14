/*
 * arraycopy.h
 *
 *  Created on: Apr 30, 2013
 *      Author: consciousness
 */

#ifndef ARRAYCOPY_H_
#define ARRAYCOPY_H_

template<typename Type> void arrayCopy(Type *destArray, Type *srcArray, int numElements)
{
	for(int i=0; i<numElements; i++)
	{
		destArray[i]=srcArray[i];
	}
}


#endif /* ARRAYCOPY_H_ */
