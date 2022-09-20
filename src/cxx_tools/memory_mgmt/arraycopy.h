/*
 * arraycopy.h
 *
 *  Created on: Apr 30, 2013
 *      Author: consciousness
 */

#ifndef ARRAYCOPY_H_
#define ARRAYCOPY_H_

template<typename Type> void arrayCopy(Type *destArray, Type *srcArray, int numElem)
{
	for (size_t i = 0; i < numElem; i++)
	{
		destArray[i] = srcArray[i];
	}
}

#endif /* ARRAYCOPY_H_ */

