/*
 * arrayinitalize.h
 *
 *  Created on: Jan 7, 2013
 *      Author: consciousness
 */

#ifndef ARRAYINITALIZE_H_
#define ARRAYINITALIZE_H_

template<typename Type> void arrayInitialize(Type *array, Type val, int numElements)
{
	for(int i=0; i<numElements; i++)
	{
		array[i]=val;
	}
}


#endif /* ARRAYINITALIZE_H_ */
