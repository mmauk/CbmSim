/*
 * arrayinitalize.h
 *
 *  Created on: Jan 7, 2013
 *      Author: consciousness
 */

#ifndef ARRAYINITALIZE_H_
#define ARRAYINITALIZE_H_

template<typename Type> void arrayInitalize(Type *array, Type val, unsigned int numElem)
{
	for (size_t i = 0; i < numElem; i++)
	{
		array[i] = val;
	}
}

#endif /* ARRAYINITALIZE_H_ */
