/*
 * arrayvalidate.h
 *
 *  Created on: Mar 9, 2013
 *      Author: consciousness
 */

#ifndef ARRAYVALIDATE_H_
#define ARRAYVALIDATE_H_

#ifdef INTELCC
#include <mathimf.h>
#else
#include <math.h>
#endif

bool validateFloatArray(float *array, unsigned int numElements);

bool validateDoubleArray(double *array, unsigned int numElements);


#endif /* ARRAYVALIDATE_H_ */
