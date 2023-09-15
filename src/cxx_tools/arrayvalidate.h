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

template <typename Type> bool validateArray(Type *array, unsigned int numElem) {
  bool valid = true;
  for (size_t i = 0; i < numElem; i++) {
    // expected types are int and float, so int arr elems get type casted
    // implicitly
    if (isnanf(array[i]) || isinff(array[i])) {
      valid = false;
      break;
    }
  }
  return valid;
}

#endif /* ARRAYVALIDATE_H_ */
