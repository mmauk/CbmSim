/*
 * dynamic2darray.h
 *
 *  Created on: Oct 8, 2012
 *      Author: consciousness
 */
#ifndef _DYNAMIC2DARRAY_H
#define _DYNAMIC2DARRAY_H

#include "file_utility.h"
#include <cstddef>
#include <cstdio>
#include <cstdlib>

template <typename Type>
Type **allocate2DArray(unsigned long long numRows, unsigned long long numCols) {
  Type **retArr = (Type **)calloc(numRows, sizeof(Type *));
  retArr[0] = (Type *)calloc(numRows * numCols, sizeof(Type));

  for (unsigned long long i = 1; i < numRows; i++) {
    retArr[i] = &(retArr[0][i * numCols]);
  }

  return retArr;
}

// potentially dangerous: does not check the diminesions of the new array
// also: caller owns the memory!
template <typename Type>
Type **transpose2DArray(Type **in, uint64_t num_rows_old,
                        uint64_t num_cols_old) {
  Type **result = allocate2DArray<Type>(num_cols_old, num_rows_old);
  for (size_t i = 0; i < num_rows_old; i++) {
    for (size_t j = 0; j < num_cols_old; j++) {
      result[j][i] = in[i][j];
    }
  }
  return result;
}

/*
 * @brief: normalizes the 2D array wrt its maximum value st.
 *         all array values lie in the interval [0.0, 1.0].
 *         Does so in-place, ie modifies the actual vals of
 *         the arr.
 */
template <typename Type>
void norm_2d_array_ip(Type **arr, uint64_t numRows, uint64_t numCols) {
  auto max_it = std::max_element(arr[0], arr[0] + numRows * numCols);
  Type val = *max_it;
  std::transform(arr[0], arr[0] + numRows * numCols, arr[0],
                 [=](Type elem) { return elem / val; });
}

template <typename T, typename V = float>
void mult_2d_array_by_val_ip(T **arr, uint64_t numRows, uint64_t numCols,
                             V val) {
  std::transform(arr[0], arr[0] + numRows * numCols, arr[0],
                 [=](T elem) { return elem * val; });
}

template <typename Type>
void write2DArray(std::string out_file_name, Type **inArr,
                  unsigned long long num_row, unsigned long long num_col,
                  bool append = false) {
  std::ios_base::openmode app_opt =
      (append) ? std::ios_base::app : (std::ios_base::openmode)0;
  std::fstream out_file_buf(out_file_name.c_str(),
                            std::ios::out | std::ios::binary | app_opt);

  if (!out_file_buf.is_open()) {
    fprintf(stderr, "[ERROR]: Couldn't open '%s' for writing. Exiting...\n",
            out_file_name.c_str());
    exit(-1);
  }
  rawBytesRW((char *)inArr[0], num_row * num_col * sizeof(Type), false,
             out_file_buf);
  out_file_buf.close();
}

template <typename Type> void delete2DArray(Type **array) {
  free(array[0]);
  free(array);
}

template <typename Type>
inline bool validate2DArray(Type **array, unsigned long long numElements) {
  return validateFloatArray(array[0], numElements);
}

#endif /* _DYNAMIC2DARRAY_H */
