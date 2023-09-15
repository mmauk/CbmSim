#ifndef BITS_H_
#define BITS_H_

#include <stdint.h>

#define BITS_PER_BYTE 8

void populate_byte_arr(uint8_t *byte_arr, const uint32_t byte_arr_len);

void populate_random_byte_arr(uint8_t *byte_arr, const uint32_t byte_arr_len);

void pack_byte_array(uint8_t *byte_arr, const uint32_t byte_arr_len,
                     uint8_t *packed_byte_arr);

void pack_2d_byte_array(uint8_t **byte_arr_2d, const uint32_t byte_arr_num_rows,
                        const uint32_t byte_arr_num_cols,
                        uint8_t *packed_byte_arr, uint32_t offset);

void print_byte_bit_repr(uint8_t byte);

void unpack_byte_array(uint8_t *packed_byte_arr, uint8_t *unpacked_byte_arr,
                       const uint32_t unpacked_byte_arr_len);

void print_byte_bit_repr_arr(uint8_t *bytes);

#endif /* BITS_H_ */
