#include <stdio.h>
#include <math.h>
#include "bits/bits.h"
#include "array_util.h"

void pack_byte_array(uint8_t *byte_arr,
					 const uint32_t byte_arr_len,
					 uint8_t *packed_byte_arr)
{
	uint8_t packed_elem_i = '\000';
	uint32_t bit_counter = 1;
	for (uint32_t i = 0; i < byte_arr_len; i++)
	{
		uint32_t bit_index  = i % 8;
		uint8_t temp_byte = byte_arr[i];
		packed_elem_i |= temp_byte << bit_index;
		if (bit_counter == 8)
		{
			uint32_t byte_index = floor(i / 8);
			packed_byte_arr[byte_index] = packed_elem_i;
			packed_elem_i = '\000';
			bit_counter = 1;
		}
		else bit_counter++;
	}
}

void pack_2d_byte_array(uint8_t **byte_arr_2d,
					 const uint32_t byte_arr_num_rows,
					 const uint32_t byte_arr_num_cols,
					 uint8_t *packed_byte_arr,
					 uint32_t offset)
{
	uint8_t *packed_byte_arr_cpy = packed_byte_arr + offset;
	for (uint32_t i = 0; i < byte_arr_num_rows; i++)
	{
		pack_byte_array(byte_arr_2d[i], byte_arr_num_cols, packed_byte_arr_cpy);
		packed_byte_arr_cpy += byte_arr_num_cols;
	}
}

void unpack_byte_array(uint8_t *packed_byte_arr,
					   uint8_t *unpacked_byte_arr,
					   const uint32_t unpacked_byte_arr_len)
{
	uint8_t bit_mask = '\001';
	for (uint32_t i = 0; i < unpacked_byte_arr_len; i++)
	{
		uint32_t bit_index = i % 8;
		uint32_t byte_index = floor(i / 8); 
		uint8_t temp_byte = packed_byte_arr[byte_index] & (bit_mask << bit_index);
		unpacked_byte_arr[i] = (temp_byte >> bit_index);
	}
}

void print_byte_bit_repr(uint8_t byte)
{
	uint8_t bit_flag = '\001';
	char byte_as_str[BITS_PER_BYTE+1];
	for (uint32_t i = 0; i < BITS_PER_BYTE; i++)
	{
		bit_flag &= byte >> i;
		if (bit_flag == '\001') byte_as_str[BITS_PER_BYTE - i - 1] = '1';
		else if (bit_flag == '\000') byte_as_str[BITS_PER_BYTE - i - 1] = '0';
		bit_flag = '\001';
	}
	byte_as_str[BITS_PER_BYTE] = '\0';
	puts(byte_as_str);
}

void print_byte_bit_repr_arr(uint8_t *bytes)
{
	FOREACH(bytes, bp) print_byte_bit_repr(*bp);
}

