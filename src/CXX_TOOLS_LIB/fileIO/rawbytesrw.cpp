 /*
 * rawbytesrw.cpp
 *
 *  Created on: Nov 7, 2012
 *      Author: consciousness
 */

#include "fileIO/rawbytesrw.h"

void rawBytesRW(char *arr, size_t byteLen, bool read, std::fstream &file)
{
	if(read)
	{
		file.read(arr, byteLen);
	}
	else
	{
		file.write(arr, byteLen);
	}
}
