 /*
 * rawbytesrw.cpp
 *
 *  Created on: Nov 7, 2012
 *      Author: consciousness
 */

#include "fileIO/rawbytesrw.h"

using namespace std;
void rawBytesRW(char *arr, unsigned long byteLen, bool read, fstream &file)
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
