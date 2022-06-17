/*
 * innetinterface.h
 *
 *  Created on: Oct 1, 2012
 *      Author: consciousness
 */

#ifndef INNETINTERFACE_H_
#define INNETINTERFACE_H_

#include <stdDefinitions/pstdint.h>

class InNetInterface
{
public:
	virtual ~InNetInterface();

	virtual const ct_uint8_t* exportAPMF()  = 0;
	virtual const ct_uint8_t* exportAPSC()  = 0;
	virtual const ct_uint8_t* exportAPGR()  = 0;

	//virtual const ct_uint32_t* exportAPBufGR() = 0;
	virtual const ct_uint32_t* exportAPBufSC() = 0;

	virtual const float* exportVmGR() = 0;
	
	virtual const float* exportVmSC() 		 = 0;

	virtual const float* exportGUBCESumGR() 		 = 0;
	virtual const float* exportgNMDAGR() 			 = 0;
	virtual const int* exportAPfromMFtoGR() 		 = 0;
	virtual const float* exportDepSumUBCGR() 		 = 0;
	virtual const float* exportDynamicSpillSumGOGR() = 0;
	virtual const float* exportGISumGR() 			 = 0;

	virtual const ct_uint32_t* exportSumGRInputGO() = 0;
	virtual const float* exportSumGOInputGO() 		= 0;
};

#endif /* INNETINTERFACE_H_ */
