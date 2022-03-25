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

	virtual void setGIncGRtoGO(float inc) = 0;
	virtual void resetGIncGRtoGO() 		  = 0;

	virtual const ct_uint8_t* exportAPMF()  = 0;
	virtual const ct_uint8_t* exportAPSC()  = 0;
	virtual const ct_uint8_t* exportAPGO()  = 0;
	virtual const ct_uint8_t* exportAPGR()  = 0;
	virtual const ct_uint8_t* exportAPUBC() = 0;

	virtual const ct_uint8_t* exportHistMF() = 0;

	virtual const ct_uint32_t* exportAPBufMF() = 0;
	virtual const ct_uint32_t* exportAPBufGR() = 0;
	virtual const ct_uint32_t* exportAPBufGO() = 0;
	virtual const ct_uint32_t* exportAPBufSC() = 0;

	virtual const float* exportVmGR() = 0;
	virtual const float* exportVmGO() = 0;
	
	virtual const float* exportExGOInput()  = 0;
	virtual const float* exportInhGOInput() = 0;

	virtual const float* exportgSum_GOGO()   = 0;
	virtual const float* exportgSum_MFGO()   = 0;
	virtual const float* exportgSum_GRGO()   = 0;
	virtual const float* exportVGOGOcouple() = 0;
	virtual const float* exportVmSC() 		 = 0;

	virtual const float* exportGESumGR() 			 = 0;
	virtual const float* exportGUBCESumGR() 		 = 0;
	virtual const float* exportgNMDAGR() 			 = 0;
	virtual const int* exportAPfromMFtoGR() 		 = 0;
	virtual const float* exportDepSumUBCGR() 		 = 0;
	virtual const float* exportDepSumGOGR() 		 = 0;
	virtual const float* exportDynamicSpillSumGOGR() = 0;
	virtual const float* exportGISumGR() 			 = 0;

	virtual const ct_uint32_t* exportSumGRInputGO() = 0;
	virtual const float* exportSumGOInputGO() 		= 0;
	virtual const float* exportGOOutSynScaleGOGO() 	= 0;
	virtual const float* exportgGOGO() 				= 0;
	virtual const float* exportvSum_GOGO() 			= 0;
	virtual const float* exportvSum_GRGO() 			= 0;
	virtual const float* exportvSum_MFGO() 			= 0;
};

#endif /* INNETINTERFACE_H_ */
