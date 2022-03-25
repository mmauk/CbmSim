/*
 * mzoneinterface.h
 *
 *  Created on: Oct 1, 2012
 *      Author: consciousness
 */

#ifndef MZONEINTERFACE_H_
#define MZONEINTERFACE_H_

#include <stdDefinitions/pstdint.h>

class MZoneInterface
{
public:
	virtual ~MZoneInterface();

	virtual void setGRPCPlastSteps(float ltdStep, float ltpStep) = 0;
	virtual void resetGRPCPlastSteps() = 0;

	virtual const ct_uint8_t* exportAPBC() = 0;
	virtual const ct_uint8_t* exportAPPC() = 0;
	virtual const ct_uint8_t* exportAPIO() = 0;
	virtual const ct_uint8_t* exportAPNC() = 0;

	virtual const ct_uint32_t* exportAPBufBC() = 0;
	virtual const ct_uint32_t* exportAPBufPC() = 0;
	virtual const ct_uint32_t* exportAPBufIO() = 0;
	virtual const ct_uint32_t* exportAPBufNC() = 0;

	virtual const float* exportgBCPC() = 0;
	virtual const float* exportgPFPC() = 0;
	virtual const float* exportVmBC()  = 0;
	virtual const float* exportVmPC()  = 0;
	virtual const float* exportVmNC()  = 0;
	virtual const float* exportVmIO()  = 0;
	virtual const float* exportPFPCWeights() = 0;
};

#endif /* MZONEINTERFACE_H_ */
