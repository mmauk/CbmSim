/*
 * eyelidintegrator.h
 *
 *  Created on: Jan 11, 2013
 *      Author: consciousness
 */

#ifndef EYELIDINTEGRATOR_H_
#define EYELIDINTEGRATOR_H_

#include <stdDefinitions/pstdint.h>
#include <math.h>

class EyelidIntegrator
{
public:
	EyelidIntegrator(unsigned int numInputCells, unsigned int msPerTimeStep,
			float gDecayTau, float gIncrease, float gShift, float gLeakRaw, float maxAmplitude);

	~EyelidIntegrator();

	float calcStep(const ct_uint8_t *apIn);

private:
	EyelidIntegrator();

	unsigned int numInCells;
	unsigned int msPerTS;

	float gDecayTau;
	float gDecay;
	float gInc;
	float gShift;
	float evMax;

	float gLeak;
	float vm;

	float *gIn;

};


#endif /* EYELIDINTEGRATOR_H_ */
