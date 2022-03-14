/*
 * eyelidintegrator.cpp
 *
 *  Created on: Jan 11, 2013
 *      Author: consciousness
 */

#include "eyelidintegrator.h"
#include <iostream>

using namespace std;

EyelidIntegrator::EyelidIntegrator(unsigned int numInputCells, unsigned int msPerTimeStep,
		float gDecayTau, float gIncrease, float gShift, float gLeakRaw, float maxAmplitude)
{
	numInCells=numInputCells;
	msPerTS=msPerTimeStep;

	this->gDecayTau=gDecayTau;
	gDecay=exp(-((int)msPerTS)/gDecayTau);
	this->gInc=gIncrease;
	this->gShift=gShift;
	evMax=maxAmplitude;

	gLeak=gLeakRaw/(6-((int)msPerTS));
	vm=0;

//	cout<<numInCells<<endl;
//	cout<<msPerTS<<endl;
//	cout<<gDecayT<<endl;
//	cout<<gDecay<<endl;
//	cout<<gInc<<endl;
//	cout<<evMax<<endl;
//	cout<<gL<<endl;
//	cout<<vm<<endl;

	gIn=new float[numInCells];
	for(int i=0; i<numInCells; i++)
	{
		gIn[i]=0;
//		cout<<gIn[i]<<endl;
	}
}

EyelidIntegrator::~EyelidIntegrator()
{
	delete gIn;
}

float EyelidIntegrator::calcStep(const ct_uint8_t *apIn)
{
	float gSum;

	gSum=0;
	for(int i=0; i<numInCells; i++)
	{
		gIn[i]=gIn[i]*gDecay;
		gIn[i]=gIn[i]+apIn[i]*gInc;
		gSum+=gIn[i];
	}

	gSum=gSum+gShift;
	gSum=(gSum>0)*gSum;

	vm=vm+(gLeak*(0-vm))+(gSum*(evMax-vm));

	return vm;
}
