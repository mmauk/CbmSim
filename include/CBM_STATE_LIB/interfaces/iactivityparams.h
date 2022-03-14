/*
 * iactivityparams.h
 *
 *  Created on: Dec 14, 2012
 *      Author: consciousness
 */

#ifndef IACTIVITYPARAMS_H_
#define IACTIVITYPARAMS_H_

#include <iostream>
#include <stdDefinitions/pstdint.h>
#include <string>
#include <map>

class IActivityParams
{
public:
	virtual ~IActivityParams();

	virtual unsigned int getMSPerTimeStep()=0;

	virtual void showParams(std::ostream &outSt)=0;

	virtual std::map<std::string, float> getParamCopy()=0;

	virtual float getParam(std::string paramName)=0;

	virtual bool setParam(std::string paramName, float value)=0;
};

#endif /* IACTIVITYPARAMS_H_ */
