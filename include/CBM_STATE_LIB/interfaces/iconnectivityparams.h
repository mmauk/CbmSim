/*
 * iconnectivityparams.h
 *
 *  Created on: Dec 14, 2012
 *      Author: consciousness
 */

#ifndef ICONNECTIVITYPARAMS_H_
#define ICONNECTIVITYPARAMS_H_

#include <iostream>
#include <map>
#include <string>

#include <stdDefinitions/pstdint.h>

class IConnectivityParams
{
public:
	virtual ~IConnectivityParams();

	virtual void showParams(std::ostream &outSt)=0;

	virtual ct_uint32_t getGOX()=0;
	virtual ct_uint32_t getGOY()=0;
	virtual ct_uint32_t getGRX()=0;
	virtual ct_uint32_t getGRY()=0;
	virtual ct_uint32_t getGLX()=0;
	virtual ct_uint32_t getGLY()=0;

	virtual ct_uint32_t getNumMF()=0;
	virtual ct_uint32_t getNumGO()=0;
	virtual ct_uint32_t getNumGR()=0;
	virtual ct_uint32_t getNumGL()=0;

	virtual ct_uint32_t getNumSC()=0;
	virtual ct_uint32_t getNumBC()=0;
	virtual ct_uint32_t getNumPC()=0;
	virtual ct_uint32_t getNumNC()=0;
	virtual ct_uint32_t getNumIO()=0;

	virtual std::map<std::string, ct_uint32_t> getParamCopy()=0;
};


#endif /* ICONNECTIVITYPARAMS_H_ */
