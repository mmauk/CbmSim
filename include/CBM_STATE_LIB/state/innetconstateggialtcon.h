/*
 * innetconstateggialtcon.h
 *
 *  Created on: Nov 26, 2013
 *      Author: consciousness
 */

#ifndef INNETCONSTATEGGIALTCON_H_
#define INNETCONSTATEGGIALTCON_H_

#include "innetconnectivitystate.h"
#include <vector>

#ifdef INTELCC
#include <mathimf.h>
#else //otherwise use standard math library
#include <math.h>
#endif

class InNetConStateGGIAltCon: public virtual InNetConnectivityState
{
public:
	InNetConStateGGIAltCon(ConnectivityParams *parameters, unsigned int msPerStep, int randSeed);

protected:
	virtual void connectGOGO(CRandomSFMT *randGen);
private:
	InNetConStateGGIAltCon();
};


#endif /* INNETCONSTATEGGIALTCON_H_ */
