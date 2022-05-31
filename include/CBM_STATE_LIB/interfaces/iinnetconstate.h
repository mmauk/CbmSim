/*
 * iinnetconstate.h
 *
 *  Created on: Dec 14, 2012
 *      Author: consciousness
 */

#ifndef IINNETCONSTATE_H_
#define IINNETCONSTATE_H_

#include <vector>
#include <stdDefinitions/pstdint.h>

class IInNetConState
{
public:
	virtual ~IInNetConState();

	//virtual std::vector<int> getpGOfromGOtoGLCon(int goN) = 0;
	//virtual std::vector<int> getpGOfromGLtoGOCon(int goN) = 0;
	//virtual std::vector<int> getpMFfromMFtoGLCon(int mfN) = 0;
	//virtual std::vector<int> getpGLfromGLtoGRCon(int glN) = 0;

	//virtual std::vector<int> getpGRfromMFtoGR(int grN) 			 = 0;
	//virtual std::vector<std::vector<int> > getpGRPopfromMFtoGR() = 0;

	//virtual std::vector<int> getpGRfromGOtoGRCon(int grN)			= 0;
	//virtual std::vector<std::vector<int> > getpGRPopfromGOtoGRCon() = 0;

	//virtual std::vector<int> getpGRfromGRtoGOCon(int grN) 		    = 0;
	//virtual std::vector<std::vector<int> > getpGRPopfromGRtoGOCon() = 0;

	//virtual std::vector<int> getpGOfromGRtoGOCon(int goN) 			= 0;
	//virtual std::vector<std::vector<int> > getpGOPopfromGRtoGOCon() = 0;

	//virtual std::vector<int> getpGOfromGOtoGRCon(int goN) 			= 0;
	//virtual std::vector<std::vector<int> > getpGOPopfromGOtoGRCon() = 0;

	//virtual std::vector<int> getpMFfromMFtoGRCon(int mfN) = 0;

	//virtual std::vector<int> getpMFfromMFtoGOCon(int mfN) 			= 0;
	//virtual std::vector<int> getpGOfromMFtoGOCon(int goN) 			= 0;
	//virtual std::vector<std::vector<int> > getpGOPopfromMFtoGOCon() = 0;

	//virtual std::vector<int> getpGOOutGOGOCon(int goN) 			 = 0;
	//virtual std::vector<std::vector<int> > getpGOPopOutGOGOCon() = 0;

	//virtual std::vector<int> getpGOInGOGOCon(int goN) 			= 0;
	//virtual std::vector<std::vector<int> > getpGOPopInGOGOCon() = 0;

	//virtual std::vector<int> getpGOCoupOutGOGOCon(int goN) 			 = 0;
	//virtual std::vector<std::vector<int> > getpGOPopCoupOutGOGOCon() = 0;

	//virtual std::vector<int> getpGOCoupInGOGOCon(int goN) 			= 0;
	//virtual std::vector<std::vector<int> > getpGOPopCoupInGOGOCon() = 0;

	//virtual std::vector<ct_uint32_t> getGOIncompIndfromGRtoGO() = 0;
	//virtual std::vector<ct_uint32_t> getGRIncompIndfromGRtoGO() = 0;

	virtual bool deleteGOGOConPair(int srcGON, int destGON) = 0;
	virtual bool addGOGOConPair(int srcGON, int destGON) 	= 0;
};

#endif /* IINNETCONSTATE_H_ */
