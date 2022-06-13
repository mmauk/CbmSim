/*
 * activityparams.h
 *
 *  Created on: Oct 11, 2012
 *      Author: varicella
 */

#ifndef ACTIVITYPARAMS_H_
#define ACTIVITYPARAMS_H_

#include <math.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include <stdDefinitions/pstdint.h>
#include "interfaces/iactivityparams.h"

class ActivityParams : public virtual IActivityParams
{
public:
	ActivityParams();
	ActivityParams(std::string actParamFile);
	ActivityParams(const ActivityParams &copyFrom);
	~ActivityParams(); /*here to serve the Rule of Three: map deallocates itself bwahaha*/

	void writeParams(std::fstream &outfile);
	unsigned int getMSPerTimeStep();
	float getParam(std::string paramName);
	bool setParam(std::string paramName, float value);
	std::string toString();
	friend std::ostream &operator<<(std::ostream &os, ActivityParams &ap);
	ActivityParams &operator=(const ActivityParams &copyFrom);

	//universtal parameters
	float msPerTimeStep;

	//---------mossy fiber variables
	float msPerHistBinMF;
	ct_uint32_t numTSinMFHist;

	//---------golgi cell variables
	float eLeakGO;
	float eMGluRGO;
	float eGABAGO;
	float threshMaxGO;
	float threshRestGO;
	float gIncMFtoGO;
	
	float NMDA_AMPAratioMFGO;
	float gDecTauMFtoGONMDA;
	float gDecayMFtoGONMDA;

	float gGABAIncGOtoGO;
	float coupleRiRjRatioGO;

	float gMGluRScaleGRtoGO;
	float gMGluRIncScaleGO;
	float mGluRScaleGO;
	float gluScaleGO;
	float gLeakGO; //derived from raw values
	float gConstGO;
	//synaptic depression test for GOGABAGO
	float goGABAGOGOSynRecTau;
	float goGABAGOGOSynRec;
	float goGABAGOGOSynDepF;

	float gDecTauMFtoGO;
	float gDecMFtoGO; //derived from tau
	float gDecTauGRtoGO;
	float gDecGRtoGO; //derived from tau
	float gGABADecTauGOtoGO;
	float gGABADecGOtoGO; //derived from tau

	float mGluRDecayGO;
	float gMGluRIncDecayGO;
	float gMGluRDecGRtoGO;
	float gluDecayGO;
	float threshDecTauGO;
	float threshDecGO; //derived from tau

	//---------granule cell variables
	float eLeakGR;
	float eGOGR;
	float eMFGR;
	float threshMaxGR;
	float threshRestGR;

	float gIncDirectMFtoGR;
	float gDirectTauMFtoGR;
	float gDirectDecMFtoGR; //derived from tau
	float gIncFracSpilloverMFtoGR;
	float gSpilloverTauMFtoGR;
	float gSpilloverDecMFtoGR;
	float recoveryTauMF;
	float fracDepMF;

	float gIncDirectGOtoGR;
	float gDirectTauGOtoGR;
	float gDirectDecGOtoGR; //derived from tau
	float gIncFracSpilloverGOtoGR;
	float gSpilloverTauGOtoGR;
	float gSpilloverDecGOtoGR; //derived from tau
	
	float recoveryTauGO;
	float fracDepGO;


	float threshDecTauGR;
	float threshDecGR; //derived from tau
	float gLeakGR; //derived from raw values

	float msPerHistBinGR;
	ct_uint32_t tsPerHistBinGR; //derived from raw values

	
	
	//------------UBC variables
	float gIncMFtoUBC;
	float gIncGOtoUBC;
	float gIncUBCtoUBC;
	float gIncUBCtoGO;
	float gIncUBCtoGR;
	float gKIncUBC;
	float gKTauUBC;
	float gConstUBC;
	float threshTauUBC;

	float threshDecTauUBC;
	float threshDecUBC;
	
	//--------stellate cell variables
	float eLeakSC;
	float gLeakSC; //derived from raw values
	float gDecTauGRtoSC;
	float gDecGRtoSC; //derived from tau
	float threshMaxSC;
	float threshRestSC;
	float threshDecTauSC;
	float threshDecSC; //derived from tau
	float gIncGRtoSC;

	//***From mzone***

	//basket cell const values
	float eLeakBC;
	float ePCtoBC;
	float gLeakBC; //derived from raw values
	float gDecTauGRtoBC;
	float gDecGRtoBC; //derived from tau
	float gDecTauPCtoBC;
	float gDecPCtoBC; //derived from tau
	float threshDecTauBC;
	float threshDecBC; //derived from tau
	float threshRestBC;
	float threshMaxBC;
	float gIncGRtoBC;
	float gIncPCtoBC;

	//purkinje cell const values
	float initSynWofGRtoPC;
	float eLeakPC;
	float eBCtoPC;
	float eSCtoPC;
	float threshMaxPC;
	float threshRestPC;
	float threshDecTauPC;
	float threshDecPC; //derived from tau
	float gLeakPC; //derived from raw values
	float gDecTauGRtoPC;
	float gDecGRtoPC; //derived from tau
	float gDecTauBCtoPC;
	float gDecBCtoPC; //derived from tau
	float gDecTauSCtoPC;
	float gDecSCtoPC; //derived from tau
	float gIncSCtoPC;
	float gIncGRtoPC;
	float gIncBCtoPC;
	ct_uint32_t tsPopHistPC;
	ct_uint32_t tsPerPopHistBinPC;
	ct_uint32_t numPopHistBinsPC;

	float synLTPStepSizeGRtoPC;
	float synLTDStepSizeGRtoPC;

	//IO cell const values
	float coupleRiRjRatioIO;
	float eLeakIO;
	float eNCtoIO;
	float gLeakIO; //derived from raw values
	float gDecTSofNCtoIO;
	float gDecTTofNCtoIO;
	float gDecT0ofNCtoIO;
	float gIncNCtoIO;
	float gIncTauNCtoIO;
	float threshRestIO;
	float threshMaxIO;
	float threshDecTauIO;
	float threshDecIO; //derived from tau
	ct_uint32_t tsLTDDurationIO;
	ct_int32_t tsLTDStartAPIO;
	ct_int32_t tsLTPStartAPIO;
	ct_int32_t tsLTPEndAPIO;
	ct_uint32_t grPCHistCheckBinIO;
	float maxExtIncVIO;

	//nucleus cell const values
	float eLeakNC;
	float ePCtoNC;
	float gmaxNMDADecTauMFtoNC;
	float gmaxNMDADecMFtoNC; //derived from tau
	float gmaxAMPADecTauMFtoNC;
	float gmaxAMPADecMFtoNC; //derived from tau
	float gNMDAIncMFtoNC;
	float gAMPAIncMFtoNC;
	float gIncAvgPCtoNC;
	float gDecTauPCtoNC;
	float gDecPCtoNC; //derived from tau
	float gLeakNC; //derived from raw values
	float threshDecTauNC;
	float threshDecNC; //derived from tau
	float threshMaxNC;
	float threshRestNC;
	float relPDecTSofNCtoIO;
	float relPDecTTofNCtoIO;
	float relPDecT0ofNCtoIO;
	float relPIncNCtoIO;
	float relPIncTauNCtoIO;
	float initSynWofMFtoNC;
	float synLTDPCPopActThreshMFtoNC;
	float synLTPPCPopActThreshMFtoNC;
	float synLTDStepSizeMFtoNC;
	float synLTPStepSizeMFtoNC;

private:
	void updateParams();
	void updateParamsOriginal();
	void updateParamsV1();

	std::map<std::string, float> paramMap;

};

#endif /* ACTIVITYPARAMS_H_ */
