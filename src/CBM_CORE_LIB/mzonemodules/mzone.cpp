/*
 * mzone.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: consciousness
 */

#include "mzonemodules/mzone.h"

using namespace std;

MZone::MZone(ConnectivityParams *conParams, ActivityParams *actParams,
		MZoneConnectivityState *conState, MZoneActivityState *actState,
		int randSeed, ct_uint32_t **actBufGRGPU,
		ct_uint32_t **delayMaskGRGPU, ct_uint64_t **histGRGPU,
		int gpuIndStart, int numGPUs)
{
	cout << "constructor entered" << endl;
	
	randGen=new CRandomSFMT0(randSeed);

	cp=conParams;
	ap=actParams;
	cs=conState;
	as=actState;

	apBufGRGPU=actBufGRGPU;
	delayBCPCSCMaskGRGPU=delayMaskGRGPU;
	historyGRGPU=histGRGPU;

	pfSynWeightPCLinear=new float[cp->numGR];

	pfPCPlastStepIO=new float[cp->numIO];

	tempGRPCLTDStep=ap->synLTDStepSizeGRtoPC;
	tempGRPCLTPStep=ap->synLTPStepSizeGRtoPC;

	this->numGPUs=numGPUs;
	this->gpuIndStart=gpuIndStart;
	cout << " HERE" << endl;
	initCUDA();
}

MZone::~MZone()
{
	//clean up allocated memory
	delete randGen;

	delete[] pfSynWeightPCLinear;
	delete[] pfPCPlastStepIO;

//	//free cuda host memory
	cudaSetDevice(0+gpuIndStart);
	cudaFreeHost(inputSumPFPCMZH);
	cudaDeviceSynchronize();

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		//free cuda device memory
		cudaFree(pfSynWeightPCGPU[i]);
		cudaFree(inputPFPCGPU[i]);
		cudaFree(inputSumPFPCMZGPU[i]);
		cudaDeviceSynchronize();
	}

	delete[] pfSynWeightPCGPU;
	delete[] inputPFPCGPU;
	delete[] inputPFPCGPUPitch;
	delete[] inputSumPFPCMZGPU;
}

void MZone::initCUDA()
{
	int maxNumGPUs;
	cudaGetDeviceCount(&maxNumGPUs);

	numGRPerGPU=cp->numGR/numGPUs;

	updatePFPCNumGRPerB=512;
	updatePFPCNumBlocks=numGRPerGPU/updatePFPCNumGRPerB;

	updatePFPCSynWNumGRPerB=512*(cp->numpPCfromGRtoPC>512)+
			cp->numpPCfromGRtoPC*(cp->numpPCfromGRtoPC<=512);
	updatePFPCSynWNumBlocks=cp->numpPCfromGRtoPC/updatePFPCSynWNumGRPerB;

	cout << "here" << endl;
	cudaSetDevice(0+gpuIndStart);
	//allocate host cuda memory
	cudaHostAlloc((void **)&inputSumPFPCMZH, cp->numPC*sizeof(float), cudaHostAllocPortable);

	cudaDeviceSynchronize();
//	cudaMallocHost((void **)&inputSumPFPCMZH, numPC*sizeof(float));
	//initialize host cuda memory
	cout << "here" << endl;
	for(int i=0; i<cp->numPC; i++)
	{
		inputSumPFPCMZH[i]=0;
	}

	for(int i=0; i<cp->numPC; i++)
	{
		for(int j=0; j<cp->numpPCfromGRtoPC; j++)
		{
			pfSynWeightPCLinear[i*cp->numpPCfromGRtoPC+j]=as->pfSynWeightPC[i][j];
		}
	}
	cout << "here" << endl;
	pfSynWeightPCGPU=new float*[numGPUs];
	inputPFPCGPU=new float*[numGPUs];
	inputPFPCGPUPitch=new size_t[numGPUs];
	inputSumPFPCMZGPU=new float*[numGPUs];
	cout << "here" << endl;

	for(int i=0; i<numGPUs; i++)
	{
		int cpyStartInd;

		cpyStartInd=i*numGRPerGPU;

		cudaSetDevice(i+gpuIndStart);
		//allocate device cuda memory
		cudaMalloc((void **)&pfSynWeightPCGPU[i], numGRPerGPU*sizeof(float));
		cudaMallocPitch((void **)&inputPFPCGPU[i], (size_t *)&inputPFPCGPUPitch[i],
				cp->numpPCfromGRtoPC*sizeof(float), cp->numPC/numGPUs);
		cudaMalloc((void **)&inputSumPFPCMZGPU[i], cp->numPC/numGPUs*sizeof(float));

		cudaDeviceSynchronize();
		//initialize device cuda memory
		cudaMemcpy(pfSynWeightPCGPU[i], &pfSynWeightPCLinear[cpyStartInd],
				numGRPerGPU*sizeof(float), cudaMemcpyHostToDevice);
		for(int j=0; j<cp->numPC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFPCGPU[i]+j*inputPFPCGPUPitch[i]),
					0, cp->numpPCfromGRtoPC*sizeof(float));
		}
		cudaMemset(inputSumPFPCMZGPU[i], 0, cp->numPC/numGPUs*sizeof(float));

		cudaDeviceSynchronize();
	}
	cout << "here" << endl;
	
	testReduction();
	cout << "After test" << endl;

}

void MZone::writeToState()
{
	cpyPFPCSynWCUDA();

	for(int i=0; i<cp->numPC; i++)
	{
		as->inputSumPFPC[i]=inputSumPFPCMZH[i];
	}
}

void MZone::cpyPFPCSynWCUDA()
{
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMemcpy((void *)&pfSynWeightPCLinear[i*numGRPerGPU], pfSynWeightPCGPU[i],
				numGRPerGPU*sizeof(float), cudaMemcpyDeviceToHost);
	}

	for(int i=0; i<cp->numPC; i++)
	{
		for(int j=0; j<cp->numpPCfromGRtoPC; j++)
		{
			as->pfSynWeightPC[i][j]=pfSynWeightPCLinear[i*cp->numpPCfromGRtoPC+j];
		}
	}
}


void MZone::setErrDrive(float errDriveRelative)
{
	as->errDrive=errDriveRelative*ap->maxExtIncVIO;
#ifdef DEBUGOUT
	//cout<<"errDrive: "<<" "<<errDriveRelative<<" "<<ap->maxErrDriveIO<<" "<<as->errDrive<<endl;
#endif
}
void MZone::updateMFActivities(const ct_uint8_t *actMF)
{
	apMFInput=actMF;
}
void MZone::updateTrueMFs(bool *trueMF)
{
	isTrueMF=trueMF;
}
void MZone::updateSCActivities(const ct_uint8_t *actSC)
{
	apSCInput=actSC;
}
void MZone::updatePFBCSum(const ct_uint32_t *pfBCSum)
{
	sumPFBCInput=pfBCSum;
}

void MZone::calcPCActivities()
{
#pragma omp parallel
	{
	//float gSCtotal = 0;
	//float gBCtotal = 0;
#pragma omp for
		for(int i=0; i<cp->numPC; i++)
		{
			float gSCPCSum;

			as->gPFPC[i]=as->gPFPC[i]+inputSumPFPCMZH[i]*ap->gIncGRtoPC;
			as->gPFPC[i]=as->gPFPC[i]*ap->gDecGRtoPC;
			as->gBCPC[i]=as->gBCPC[i]+as->inputBCPC[i]*ap->gIncBCtoPC;
			as->gBCPC[i]=as->gBCPC[i]*ap->gDecBCtoPC;

			gSCPCSum=0;
			for(int j=0; j<cp->numpPCfromSCtoPC; j++)
			{
				as->gSCPC[i][j]=as->gSCPC[i][j]+ap->gIncSCtoPC*(1-as->gSCPC[i][j])*as->inputSCPC[i][j];
				as->gSCPC[i][j]=as->gSCPC[i][j]*ap->gDecSCtoPC;//GSCDECAYPC;
				gSCPCSum+=as->gSCPC[i][j];
			}

	
			as->vPC[i]=as->vPC[i]+
					(ap->gLeakPC*(ap->eLeakPC-as->vPC[i]))-
					(as->gPFPC[i]*as->vPC[i])+
					(as->gBCPC[i]*(ap->eBCtoPC-as->vPC[i]))+
					(gSCPCSum*(ap->eSCtoPC-as->vPC[i]));	
			as->threshPC[i]=as->threshPC[i]+(ap->threshDecPC*(ap->threshRestPC-as->threshPC[i]));

			as->apPC[i]=as->vPC[i]>as->threshPC[i];
			as->apBufPC[i]=(as->apBufPC[i]<<1)|(as->apPC[i]*0x00000001);

			as->threshPC[i]=as->apPC[i]*ap->threshMaxPC+(!as->apPC[i])*as->threshPC[i];
			as->pcPopAct=as->pcPopAct+as->apPC[i];
			//gSCtotal = gSCtotal + gSCPCSum;
			//gBCtotal = gBCtotal + as->gBCPC[i];
		}
	//cout << "SC/BC ratio: " << (gSCtotal/cp->numPC)/(gBCtotal/cp->numPC) << endl;
	
	}		
//	memset(inputBCPC, 0, numPC*sizeof(unsigned char)); TODO: update BC to PC out somewhere else?
//cout << inputSumPFPCMZH[1] << endl;
}


void MZone::calcBCActivities(ct_uint32_t **pfInput)
{


#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numBC; i++)
		{
			int totalPFInput=0;
			for(int j=0; j<numGPUs; j++)
			{
				totalPFInput+=pfInput[j][i];	
			}
			
			//if(i==0){ cout << totalPFInput << endl; }
			
			as->gPFBC[i]=as->gPFBC[i]+(sumPFBCInput[i]*ap->gIncGRtoBC);
			as->gPFBC[i]=as->gPFBC[i]*ap->gDecGRtoBC;
			as->gPCBC[i]=as->gPCBC[i]+(as->inputPCBC[i]*ap->gIncPCtoBC);
			as->gPCBC[i]=as->gPCBC[i]*ap->gDecPCtoBC;

			as->vBC[i]=as->vBC[i]+
					(ap->gLeakBC*(ap->eLeakBC-as->vBC[i]))-
					(as->gPFBC[i]*as->vBC[i])+
					(as->gPCBC[i]*(ap->ePCtoBC-as->vBC[i]));
			as->threshBC[i]=as->threshBC[i]+ap->threshDecBC*(ap->threshRestBC-as->threshBC[i]);
			as->apBC[i]=as->vBC[i]>as->threshBC[i];
			as->apBufBC[i]=(as->apBufBC[i]<<1)|(as->apBC[i]*0x00000001);

			as->threshBC[i]=as->apBC[i]*ap->threshMaxBC+(!as->apBC[i])*(as->threshBC[i]);
		}
	}
//	cout<<gPFBC[127]<<" "<<inputSumPFBCH[127]<<" "<<gPCBC[127]<<" "<<inputPCBC[127]<<vBC[127]<<apBC[127]<<endl;
//	cout<<"diagPFBCH: "<<pfbcsumdiag<<" diagPFBCGPU: "<<inputSumPFBCH[0]<<endl;

//	memset(inputPCBC, 0, numBC*sizeof(unsigned char));
//	memset(sumPFBCInput, 0, numBC*sizeof(unsigned short));

//cout << totalPFInput << endl;
}

void MZone::calcIOActivities()
{
#pragma omp parallel for
	clock_t t;
	t = clock();
	srand(t);
	
	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	float gNoise = (r - 0.5)*2.0;
	for(int i=0; i<cp->numIO; i++)
	{
		float gNCSum;
//		float gHMax;
//		float gHTau;
//		float gLtCaHMax;
//		float gLtCaM;

//		cout<<threshIO[i]<<" ";
		//calculate DCN input conductance
		gNCSum=0;
		for(int j=0; j<cp->numpIOfromNCtoIO; j++)
		{
			as->gNCIO[i][j]=as->gNCIO[i][j]*
					exp(-ap->msPerTimeStep/(-ap->gDecTSofNCtoIO*exp(-as->gNCIO[i][j]/ap->gDecTTofNCtoIO)+ap->gDecT0ofNCtoIO));
			as->gNCIO[i][j]=as->gNCIO[i][j]+as->inputNCIO[i][j]*ap->gIncNCtoIO*exp(-as->gNCIO[i][j]/ap->gIncTauNCtoIO);
			gNCSum=gNCSum+as->gNCIO[i][j];

			as->inputNCIO[i][j]=0;
		}

		//TODO: refactor 3.1, where did this come from?
		//TODO: refactor 1.5, where did this come from?
		gNCSum=1.5*gNCSum/3.1;

		as->vIO[i]=as->vIO[i]+
				ap->gLeakIO*(ap->eLeakIO-as->vIO[i])+
				gNCSum*(ap->eNCtoIO-as->vIO[i])+
				as->vCoupleIO[i]+
				as->errDrive + gNoise;

		as->apIO[i]=as->vIO[i]>as->threshIO[i];
		as->apBufIO[i]=(as->apBufIO[i]<<1)|(as->apIO[i]*0x00000001);

		as->threshIO[i]=ap->threshMaxIO*as->apIO[i]+
				(!as->apIO[i])*(as->threshIO[i]+ap->threshDecIO*(ap->threshRestIO-as->threshIO[i]));
//		cout<<vIO[i]<<" "<<threshIO[i]<<
//				" "<<inputNCIO[i][0]<<" "<<gNCSum<<" "<<apIO[i]<<" "<<threshDecayIO<<endl;

//		//calculate gH
//		gHMax=1/(1+exp((vIO[i]+GHMAXVIO)/8));
//		gHTau=exp(0.033*(vIO[i]+GHTAUVIO))/(0.011*(1+exp(0.083*(vIO[i]+GHTAUVIO))));
//		gHIO[i]=gHIO[i]+(gHMax-gHIO[i])/gHTau;
//
//		//calculate gLtCa
//		gLtCaHMax=1/(1+exp((vIO[i]+GLTCAHMAXVIO)/8.6));
//		gLtCaHIO[i]=gLtCaHIO[i]+(gLtCaHMax-gLtCaHIO[i])/GLTCAHTIO;
//		gLtCaM=1/powf(1+exp((GLTCAMMAXVIO-vIO[i])/4.2), 3);
//		gLtCaIO[i]=gLtCaM*gLtCaHIO[i]*4;
//
//		//calculate [Ca]
//		caIO[i]=caIO[i]+(gLtCaIO[i]>0)*(1-caIO[i])*gLtCaIO[i]*GLTCAHTIO;
//		caIO[i]=caIO[i]*CADECAYIO;
//
//		//calculate gKCa
//		gKCaIO[i]=gKCaIO[i]+(1-gKCaIO[i])*(caIO[i]-0.2)*0.05;
//		gKCaIO[i]=(!(gKCaIO[i]<0))*gKCaIO[i];
//
//		//calculate vm
//		vIO[i]=vIO[i]+GLEAKIO*(ELEAKIO-vIO[i])+
//			gHIO[i]*0.04*(EHIO-vIO[i])+
//			gNCSum*1.5*(ENCIO-vIO[i])+
//			gKCaIO[i]*0.1*(EKCAIO-vIO[i])+
//			gLtCaIO[i]*(ECAIO-vIO[i]);//+
////			MAXUSDR*(t==USONSET);
//
////		cout<<vIO[i]<<endl;
//		//calculate ap, thresh and [Ca]
//		apIO[i]=vIO[i]>threshIO[i];
//		threshIO[i]=-20*apIO[i]+(!apIO[i])*(threshIO[i]+(0.2*(THRESHBASEIO-threshIO[i])));
//		caIO[i]=caIO[i]+apIO[i]*(1-caIO[i])*0.2;
	}

	as->errDrive=0;
//	cout<<endl;

//	memset(inputNCIO, false, numIO*numNCInPerIO*sizeof(bool));
//	cout<<inputNCIO[0][0]<<endl;
}

void MZone::calcNCActivities()
{

float gDecay = exp(-1.0 / 20.0); 

#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numNC; i++)
		{
			float gMFNMDASum;
			float gMFAMPASum;
			float gPCNCSum;

			int inputPCNCSum;
			int inputMFNCSum;

			gMFNMDASum=0;
			gMFAMPASum=0;
			inputMFNCSum=0;
			for(int j=0; j<cp->numpNCfromMFtoNC; j++)
			{
				inputMFNCSum+=as->inputMFNC[i][j];
				
				
				
				
			/*	as->gmaxNMDAofMFtoNC[i][j]=as->gmaxNMDAofMFtoNC[i][j]*ap->gmaxNMDADecMFtoNC+
						as->inputMFNC[i][j]*as->mfSynWeightNC[i][j]*(1-as->gmaxNMDAofMFtoNC[i][j]);
				as->gMFNMDANC[i][j]=as->gMFNMDANC[i][j]+ap->gNMDAIncMFtoNC*(as->gmaxNMDAofMFtoNC[i][j]-as->gMFNMDANC[i][j]);
				gMFNMDASum=gMFNMDASum+as->gMFNMDANC[i][j];
				
				as->gmaxAMPAofMFtoNC[i][j]=as->gmaxAMPAofMFtoNC[i][j]*ap->gmaxAMPADecMFtoNC+
						as->inputMFNC[i][j]*as->mfSynWeightNC[i][j]*(1-as->gmaxAMPAofMFtoNC[i][j]);
				as->gMFAMPANC[i][j]=as->gMFAMPANC[i][j]+ap->gAMPAIncMFtoNC*(as->gmaxAMPAofMFtoNC[i][j]-as->gMFAMPANC[i][j]);
				gMFAMPASum=gMFAMPASum+as->gMFAMPANC[i][j];
				
*/

				as->gMFAMPANC[i][j] = as->gMFAMPANC[i][j]*gDecay + (ap->gAMPAIncMFtoNC*as->inputMFNC[i][j]*as->mfSynWeightNC[i][j]);
				gMFAMPASum=gMFAMPASum+as->gMFAMPANC[i][j];

				


			}
			gMFNMDASum=gMFNMDASum*ap->msPerTimeStep/((float)cp->numpNCfromMFtoNC);
			gMFAMPASum=gMFAMPASum*ap->msPerTimeStep/((float)cp->numpNCfromMFtoNC);
			gMFNMDASum=gMFNMDASum*as->vNC[i]/(-80.0f);

			gPCNCSum=0;
			inputPCNCSum=0;
			for(int j=0; j<cp->numpNCfromPCtoNC; j++)
			{
				inputPCNCSum+=as->inputPCNC[i][j];

				as->gPCNC[i][j]=as->gPCNC[i][j]*ap->gDecPCtoNC+as->inputPCNC[i][j]*ap->gIncAvgPCtoNC*(1-as->gPCNC[i][j]);
				gPCNCSum=gPCNCSum+as->gPCNC[i][j];
			}
			gPCNCSum=gPCNCSum*ap->msPerTimeStep/((float)cp->numpNCfromPCtoNC);
			//cout<<as->vNC[i]<<" "<<ap->gLeakNC*(ap->eLeakNC-as->vNC[i])<<" "<<0-(gMFNMDASum+gMFAMPASum)*as->vNC[i]<<" "<<gPCNCSum*(ap->ePCtoNC-as->vNC[i])<<" "<<inputMFNCSum<<" "<<inputPCNCSum<<endl;
			
			as->vNC[i]= as->vNC[i]+ap->gLeakNC*(ap->eLeakNC-as->vNC[i])-
					(gMFNMDASum+gMFAMPASum)*as->vNC[i]+gPCNCSum*(ap->ePCtoNC-as->vNC[i]);
			
			as->threshNC[i]=as->threshNC[i]+ap->threshDecNC*(ap->threshRestNC-as->threshNC[i]);
			as->apNC[i]=as->vNC[i]>as->threshNC[i];
			as->apBufNC[i]=(as->apBufNC[i]<<1)|(as->apNC[i]*0x00000001);

			as->threshNC[i]=as->apNC[i]*ap->threshMaxNC+(!as->apNC[i])*as->threshNC[i];
		}
	}

#ifdef DEBUGOUT
//	cout<<"NC[0] :"<<as->vNC[0]<<" "<<as->threshNC[0]<<" "<<as->apNC[0]<<" "
//			<<gMFNMDASum<<" "<<gMFAMPASum<<" "<<gPCNCSum<<" "<<endl;
#endif
//	cout<<"-----------"<<endl;

//	memset(inputMFNC, false, numNC*numMFInPerNC*sizeof(bool));
//	memset(inputPCNC, false, numNC*numPCInPerNC*sizeof(bool));
}

void MZone::updatePCOut()
{
#ifdef DEBUGOUT
	cout<<"resetting inputPCBC "<<cp->numBC<<endl;
#endif
	for(int i=0; i<cp->numBC; i++)
	{
		as->inputPCBC[i]=0;
	}
#ifdef DEBUGOUT
	cout<<"updating pc to bc "<<endl;
#endif
	for(int i=0; i<cp->numPC; i++)
	{
		for(int j=0; j<cp->numpPCfromPCtoBC; j++)
		{
#ifdef DEBUGOUT
			cout<<"i "<<i<<" j "<<j<<endl;
#endif
			as->inputPCBC[cs->pPCfromPCtoBC[i][j]]+=as->apPC[i];
		}
	}
#ifdef DEBUGOUT
	cout<<"updating pc to nc "<<endl;
#endif
	for(int i=0; i<cp->numNC; i++)
	{
		for(int j=0; j<cp->numpNCfromPCtoNC; j++)
		{
#ifdef DEBUGOUT
			cout<<"i "<<i<<" j "<<j<<" cs->pNCfromPCtoNC[i][j]"<<cs->pNCfromPCtoNC[i][j]<<endl;
#endif
			as->inputPCNC[i][j]=as->apPC[cs->pNCfromPCtoNC[i][j]];
		}
	}
#ifdef DEBUGOUT
	cout<<"finished "<<endl;
#endif
}

void MZone::updateBCPCOut()
{
	for(int i=0; i<cp->numPC; i++)
	{
		as->inputBCPC[i]=0;
	}
	for(int i=0; i<cp->numBC; i++)
	{
		if(as->apBC[i])
		{
			for(int j=0; j<cp->numpBCfromBCtoPC; j++)
			{
				as->inputBCPC[cs->pBCfromBCtoPC[i][j]]++;
			}
		}
	}
}

void MZone::updateSCPCOut()
{
#pragma omp parallel for
	for(int i=0; i<cp->numPC; i++)
	{
		for(int j=0; j<cp->numpPCfromSCtoPC; j++)
		{
			as->inputSCPC[i][j]=apSCInput[cs->pPCfromSCtoPC[i][j]];
		}
	}
}

void MZone::updateIOOut()
{
	for(int i=0; i<cp->numIO; i++)
	{
//		as->pfPCPlastTimerIO[i]=(!as->apIO[i])*(as->pfPCPlastTimerIO[i]+1)+as->apIO[i]*(-100);
		as->pfPCPlastTimerIO[i]=(!as->apIO[i])*(as->pfPCPlastTimerIO[i]+1)+as->apIO[i]*ap->tsLTPEndAPIO;
		as->vCoupleIO[i]=0;
		for(int j=0; j<cp->numpIOInIOIO; j++)
		{
			as->vCoupleIO[i]=as->vCoupleIO[i]+ap->coupleRiRjRatioIO*(as->vIO[cs->pIOInIOIO[i][j]]-as->vIO[i]);
		}

//		as->vIO[i]=as->vIO[i]+as->vCoupleIO[i];
	
	
	}
}

void MZone::updateNCOut()
{
	for(int i=0; i<cp->numNC; i++)
	{
		as->synIOPReleaseNC[i]=as->synIOPReleaseNC[i]*
				exp(-ap->msPerTimeStep/(ap->relPDecTSofNCtoIO*exp(-as->synIOPReleaseNC[i]/ap->relPDecTTofNCtoIO)+ap->relPDecT0ofNCtoIO));
		as->synIOPReleaseNC[i]=as->synIOPReleaseNC[i]+as->apNC[i]*ap->relPIncNCtoIO*
				exp(-as->synIOPReleaseNC[i]/ap->relPIncTauNCtoIO);
	}

	for(int i=0; i<cp->numIO; i++)
	{
		for(int j=0; j<cp->numpIOfromNCtoIO; j++)
		{
			as->inputNCIO[i][j]=(randGen->Random()<as->synIOPReleaseNC[cs->pIOfromNCtoIO[i][j]]);
		}
	}
}

void MZone::updateMFNCOut()
{
	for(int i=0; i<cp->numNC; i++)
	{
		for(int j=0; j<cp->numpNCfromMFtoNC; j++)
		{
//			cout<<"i "<<i<<" j "<<j<<" pNCfromMFtoNC[i][j] "<<cs->pNCfromMFtoNC[i][j]<<endl;
			as->inputMFNC[i][j]=apMFInput[cs->pNCfromMFtoNC[i][j]];
		}
	}
}

void MZone::updateMFNCSyn(const ct_uint8_t *histMF, unsigned long t)
{
	
	bool reset;
	float avgAllAPPC;
	bool doLTD;
	bool doLTP;

#ifdef DEBUGOUT
	float sumSynW;
#endif
	reset=(t%ap->tsPerPopHistBinPC==0);
	if(!reset)
	{
		return;
	}
	histMFInput=histMF;

	as->histPCPopActSum=(as->histPCPopActSum)-(as->histPCPopAct[as->histPCPopActCurBinN])+(as->pcPopAct);
	as->histPCPopAct[as->histPCPopActCurBinN]=as->pcPopAct;
	as->pcPopAct=0;
	as->histPCPopActCurBinN++;
	as->histPCPopActCurBinN=as->histPCPopActCurBinN%ap->numPopHistBinsPC;

	avgAllAPPC=((float)as->histPCPopActSum)/ap->numPopHistBinsPC;

#ifdef DEBUGOUT
	cout<<"avgAllAPPC: "<<avgAllAPPC<<" "<<endl;
#endif

	doLTD=false;
	doLTP=false;
	if(avgAllAPPC>=ap->synLTDPCPopActThreshMFtoNC && !as->noLTDMFNC)
	{
		doLTD=true;
		as->noLTDMFNC=true;
	}
	else if(avgAllAPPC<ap->synLTDPCPopActThreshMFtoNC)
	{
		as->noLTDMFNC=false;
	}

	if(avgAllAPPC<=ap->synLTPPCPopActThreshMFtoNC && !as->noLTPMFNC)
	{
		doLTP=true;
		as->noLTPMFNC=true;
	}
	else if(avgAllAPPC>ap->synLTPPCPopActThreshMFtoNC)
	{
		as->noLTPMFNC=false;
	}

//	cout<<"MFNC plasticity: "<<avgAllAPPC<<" "<<doLTP<<" "<<doLTD<<endl;
#ifdef DEBUGOUT
	sumSynW=0;
#endif
	for(int i=0; i<cp->numNC; i++)
	{
		for(int j=0; j<cp->numpNCfromMFtoNC; j++)
		{
			float synWDelta;
			synWDelta=histMFInput[cs->pNCfromMFtoNC[i][j]]*(doLTD*ap->synLTDStepSizeMFtoNC+doLTP*ap->synLTPStepSizeMFtoNC);
			as->mfSynWeightNC[i][j]=as->mfSynWeightNC[i][j]+synWDelta;
			as->mfSynWeightNC[i][j]=(as->mfSynWeightNC[i][j]>0)*as->mfSynWeightNC[i][j];
			as->mfSynWeightNC[i][j]=(as->mfSynWeightNC[i][j]<=1)*as->mfSynWeightNC[i][j]+(as->mfSynWeightNC[i][j]>1);
			//Now uses isTrueMF to take collaterals into account
			as->mfSynWeightNC[i][j]=as->mfSynWeightNC[i][j]*isTrueMF[cs->pNCfromMFtoNC[i][j]];
#ifdef DEBUGOUT
			sumSynW+=as->mfSynWeightNC[i][j];
#endif
		}
//		cout<<endl<<mfSynWChangeNC[i][0]<<" "<<mfSynWeightNC[i][0]<<endl;
	}
#ifdef DEBUGOUT
	cout<<sumSynW/(cp->numMF)<<endl;
#endif
//	cout<<as->mfSynWeightNC[0][0]<<" "<<as->mfSynWeightNC[0][1]<<" "<<as->mfSynWeightNC[0][2]<<" "<<as->mfSynWeightNC[0][3]<<endl;
}

void MZone::runPFPCOutCUDA(cudaStream_t **sts, int streamN)
{
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		callUpdatePFPCOutKernel(sts[i][streamN], updatePFPCNumBlocks, updatePFPCNumGRPerB,
				apBufGRGPU[i], delayBCPCSCMaskGRGPU[i],
				pfSynWeightPCGPU[i], inputPFPCGPU[i], inputPFPCGPUPitch[i], cp->numpPCfromGRtoPCP2);
	}
//	callUpdatePFPCOutKernel<numPFInPerPC, 1024, 1024>(st, apBufGRGPU, delayBCPCSCMaskGRGPU, pfSynWeightPCGPU, inputPFPCGPU, inputPFPCGPUPitch);
}

void MZone::runPFPCSumCUDA(cudaStream_t **sts, int streamN)
{
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
//	callSumKernel<float, 512, true, false>
//	(sts[i][streamN], inputPFPCGPU[i], inputPFPCGPUPitch[i],
//			inputSumPFPCMZGPU[i], 1, numPC/numGPUs, 1, numPFInPerPC);
		callSumKernel<float, true, false>(sts[i][streamN], inputPFPCGPU[i], inputPFPCGPUPitch[i],
				inputSumPFPCMZGPU[i], 1, cp->numPC/numGPUs, 1, cp->numpPCfromGRtoPC);
	}
}

void MZone::cpyPFPCSumCUDA(cudaStream_t **sts, int streamN)
{
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMemcpyAsync(&inputSumPFPCMZH[cp->numPC*i/numGPUs], inputSumPFPCMZGPU[i],
				cp->numPC/numGPUs*sizeof(float),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void MZone::runPFPCPlastCUDA(cudaStream_t **sts, int streamN, unsigned long t)
{
	cudaError_t error;
	if(t%ap->tsPerHistBinGR==0)
	{
		int curGROffset;
		int curGPUInd;
		int curIOInd;

		int numGRPerIO;

		curGROffset=0;
		curGPUInd=0;
		curIOInd=0;

		numGRPerIO=cp->numGR/cp->numIO;

		for(int i=0; i<cp->numIO; i++)
		{
		
			
			
			
			
			if(as->pfPCPlastTimerIO[i]<(ap->tsLTDStartAPIO+((int)(ap->tsLTDDurationIO))) && as->pfPCPlastTimerIO[i] >= ap->tsLTDStartAPIO)
			{
				pfPCPlastStepIO[i]=tempGRPCLTDStep;//ap->synLTDStepSizeGRtoPC;
			}
			else if(as->pfPCPlastTimerIO[i]>=ap->tsLTPStartAPIO || as->pfPCPlastTimerIO[i]<ap->tsLTPEndAPIO)
			{
				pfPCPlastStepIO[i]=tempGRPCLTPStep;//ap->synLTPStepSizeGRtoPC;
			}
			else
			{
				pfPCPlastStepIO[i]=0;
			}
		}

#ifdef DEBUGOUT
		cout<<"pfPCPlastStepiO[0] "<<pfPCPlastStepIO[0]<<" "<<as->pfPCPlastTimerIO[0]<<endl;
#endif
		error=cudaSetDevice(curGPUInd+gpuIndStart);
//		cerr<<"runPFPCPlastCUDA: switching to gpu #"<<curGPUInd<<
//				" :"<<cudaGetErrorString(error)<<endl;
		for(int i=0; i<cp->numGR; i+=cp->numpPCfromGRtoPC)
		{
//			cerr<<"i:"<<i<<endl;
			if(i>=(curGPUInd+1)*numGRPerGPU)
			{
				curGPUInd++;
				curGROffset=0;
				error=cudaSetDevice(curGPUInd+gpuIndStart);
//				cerr<<"runPFPCPlastCUDA: switching to gpu #"<<curGPUInd<<
//						" :"<<cudaGetErrorString(error)<<endl;
			}
//			cerr<<"currentGPUInd: "<<curGPUInd<<endl;
//			cerr<<"currentGROffset: "<<curGROffset<<endl;
			if(i>=(curIOInd+1)*numGRPerIO)
			{
				curIOInd++;
			}
//			cerr<<"currentIOInd: "<<curIOInd<<endl;
//			cerr<<pfPCPlastStepIO[curIOInd]<<endl;
//			if(pfPCPlastStepIO[curIOInd]!=0)
//			{
			callUpdatePFPCPlasticityIOKernel(sts[curGPUInd][streamN+curIOInd],
					updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB, pfSynWeightPCGPU[curGPUInd],
					historyGRGPU[curGPUInd], ap->grPCHistCheckBinIO, curGROffset, pfPCPlastStepIO[curIOInd]);
//			}
//			error=cudaGetLastError();
//			cerr<<"runPFPCPlastCUDA: kernel launch for gpu #"<<curGPUInd<<
//					" :"<<cudaGetErrorString(error)<<endl;

			curGROffset=curGROffset+cp->numpPCfromGRtoPC;
		}
//		for(int i=0; i<numGPUs; i++)
//		{
//			cudaSetDevice(i+gpuIndStart);
//
////			for(int j=0; j<numIO; j++)
////			{
////				callUpdatePFPCPlasticityIOKernel(sts[i], pfPlastTimerIO[i], pfSynWeightPCGPU, historyGRGPU, i*numGR/numIO, pfPCLTDDecPF, pfPCLTPIncPF);
////			}
//		}
	}
}

void MZone::setGRPCPlastSteps(float ltdStep, float ltpStep)
{
	tempGRPCLTDStep=ltdStep;
	tempGRPCLTPStep=ltpStep;
}

void MZone::resetGRPCPlastSteps()
{
	tempGRPCLTDStep=ap->synLTDStepSizeGRtoPC;
	tempGRPCLTPStep=ap->synLTPStepSizeGRtoPC;
}

const float* MZone::exportPFPCWeights(){
	cpyPFPCSynWCUDA();
	return (const float *)pfSynWeightPCLinear; 
}

const ct_uint8_t* MZone::exportAPNC()
{
	return (const ct_uint8_t *)as->apNC;
}
const ct_uint8_t* MZone::exportAPBC()
{
	return (const ct_uint8_t *)as->apBC;
}
const ct_uint8_t* MZone::exportAPPC()
{
	return (const ct_uint8_t *)as->apPC;
}
const ct_uint8_t* MZone::exportAPIO()
{
	return (const ct_uint8_t *)as->apIO;
}

const float* MZone::exportgBCPC()
{
	return (const float *)as->gBCPC;
}
const float* MZone::exportgPFPC()
{
	return (const float *)as->gPFPC;
}

const float* MZone::exportVmBC()
{
	return (const float *)as->vBC;
}
const float* MZone::exportVmPC()
{
	return (const float *)as->vPC;
}
const float* MZone::exportVmNC()
{
	return (const float *)as->vNC;
}
const float* MZone::exportVmIO()
{
	return (const float *)as->vIO;
}

const unsigned int* MZone::exportAPBufBC()
{
	return (const unsigned int *)as->apBufBC;
}
const ct_uint32_t* MZone::exportAPBufPC()
{
	return (const ct_uint32_t *)as->apBufPC;
}
const ct_uint32_t* MZone::exportAPBufIO()
{
	return (const ct_uint32_t *)as->apBufIO;
}
const ct_uint32_t* MZone::exportAPBufNC()
{
	return (const ct_uint32_t *)as->apBufNC;
}

void MZone::testReduction()
{
	cudaError_t error;

	float *hostTestData;

	float *hostPCSum;
	float *hostBCSum;
	float *hostSCSum;


//	float **gpuTestData;
//	float *gpuToHostTestData;

	float **gpuPCTestData;
	float **gpuBCTestData;
	float **gpuSCTestData;
	size_t *gpuPCP;
	size_t *gpuBCP;
	size_t *gpuSCP;
	float **gpuPCSum;
	float **gpuBCSum;
	float **gpuSCSum;

	float *gpuToHostPCSum;
	float *gpuToHostBCSum;
	float *gpuToHostSCSum;

	cudaStream_t *sts;

	hostTestData=new float[cp->numGR];

	hostPCSum=new float[cp->numPC];
	hostBCSum=new float[cp->numBC];
	hostSCSum=new float[cp->numSC];

	gpuToHostPCSum=new float[cp->numPC];
	gpuToHostBCSum=new float[cp->numBC];
	gpuToHostSCSum=new float[cp->numSC];

//	gpuTestData=new float*[numGPUs];
//	gpuToHostTestData=new float[cp->numGR];


	gpuPCTestData=new float*[numGPUs];
	gpuBCTestData=new float*[numGPUs];
	gpuSCTestData=new float*[numGPUs];

	gpuPCP=new size_t[numGPUs];
	gpuBCP=new size_t[numGPUs];
	gpuSCP=new size_t[numGPUs];

	gpuPCSum=new float*[numGPUs];
	gpuBCSum=new float*[numGPUs];
	gpuSCSum=new float*[numGPUs];

	for(int i=0; i<cp->numGR; i++)
	{
		hostTestData[i]=randGen->Random();
	}

	cout<<"numGPUs "<<numGPUs<<" numGRPerGPU "<<numGRPerGPU<<endl;

//	for(int i=0; i<numGPUs; i++)
//	{
//		cudaSetDevice(i+gpuIndStart);
//
//		cudaMalloc(&gpuTestData[i], cp->numGR*sizeof(float));
//
//		cudaDeviceSynchronize();
//
//		cudaMemcpy(gpuTestData[i], &hostTestData[i*numGRPerGPU],
//				numGRPerGPU*sizeof(float), cudaMemcpyHostToDevice);
//
//		cudaDeviceSynchronize();
//
//		cudaMemcpy(&gpuToHostTestData[i*numGRPerGPU], gpuTestData[i],
//				numGRPerGPU*sizeof(float), cudaMemcpyDeviceToHost);
//
//		cudaDeviceSynchronize();
//	}
//
//	for(int i=0; i<20; i++)
//	{
//		cout<<i<<" "<<hostTestData[i]<<" "<<gpuToHostTestData[i]<<endl;
//	}


	sts=new cudaStream_t[numGPUs];

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);

		cudaStreamCreate(&sts[i]);

		cudaMallocPitch(&gpuPCTestData[i], &gpuPCP[i],
				cp->numpPCfromGRtoPC*sizeof(float), cp->numPC/numGPUs);
		cudaMallocPitch(&gpuBCTestData[i], &gpuBCP[i],
				cp->numpBCfromGRtoBC*sizeof(float), cp->numBC/numGPUs);
		cudaMallocPitch(&gpuSCTestData[i], &gpuSCP[i],
				cp->numpSCfromGRtoSC*sizeof(float), cp->numSC/numGPUs);

		cudaMalloc(&gpuPCSum[i], cp->numPC/numGPUs*sizeof(float));
		cudaMalloc(&gpuBCSum[i], cp->numBC/numGPUs*sizeof(float));
		cudaMalloc(&gpuSCSum[i], cp->numSC/numGPUs*sizeof(float));

		error=cudaGetLastError();
		cout<<"allocating for gpu "<<i<<" "<<cudaGetErrorString(error)<<endl;

		cudaDeviceSynchronize();

	}

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);

		for(int j=0; j<cp->numPC/numGPUs; j++)
		{
			cudaMemcpy(((char *)gpuPCTestData[i]+j*gpuPCP[i]),
					&hostTestData[i*numGRPerGPU+j*cp->numpPCfromGRtoPC],
					cp->numpPCfromGRtoPC*sizeof(float), cudaMemcpyHostToDevice);
		}

		for(int j=0; j<cp->numBC/numGPUs; j++)
		{
			cudaMemcpy(((char *)gpuBCTestData[i]+j*gpuBCP[i]),
					&hostTestData[i*numGRPerGPU+j*cp->numpBCfromGRtoBC],
					cp->numpBCfromGRtoBC*sizeof(float), cudaMemcpyHostToDevice);
		}

		for(int j=0; j<cp->numSC/numGPUs; j++)
		{
			cudaMemcpy(((char *)gpuSCTestData[i]+j*gpuSCP[i]),
					&hostTestData[i*numGRPerGPU+j*cp->numpSCfromGRtoSC],
					cp->numpSCfromGRtoSC*sizeof(float), cudaMemcpyHostToDevice);
		}

		error=cudaGetLastError();
		cout<<"copying memory for gpu "<<i<<" "<<cudaGetErrorString(error)<<endl;

		cudaDeviceSynchronize();
	}

	for(int i=0; i<cp->numPC; i++)
	{
		hostPCSum[i]=0;

		for(int j=0; j<cp->numpPCfromGRtoPC; j++)
		{
			hostPCSum[i]+=hostTestData[i*cp->numpPCfromGRtoPC+j];
		}
	}

	for(int i=0; i<cp->numBC; i++)
	{
		hostBCSum[i]=0;

		for(int j=0; j<cp->numpBCfromGRtoBC; j++)
		{
			hostBCSum[i]+=hostTestData[i*cp->numpBCfromGRtoBC+j];
		}
	}

	for(int i=0; i<cp->numSC; i++)
	{
		hostSCSum[i]=0;

		for(int j=0; j<cp->numpSCfromGRtoSC; j++)
		{
			hostSCSum[i]+=hostTestData[i*cp->numpSCfromGRtoSC+j];
		}
	}

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		callSumKernel<float, true, false>(sts[i], gpuPCTestData[i], gpuPCP[i],
				gpuPCSum[i], 1, cp->numPC/numGPUs, 1, cp->numpPCfromGRtoPC);

		callSumKernel<float, true, false>(sts[i], gpuBCTestData[i], gpuBCP[i],
				gpuBCSum[i], 1, cp->numBC/numGPUs, 1, cp->numpBCfromGRtoBC);

		callSumKernel<float, true, false>(sts[i], gpuSCTestData[i], gpuSCP[i],
				gpuSCSum[i], 1, cp->numSC/numGPUs, 1, cp->numpSCfromGRtoSC);

		cudaDeviceSynchronize();

		error=cudaGetLastError();
		cout<<"calling sum kernels for gpu "<<i<<" "<<cudaGetErrorString(error)<<endl;
	}


	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);

		cudaMemcpy(&gpuToHostPCSum[i*cp->numPC/numGPUs], gpuPCSum[i],
				32*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gpuToHostBCSum[i*cp->numBC/numGPUs], gpuBCSum[i],
				cp->numBC/numGPUs*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gpuToHostSCSum[i*cp->numSC/numGPUs], gpuSCSum[i],
				cp->numSC/numGPUs*sizeof(float), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
	}

	cout<<cp->numPC/numGPUs<<" "<<cp->numBC/numGPUs<<" "<<cp->numSC/numGPUs<<endl;

	/*
	cout<<"PCs: "<<endl;
	for(int i=0; i<cp->numPC; i++)
	{
		cout<<i<<" H: "<<hostPCSum[i]<<" D: "<<gpuToHostPCSum[i]<<endl;
	}

	cout<<"BCs: "<<endl;
	for(int i=0; i<cp->numBC; i++)
	{
		cout<<i<<" H: "<<hostBCSum[i]<<" D: "<<gpuToHostBCSum[i]<<endl;
	}

	cout<<"SCs: "<<endl;
	for(int i=0; i<cp->numSC; i++)
	{
		cout<<i<<" H: "<<hostSCSum[i]<<" D: "<<gpuToHostSCSum[i]<<endl;
	}*/

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaFree(gpuPCTestData[i]);
		cudaFree(gpuBCTestData[i]);
		cudaFree(gpuSCTestData[i]);
	
	/***********************************************************NEW******************************/
		cudaFree(gpuPCSum[i]);
		cudaFree(gpuBCSum[i]);
		cudaFree(gpuSCSum[i]);
	/***********************************************************NEW******************************/

		
	}

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaStreamDestroy(sts[i]);
	}
	delete[] sts;

	delete[] hostTestData;
	delete[] hostPCSum;
	delete[] hostBCSum;
	delete[] hostSCSum;

	delete[] gpuToHostPCSum;
	delete[] gpuToHostBCSum;
	delete[] gpuToHostSCSum;

	delete[] gpuPCTestData;
	delete[] gpuBCTestData;
	delete[] gpuSCTestData;

	delete[] gpuPCP;
	delete[] gpuSCP;
	delete[] gpuBCP;

	delete[] gpuPCSum;
	delete[] gpuSCSum;
	delete[] gpuBCSum;

}


