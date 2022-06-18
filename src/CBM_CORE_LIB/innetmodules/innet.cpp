/*
 * innet.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#include "innetmodules/innet.h"

using namespace std;

InNet::InNet() {}

InNet::InNet(ActivityParams &ap, InNetConnectivityState *cs, InNetActivityState *as,
		int gpuIndStart, int numGPUs)
{
   	this->ap = ap; /* thas a deep copy! */
	// TODO: deep copy below?
	this->cs = cs; /* this is a shallow copy *yikes* */
	this->as = as; /* as is this :/ */

	this->gpuIndStart = gpuIndStart;
	this->numGPUs	  = numGPUs;

	// why do we allocate these here???
	gGOGRT = allocate2DArray<float>(MAX_NUM_P_GR_FROM_GO_TO_GR, NUM_GR);
	gMFGRT = allocate2DArray<float>(MAX_NUM_P_GR_FROM_MF_TO_GR, NUM_GR);

	pGRDelayfromGRtoGOT = allocate2DArray<ct_uint32_t>(MAX_NUM_P_GR_FROM_GR_TO_GO, NUM_GR);
	pGRfromMFtoGRT		= allocate2DArray<ct_uint32_t>(MAX_NUM_P_GR_FROM_MF_TO_GR, NUM_GR);
	pGRfromGOtoGRT		= allocate2DArray<ct_uint32_t>(MAX_NUM_P_GR_FROM_GO_TO_GR, NUM_GR);
	
	pGRfromGRtoGOT = allocate2DArray<ct_uint32_t>(MAX_NUM_P_GR_FROM_GR_TO_GO, NUM_GR);
	
	apBufGRHistMask = (1<<(ap.tsPerHistBinGR))-1;

	sumGRInputGO 		   = new ct_uint32_t[NUM_GO];
	sumInputGOGABASynDepGO = new float[NUM_GO];

	initCUDA();
}

InNet::~InNet()
{
	std::cout << "**********************************************CUDA DESTRUCTOR ENTERED**********************************************" << std::endl;

	//gpu related host variables
	cudaFreeHost(outputGRH);
	cudaFreeHost(inputSumPFBCH);
	cudaFreeHost(inputSumPFSCH);

	//gpu variables
	std::cout << "numGPUs: " << numGPUs << std::endl;

	for (int i = 0; i < numGPUs; i++)
	{
		cout << i+gpuIndStart << endl;
		cudaSetDevice(i+gpuIndStart);
		
		cudaDeviceSynchronize();
	
		//mf variables
		cudaFreeHost(apMFH[i]);
		cudaFreeHost(depAmpMFH[i]);
		cudaFree(apMFGPU[i]);
		cudaFree(depAmpMFGPU[i]);
		//GR variables
		cudaFree(outputGRGPU[i]);
		cudaFree(vGRGPU[i]);
		cudaFree(gKCaGRGPU[i]);
		cudaFree(depAmpMFGRGPU[i]);
		cudaFree(gLeakGRGPU[i]);
		cudaFree(gNMDAGRGPU[i]);
		cudaFree(gNMDAIncGRGPU[i]);
		cudaFree(gEGRGPU[i]);
		cudaFree(gEGRSumGPU[i]);
		cudaFree(gEDirectGPU[i]);
		cudaFree(gESpilloverGPU[i]);
		cudaFree(apMFtoGRGPU[i]);
		cudaFree(numMFperGR[i]);
		cudaFree(gIGRGPU[i]);
		cudaFree(gIGRSumGPU[i]);
		cudaFree(gIDirectGPU[i]);
		cudaFree(gISpilloverGPU[i]);
		cudaFree(apBufGRGPU[i]);
		cudaFree(apGRGPU[i]);
		cudaFree(threshGRGPU[i]);
		cudaFree(delayGOMasksGRGPU[i]);
		cudaFree(delayBCPCSCMaskGRGPU[i]);
		
		cudaFree(delayBCMasksGRGPU[i]);
		cudaFree(grConGROutBCGPU[i]);
		cudaFree(numBCOutPerGRGPU[i]);
		
		cudaFree(numGOOutPerGRGPU[i]);
		cudaFree(grConGROutGOGPU[i]);
		cudaFree(numGOInPerGRGPU[i]);
		cudaFree(grConGOOutGRGPU[i]);
		cudaFree(numMFInPerGRGPU[i]);
		
		cudaFree(grConMFOutGRGPU[i]);
		
		cudaFree(historyGRGPU[i]);

		//GO variables
		cudaFreeHost(apGOH[i]);
		cudaFree(apGOGPU[i]);
		cudaFree(grInputGOGPU[i]);
		cudaFree(grInputGOSumGPU[i]);
		cudaFreeHost(grInputGOSumH[i]);
		cudaFreeHost(depAmpGOH[i]);
		cudaFree(depAmpGOGPU[i]);
		cudaFree(depAmpGOGRGPU[i]);
		cudaFreeHost(dynamicAmpGOH[i]);
		cudaFree(dynamicAmpGOGPU[i]);
		cudaFree(dynamicAmpGOGRGPU[i]); 

		//BC variables
		cudaFree(grInputBCGPU[i]);
		cudaFree(grInputBCSumGPU[i]);
		cudaFreeHost(grInputBCSumH[i]);
		
		cudaFree(inputPFBCGPU[i]);
		cudaFree(inputSumPFBCGPU[i]);

		//SC variables
		cudaFree(inputPFSCGPU[i]);
		cudaFree(inputSumPFSCGPU[i]);
		//end gpu variables

		cudaDeviceSynchronize();

	cout << "***************GPU DELETED************" << endl;
	
	}

	//mf
	delete[] apMFH;
	delete[] apMFGPU;
	delete[] depAmpMFH;
	delete[] depAmpMFGPU;
	delete[] depAmpMFGRGPU;
	
	//gr
	delete2DArray<float>(gMFGRT);
	delete2DArray<float>(gGOGRT);
	delete2DArray<ct_uint32_t>(pGRDelayfromGRtoGOT);
	delete2DArray<ct_uint32_t>(pGRfromMFtoGRT);
	delete2DArray<ct_uint32_t>(pGRfromGOtoGRT);
	delete2DArray<ct_uint32_t>(pGRfromGRtoGOT);
	
	delete[] gEGRGPU;
	delete[] gEGRGPUP;
	delete[] gEGRSumGPU;
	delete[] gEDirectGPU;
	delete[] gESpilloverGPU;
	delete[] apMFtoGRGPU;
	delete[] numMFperGR;
	

	delete[] gIGRGPU;
	delete[] gIGRGPUP;
	delete[] gIGRSumGPU;
	delete[] gIDirectGPU;
	delete[] gISpilloverGPU;

	delete[] apBufGRGPU;
	delete[] outputGRGPU;
	delete[] apGRGPU;

	delete[] threshGRGPU;
	delete[] vGRGPU;
	delete[] gKCaGRGPU;
	delete[] gLeakGRGPU;
	delete[] gNMDAGRGPU;
	delete[] gNMDAIncGRGPU;
	delete[] historyGRGPU;

	delete[] delayGOMasksGRGPU;
	delete[] delayGOMasksGRGPUP;
	delete[] delayBCPCSCMaskGRGPU;

	delete[] delayBCMasksGRGPU;
	delete[] delayBCMasksGRGPUP;
	delete[] grConGROutBCGPU;
	delete[] grConGROutBCGPUP;
	delete[] numBCOutPerGRGPU;
	
	delete[] numGOOutPerGRGPU;
	delete[] grConGROutGOGPU;
	delete[] grConGROutGOGPUP;

	delete[] numGOInPerGRGPU;
	delete[] grConGOOutGRGPU;
	delete[] grConGOOutGRGPUP;

	delete[] numMFInPerGRGPU;
	delete[] grConMFOutGRGPU;
	delete[] grConMFOutGRGPUP;

	delete[] grInputGOSumH;
	delete[] grInputBCSumH;

	//go
	delete[] apGOH;
	delete[] apGOGPU;
	delete[] grInputGOGPU;
	delete[] grInputGOGPUP;
	delete[] grInputGOSumGPU;
	delete[] sumGRInputGO;
	delete[] sumInputGOGABASynDepGO;
	delete[] depAmpGOH;
	delete[] depAmpGOGPU;
	delete[] depAmpGOGRGPU;
	delete[] dynamicAmpGOH;
	delete[] dynamicAmpGOGPU;
	delete[] dynamicAmpGOGRGPU;

	//bc
	delete[] grInputBCGPU;
	delete[] grInputBCGPUP;
	delete[] grInputBCSumGPU;
	
	delete[] inputPFBCGPU;
	delete[] inputPFBCGPUP;
	delete[] inputSumPFBCGPU;

	//sc
	delete[] inputPFSCGPU;
	delete[] inputPFSCGPUP;
	delete[] inputSumPFSCGPU;

	delete[] plasScalerEx;
	delete[] plasScalerInh;
	delete2DArray<float>(goExScaler);	
	delete2DArray<float>(goInhScaler);	
	delete2DArray<float>(goFRArray);	
}

void InNet::writeToState()
{
	cudaError_t error;
	//GR variables
	// WARNING THIS IS A HORRIBLE IDEA. IF YOU GET BUGS CONSIDER THIS!
	// Reason: the apGR is a unique_ptr. it should only be modifed in the scope
	// that it is defined in.
	getGRGPUData<ct_uint8_t>(outputGRGPU, as->apGR.get());
	getGRGPUData<ct_uint32_t>(apBufGRGPU, as->apBufGR.get());
	getGRGPUData<float>(gEGRSumGPU, as->gMFSumGR.get());
	getGRGPUData<float>(gIGRSumGPU, as->gGOSumGR.get());

	getGRGPUData<float>(threshGRGPU, as->threshGR.get());
	getGRGPUData<float>(vGRGPU, as->vGR.get());
	getGRGPUData<float>(gKCaGRGPU, as->gKCaGR.get());
	getGRGPUData<ct_uint64_t>(historyGRGPU, as->historyGR.get());
	for(int i=0; i<numGPUs; i++)
	{
		int cpyStartInd;
		int cpySize;

		cpyStartInd=numGRPerGPU*i;
		cpySize=numGRPerGPU;

		cudaSetDevice(i+gpuIndStart);

		for(int j=0; j<MAX_NUM_P_GR_FROM_MF_TO_GR; j++)
		{
			error=cudaMemcpy(&gMFGRT[j][cpyStartInd], (void *)((char *)gEGRGPU[i]+j*gEGRGPUP[i]),
					cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		}

		for(int j=0; j<MAX_NUM_P_GR_FROM_GO_TO_GR; j++)
		{
			error=cudaMemcpy(&gGOGRT[j][cpyStartInd], (void *)((char *)gIGRGPU[i] + j * gIGRGPUP[i]),
					cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		}
	}

	for (int i = 0; i < MAX_NUM_P_GR_FROM_MF_TO_GR; i++)
	{
		for (int j = 0; j < NUM_GR; j++)
		{
			// NOTE: gMFGR now 1D array.
			as->gMFGR[j * NUM_GR + i] = gMFGRT[i][j];
		}
	}

	for (int i = 0; i < MAX_NUM_P_GR_FROM_GO_TO_GR; i++)
	{
		for (int j = 0; j < NUM_GR; j++)
		{
			as->gGOGR[j * NUM_GR + i] = gGOGRT[i][j];
		}
	}
}

// wtf is this why does it not return numgpus
void getnumGPUs()
{
	cout << "NUMGPUS = " << endl;
//	cout << numGPUs << endl;
}

//void InNet::grStim(int startGRStim, int numGRStim)
//{
//	// might be a useless operation. would the state of these arrays
//	// on cpu be the same as on gpu at the time we modify the segment below?
//	// if so, no need to get gpu data. if not, need to get gpu data
//	getGRGPUData<ct_uint8_t>(outputGRGPU, as->apGR.get());
//	getGRGPUData<ct_uint32_t>(apBufGRGPU, as->apBufGR.get());
//	for (int j = startGRStim; j <= startGRStim + numGRStim; j++)
//	{
//		/* as-> apBufGR[j] |= 1u; // try this to see if we get the same result */
//		as->apBufGR[j] = as->apBufGR[j] | 1u; 
//		outputGRH[j] = true;
//	}
//	
//	for (int i = 0; i < numGPUs; i++)
//	{
//		int cpyStartInd;
//		int cpySize;
//
//		cpyStartInd=numGRPerGPU*i;//numGR*i/numGPUs;
//		cpySize=numGRPerGPU;
//		cudaSetDevice(i+gpuIndStart);
//
//		cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]),
//				cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
//		cudaMemcpy(outputGRGPU[i], &outputGRH[cpyStartInd],
//				cpySize*sizeof(ct_uint8_t), cudaMemcpyHostToDevice);
//	}
//}

const ct_uint8_t* InNet::exportAPMF()
{
	return (const ct_uint8_t *)apMFOut;
}

const ct_uint8_t* InNet::exportAPSC()
{
	return (const ct_uint8_t *)as->apSC.get();
}

const ct_uint8_t* InNet::exportAPGR()
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpy((void *)&outputGRH[i*numGRPerGPU], outputGRGPU[i],
				numGRPerGPU*sizeof(ct_uint8_t), cudaMemcpyDeviceToHost);
#ifdef DEBUGOUT
		cerr<<"exportAPGR cuda memcpy: "<<cudaGetErrorString(error)<<endl;
#endif
	}

	return (const ct_uint8_t *)outputGRH;
}

const ct_uint32_t* InNet::exportSumGRInputGO()
{
	return (const ct_uint32_t *)sumGRInputGO;
}

const float* InNet::exportSumGOInputGO()
{
	return (const float *)sumInputGOGABASynDepGO;
}

const ct_uint32_t* InNet::exportPFBCSum()
{
	return (const ct_uint32_t *) inputSumPFBCH;
}

ct_uint32_t** InNet::getApBufGRGPUPointer()
{
	return apBufGRGPU;
}

ct_uint32_t** InNet::getDelayBCPCSCMaskGPUPointer()
{
	return delayBCPCSCMaskGRGPU;
}

// YIKES this should be deprecated: control class should not be able
// to interact directly with the gpu pointers here!!!
ct_uint64_t** InNet::getHistGRGPUPointer()
{
	return historyGRGPU;
}

ct_uint32_t** InNet::getGRInputGOSumHPointer()
{
	return grInputGOSumH;
}

ct_uint32_t** InNet::getGRInputBCSumHPointer()
{
	return grInputBCSumH;
}

void InNet::updateMFActivties(const ct_uint8_t *actInMF)
{
	apMFOut = actInMF;
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<NUM_MF; i++)
		{
			as->histMF[i]=as->histMF[i] || (actInMF[i]>0);
			for(int j=0; j<numGPUs; j++)
			{
				apMFH[j][i]=(actInMF[i]>0);
			}
			as->apBufMF[i]=(as->apBufMF[i]<<1)|((actInMF[i]>0)*0x00000001);
		}
	}
}

void InNet::calcGOActivities(float goMin, int simNum, float GRGO, float MFGO, float GOGR, float gogoW)
{
	//50ms
	for (int i = 0; i < NUM_GO; i++)
	{
		sumGRInputGO[i] = 0;

		for (int j = 0; j < numGPUs; j++)
		{
			sumGRInputGO[i] += grInputGOSumH[j][i];
		}		
	}

#pragma omp parallel for
	for (int i = 0; i < NUM_GO; i++)
	{
		float gLeakGO = 0.02;
		//NMDA Low
		float gNMDAIncGRGO = (0.00000082263 * as->vGO[i] * as->vGO[i] * as->vGO[i])
		   + (0.00021653 * as->vGO[i] * as->vGO[i]) + (0.0195 * as->vGO[i]) + 0.6117; 

		//NMDA High
		as->gNMDAIncMFGO[i] = (0.00000011969 * as->vGO[i] * as->vGO[i] * as->vGO[i])
		   + (0.000089369 * as->vGO[i] * as->vGO[i]) + (0.0151 * as->vGO[i]) + 0.7713; 
	
		as->gSum_MFGO[i] = (as->inputMFGO[i] * MFGO) + as->gSum_MFGO[i] * ap.gDecMFtoGO;
		as->gSum_GOGO[i] = 0;
		as->gNMDAMFGO[i] = as->inputMFGO[i] * (MFGO * ap.NMDA_AMPAratioMFGO * as->gNMDAIncMFGO[i])
		   + as->gNMDAMFGO[i] * ap.gDecayMFtoGONMDA;	
		
		as->gGRGO[i] = (sumGRInputGO[i] * GRGO * as->synWscalerGRtoGO[i]) +as->gGRGO[i] * ap.gDecGRtoGO;
		as->gGRGO_NMDA[i] = sumGRInputGO[i] * ((GRGO * as->synWscalerGRtoGO[i]) * 0.6 * gNMDAIncGRGO)
		   + as->gGRGO_NMDA[i] * ap.gDecayMFtoGONMDA;
		
		as->threshCurGO[i] = as->threshCurGO[i] + (ap.threshRestGO - as->threshCurGO[i]) * ap.threshDecGO;
		
		as->vGO[i] = as->vGO[i] + (gLeakGO * (ap.eLeakGO - as->vGO[i])) + (as->gSum_GOGO[i] * (ap.eGABAGO - as->vGO[i]))
				- (as->gSum_MFGO[i] + as->gGRGO[i] + as->gNMDAMFGO[i]
					  + as->gGRGO_NMDA[i]) * as->vGO[i]
				- (as->vCoupleGO[i] * as->vGO[i]);
		
		if(as->vGO[i] > ap.threshMaxGO) as->vGO[i] = ap.threshMaxGO;
		
		as->apGO[i] = as->vGO[i] > as->threshCurGO[i];
		as->apBufGO[i] = (as->apBufGO[i] << 1) | (as->apGO[i] * 0x00000001);

		as->threshCurGO[i] = as->apGO[i] * ap.threshMaxGO + (!as->apGO[i]) * as->threshCurGO[i];

		as->inputMFGO[i]  = 0;
		as->inputGOGO[i]  = 0;
	}

	for (int i = 0; i < NUM_GO; i++)
	{
		for (int j = 0; j < numGPUs; j++)
		{
			apGOH[j][i] = as->apGO[i];		
		}
	}
	as->goTimeStep++;
}


void InNet::calcSCActivities()
{
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < NUM_SC; i++)
		{
			as->gPFSC[i] = as->gPFSC[i] + inputSumPFSCH[i] * ap.gIncGRtoSC;
			as->gPFSC[i] = as->gPFSC[i] * ap.gDecGRtoSC;

			as->vSC[i] = as->vSC[i] + ap.gLeakSC * (ap.eLeakSC - as->vSC[i]) - as->gPFSC[i] * as->vSC[i];

			as->apSC[i] = (as->vSC[i] > as->threshSC[i]);
			as->apBufSC[i] = (as->apBufSC[i] << 1) | (as->apSC[i] * 0x00000001);

			as->threshSC[i] = as->threshSC[i] + ap.threshDecSC * (ap.threshRestSC - as->threshSC[i]);
			as->threshSC[i] = as->apSC[i] * ap.threshMaxSC + (!as->apSC[i]) * as->threshSC[i];
		}
	}
}

void InNet::updateMFtoGROut()
{
	float recoveryRate = 1 / ap.recoveryTauMF;

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < NUM_MF; i++)
		{			
			as->depAmpMFGR[i] = apMFH[0][i] * as->depAmpMFGR[i] * ap.fracDepMF
			   + (!apMFH[0][i]) * (as->depAmpMFGR[i] + recoveryRate * (1 - as->depAmpMFGR[i])); 
#pragma omp critical	
			{	
				for (int j = 0; j < numGPUs; j++)
				{
					depAmpMFH[j][i] = as->depAmpMFGR[i];
				}
			}	
		}	
	}
}

void InNet::updateMFtoGOOut()
{
	float recoveryRate = 1 / ap.recoveryTauMF;

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < NUM_MF; i++)
		{			
			as->gi_MFtoGO[i] = apMFH[0][i] * ap.gIncMFtoGO * as->depAmpMFGO[i]
			   + as->gi_MFtoGO[i] * ap.gDecMFtoGO; 
			as->depAmpMFGO[i] = apMFH[0][i] * as->depAmpMFGO[i] * ap.fracDepMF
			   + (!apMFH[0][i]) * (as->depAmpMFGO[i] + recoveryRate * (1 - as->depAmpMFGO[i])); 

			if (apMFH[0][i])
			{
#pragma omp critical
				{
					for (int j = 0; j < cs->numpMFfromMFtoGO[i]; j++)
					{
						as->inputMFGO[cs->pMFfromMFtoGO[i][j]]++;	
					}
				}
			
			}

		}
	
	}
}

void InNet::updateGOtoGROutParameters(float GOGR, float spillFrac)
{

	float scalerGOGR = GOGR*ap.gIncFracSpilloverGOtoGR*1.4;
	float halfShift = 12.0;//shift;
	float steepness = 20.0;//steep; 
	float recoveryRate = 1/ap.recoveryTauGO;
	float baselvl = spillFrac*GOGR;

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < NUM_GO; i++)
		{			
			as->depAmpGOGR[i] = 1;

			as->dynamicAmpGOGR[i] = baselvl + (scalerGOGR * (1 / (1 + (exp((counter[i] - halfShift) / steepness)))));
			counter[i] = (1 - as->apGO[i]) * counter[i] + 1; 

#pragma omp critical
			{
				for (int j = 0; j < numGPUs; j++)
				{
					depAmpGOH[j][i] = 1;
					dynamicAmpGOH[j][i] = as->dynamicAmpGOGR[i];
				}
			}
		}
	}
}

void InNet::updateGOtoGOOut()
{
	float gjCoupleScaler = ap.coupleRiRjRatioGO;
	float recoveryRate = 1 / ap.recoveryTauGO;

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < NUM_GO; i++)
		{
			
			as->gi_GOtoGO[i] =  as->apGO[i] * ap.gGABAIncGOtoGO * as->depAmpGOGO[i]
			   + as->gi_GOtoGO[i] * ap.gGABADecGOtoGO; 
			as->depAmpGOGO[i] = 1;
		}

		for (int i = 0; i < NUM_GO; i++)
		{
			if (as->apGO[i])
			{
				for (int j = 0; j < cs->numpGOGABAOutGOGO[i]; j++)
				{
					as->inputGOGO[cs->pGOGABAOutGOGO[i][j]]++;	
				}
			
			}
		}

#pragma omp for
		for(int i=0; i<NUM_GO; i++)
		{
			float threshCoupleGO;

			as->vCoupleGO[i]=0;

			threshCoupleGO=0;
			for(int j=0; j<cs->numpGOCoupInGOGO[i]; j++)
			{
				as->vCoupleGO[i] = as->vCoupleGO[i] + ((as->vGO[cs->pGOCoupInGOGO[i][j]] - as->vGO[i])
					  * gjCoupleScaler * cs->pGOCoupInGOGOCCoeff[i][j]);
			}
		}
	}
}

void InNet::resetMFHist(unsigned long t)
{
	if(t%ap.numTSinMFHist==0)
	{
#pragma omp parallel
		{
#pragma omp for
			for(int i=0; i<NUM_MF; i++)
			{
				as->histMF[i] = false;
			}
		}
	}
}

void InNet::runGRActivitiesCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	
	float gAMPAInc = (ap.gIncDirectMFtoGR)+(ap.gIncDirectMFtoGR*ap.gIncFracSpilloverMFtoGR); 

	for(int i=0; i<numGPUs; i++)
	{	
		error=cudaSetDevice(i+gpuIndStart);
		callGRActKernel(sts[i][streamN], calcGRActNumBlocks, calcGRActNumGRPerB,
				vGRGPU[i], gKCaGRGPU[i], gLeakGRGPU[i], gNMDAGRGPU[i], gNMDAIncGRGPU[i], threshGRGPU[i],
				apBufGRGPU[i], outputGRGPU[i], apGRGPU[i], apMFtoGRGPU[i], gEGRSumGPU[i], 
				gIGRSumGPU[i], ap.eLeakGR, ap.eGOGR, gAMPAInc, ap.threshRestGR, ap.threshMaxGR,
				ap.threshDecGR);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"grActivityCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
		}
}

void InNet::runSumPFBCCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callSumKernel<ct_uint32_t, true, false>
		(sts[i][streamN], inputPFBCGPU[i], inputPFBCGPUP[i],
				inputSumPFBCGPU[i], 1, NUM_BC/numGPUs, 1, NUM_P_BC_FROM_GR_TO_BC);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runSumPFBCCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::runSumPFSCCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callSumKernel<ct_uint32_t, true, false>
		(sts[i][streamN], inputPFSCGPU[i], inputPFSCGPUP[i],
				inputSumPFSCGPU[i], 1, NUM_SC/numGPUs, 1, NUM_P_SC_FROM_GR_TO_SC);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runSumPFBCCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runSumGRGOOutCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callSumGRGOOutKernel(sts[i][streamN], sumGRGOOutNumBlocks, sumGRGOOutNumGOPerB,
				updateGRGOOutNumGRRows, grInputGOGPU[i], grInputGOGPUP[i], grInputGOSumGPU[i]);
	}
}
void InNet::runSumGRBCOutCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callSumGRBCOutKernel(sts[i][streamN], sumGRBCOutNumBlocks, sumGRBCOutNumBCPerB,
				updateGRBCOutNumGRRows, grInputBCGPU[i], grInputBCGPUP[i], grInputBCSumGPU[i]);
	}
}

void InNet::cpyDepAmpMFHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(depAmpMFGPU[i], depAmpMFH[i],
			NUM_MF*sizeof(float), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPMFHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}


void InNet::cpyAPMFHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(apMFGPU[i], apMFH[i],
			NUM_MF*sizeof(ct_uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPMFHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyDepAmpUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN) {}


void InNet::cpyAPUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN) {}

void InNet::cpyDepAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN) {}

void InNet::cpyDynamicAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(dynamicAmpGOGPU[i], dynamicAmpGOH[i],
			NUM_GO*sizeof(float), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyDynamicAmpGOGRHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyAPGOHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(apGOGPU[i], apGOH[i],
			NUM_GO*sizeof(ct_uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPGOHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::runUpdateMFInGRCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateMFInGROPKernel(sts[i][streamN], updateMFInGRNumBlocks, updateMFInGRNumGRPerB,
				NUM_MF, apMFGPU[i], depAmpMFGRGPU[i] ,gEGRGPU[i], gEGRGPUP[i],   
				grConMFOutGRGPU[i], grConMFOutGRGPUP[i],
				numMFInPerGRGPU[i], apMFtoGRGPU[i], gEGRSumGPU[i], gEDirectGPU[i], gESpilloverGPU[i], 
				ap.gDirectDecMFtoGR, ap.gIncDirectMFtoGR, ap.gSpilloverDecMFtoGR,
				ap.gIncFracSpilloverMFtoGR);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGRCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::runUpdateMFInGRDepressionCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateMFInGRDepressionOPKernel(sts[i][streamN], updateMFInGRNumBlocks,
				updateMFInGRNumGRPerB, NUM_MF, depAmpMFGPU[i], grConMFOutGRGPU[i],
				grConMFOutGRGPUP[i], numMFInPerGRGPU[i], numMFperGR[i], depAmpMFGRGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGRDepressionCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::runUpdateUBCInGRCUDA(cudaStream_t **sts, int streamN) {}


void InNet::runUpdateUBCInGRDepressionCUDA(cudaStream_t **sts, int streamN) {}

void InNet::runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN, float GOGR)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateInGROPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				NUM_GO, apGOGPU[i], dynamicAmpGOGRGPU[i], gIGRGPU[i], gIGRGPUP[i],
				grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
				numGOInPerGRGPU[i], gIGRSumGPU[i], gIDirectGPU[i], gISpilloverGPU[i], 
				ap.gDirectDecGOtoGR, GOGR, ap.gIncFracSpilloverGOtoGR, ap.gSpilloverDecGOtoGR);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateGOInGRCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::runUpdateGOInGRDepressionCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateGOInGRDepressionOPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				NUM_MF, depAmpGOGPU[i], grConGOOutGRGPU[i], grConGOOutGRGPUP[i], numGOInPerGRGPU[i],
				depAmpGOGRGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGRDepressionCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runUpdateGOInGRDynamicSpillCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateGOInGRDynamicSpillOPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				NUM_MF, dynamicAmpGOGPU[i], grConGOOutGRGPU[i], grConGOOutGRGPUP[i], numGOInPerGRGPU[i],
				dynamicAmpGOGRGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGRDynamicSpillCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::runUpdatePFBCSCOutCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdatePFBCSCOutKernel(sts[i][streamN], updatePFBCSCNumBlocks, updatePFBCSCNumGRPerB,
				apBufGRGPU[i], delayBCPCSCMaskGRGPU[i],
				inputPFBCGPU[i], inputPFBCGPUP[i], NUM_P_BC_FROM_GR_TO_BC_P2, /* why pwr 2? */
				inputPFSCGPU[i], inputPFSCGPUP[i], NUM_P_SC_FROM_GR_TO_SC_P2); /* why pwr 2?*/
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdatePFBCSCOutCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateGROutGOKernel(sts[i][streamN], updateGRGOOutNumGRRows, updateGRGOOutNumGRPerR,
				NUM_GO, apBufGRGPU[i], grInputGOGPU[i], grInputGOGPUP[i],
				delayGOMasksGRGPU[i], delayGOMasksGRGPUP[i],
				grConGROutGOGPU[i], grConGROutGOGPUP[i], numGOOutPerGRGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateGROutGOCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::runUpdateGROutBCCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateGROutBCKernel(sts[i][streamN], updateGRBCOutNumGRRows, updateGRBCOutNumGRPerR,
				NUM_BC, apBufGRGPU[i], grInputBCGPU[i], grInputBCGPUP[i],
				delayBCMasksGRGPU[i], delayBCMasksGRGPUP[i],
				grConGROutBCGPU[i], grConGROutBCGPUP[i], numBCOutPerGRGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateGROutBCCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyPFBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(&inputSumPFBCH[NUM_BC*i/numGPUs], inputSumPFBCGPU[i],
				NUM_BC/numGPUs*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyPFBCSumGPUtoHostCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyPFSCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(&inputSumPFSCH[NUM_SC*i/numGPUs], inputSumPFSCGPU[i],
				NUM_SC/numGPUs*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyPFSCSumGPUtoHostCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cpyGRGOSumGPUtoHostCUDA(sts, streamN, grInputGOSumH);
}

void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN, ct_uint32_t **grInputGOSumHost)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);

		error=cudaMemcpyAsync(grInputGOSumHost[i], grInputGOSumGPU[i], NUM_GO*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::cpyGRBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{

	cpyGRBCSumGPUtoHostCUDA(sts, streamN, grInputBCSumH);
}

void InNet::cpyGRBCSumGPUtoHostCUDA(cudaStream_t **sts,
	int streamN, ct_uint32_t **grInputBCSumHost)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);

		error=cudaMemcpyAsync(grInputBCSumHost[i], grInputBCSumGPU[i], NUM_BC*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, unsigned long t)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		if(t%ap.tsPerHistBinGR==0)
		{
			callUpdateGRHistKernel(sts[i][streamN], updateGRHistNumBlocks, updateGRHistNumGRPerB,
					apBufGRGPU[i], historyGRGPU[i], apBufGRHistMask);
		}
	}
}

/* =========================== PROTECTED FUNCTIONS ============================= */

void InNet::initCUDA()
{
	cudaError_t error;
	int maxNumGPUs;

	error = cudaGetDeviceCount(&maxNumGPUs);
	cerr<<"CUDA number of devices: "<<maxNumGPUs<<", "<<cudaGetErrorString(error)<<endl;
	cerr<<"number of devices used: "<<numGPUs<<" starting at GPU# "<<gpuIndStart<<endl;

	numGRPerGPU=NUM_GR / numGPUs;
	calcGRActNumGRPerB=512;
	calcGRActNumBlocks=numGRPerGPU/calcGRActNumGRPerB;

	updateGRGOOutNumGRPerR=512 * (NUM_GO>512) + NUM_GO*(NUM_GO<=512);
	updateGRGOOutNumGRRows=numGRPerGPU/updateGRGOOutNumGRPerR;

	sumGRGOOutNumGOPerB=1024*(NUM_GO>1024)+NUM_GO*(NUM_GO<=1024);
	sumGRGOOutNumBlocks=NUM_GO/sumGRGOOutNumGOPerB;

	updateMFInGRNumGRPerB=1024*(NUM_MF>1024)+(NUM_MF<=1024)*NUM_MF;
	updateMFInGRNumBlocks=numGRPerGPU/updateMFInGRNumGRPerB;

	updateUBCInGRNumGRPerB=1024*(NUM_UBC>1024)+(NUM_UBC<=1024)*NUM_UBC;
	updateUBCInGRNumBlocks=numGRPerGPU/updateUBCInGRNumGRPerB;

	updateGOInGRNumGRPerB=1024*(NUM_GO>=1024)+(NUM_GO<1024)*NUM_GO;
	updateGOInGRNumBlocks=numGRPerGPU/updateGOInGRNumGRPerB;

	updateGRBCOutNumGRPerR=512*(NUM_BC>512)+NUM_BC*(NUM_BC<=512);
	updateGRBCOutNumGRRows=numGRPerGPU/updateGRBCOutNumGRPerR;
	
	sumGRBCOutNumBCPerB=1024*(NUM_BC>1024)+NUM_BC*(NUM_BC<=1024);
	sumGRBCOutNumBlocks=NUM_BC/sumGRBCOutNumBCPerB;
		
	updatePFBCSCNumGRPerB=512;
	updatePFBCSCNumBlocks=numGRPerGPU/updatePFBCSCNumGRPerB;

	updateGRHistNumGRPerB=1024;
	updateGRHistNumBlocks=numGRPerGPU/updateGRHistNumGRPerB;


	cerr<<"numGRPerGPU: "<<numGRPerGPU<<endl;
	cerr<<"calcGRActNumBlocks "<<calcGRActNumBlocks<<endl;

	cerr<<"updateGRGOOutNumGRPerR "<<updateGRGOOutNumGRPerR<<endl;
	cerr<<"updateGRGOOutNumGRRows "<<updateGRGOOutNumGRRows<<endl;
	
	cerr<<"updateGRBCOutNumGRPerR "<<updateGRBCOutNumGRPerR<<endl;
	cerr<<"updateGRBCOutNumGRRows "<<updateGRBCOutNumGRRows<<endl;

	cerr<<"sumGRGOOutNumGOPerB "<<sumGRGOOutNumGOPerB<<endl;
	cerr<<"sumGRGOOutNumBlocks "<<sumGRGOOutNumBlocks<<endl;
	
	cerr<<"sumGRBCOutNumBCPerB "<<sumGRBCOutNumBCPerB<<endl;
	cerr<<"sumGRBCOutNumBlocks "<<sumGRBCOutNumBlocks<<endl;

	cerr<<"updateMFInGRNumGRPerB "<<updateMFInGRNumGRPerB<<endl;
	cerr<<"updateMFInGRNumBlocks "<<updateMFInGRNumBlocks<<endl;
	
	cerr<<"updateUBCInGRNumGRPerB "<<updateUBCInGRNumGRPerB<<endl;
	cerr<<"updateUBCInGRNumBlocks "<<updateUBCInGRNumBlocks<<endl;

	cerr<<"updateGOInGRNumGRPerB "<<updateGOInGRNumGRPerB<<endl;
	cerr<<"updateGOInGRNumBlocks "<<updateGOInGRNumBlocks<<endl;

	cerr<<"updateGRHistNumBlocks "<<updateGRHistNumBlocks<<endl;

	cerr<<"input network cuda init..."<<endl;
	
	initUBCCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA UBC init: "<<cudaGetErrorString(error)<<endl;
	initMFCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA MF init: "<<cudaGetErrorString(error)<<endl;
	initGRCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA gr init: "<<cudaGetErrorString(error)<<endl;
	initGOCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA go init: "<<cudaGetErrorString(error)<<endl;
	initBCCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA bc init: "<<cudaGetErrorString(error)<<endl;
	initSCCUDA();
	error=cudaGetLastError();
	cerr<<"CUDA sc init: "<<cudaGetErrorString(error)<<endl;
}

void InNet::initUBCCUDA()
{
	cudaError_t error;

	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		cudaDeviceSynchronize();
	}

}

void InNet::initMFCUDA()
{
	cudaError_t error;

	apMFGPU=new ct_uint32_t*[numGPUs];
	apMFH=new ct_uint32_t*[numGPUs];
	depAmpMFH=new float*[numGPUs];
	depAmpMFGPU=new float*[numGPUs];

	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		cerr<<"setting device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apMFGPU[i], NUM_MF*sizeof(ct_uint32_t));
		cerr<<"Allocating apMFGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMallocHost((void **)&apMFH[i], NUM_MF*sizeof(ct_uint32_t));
		cerr<<"Allocating apMFH for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
			
		error=cudaMallocHost((void **)&depAmpMFH[i], NUM_MF*sizeof(float));
		cerr<<"Allocating depAmpMFH for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc<float>(&(depAmpMFGPU[i]), NUM_MF*sizeof(float));
		cerr<<"Allocating depAmpMFGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;

		cudaDeviceSynchronize();

		cudaMemset(apMFGPU[i], 0, NUM_MF*sizeof(ct_uint32_t));
		cudaMemset(depAmpMFGPU[i], 1, NUM_MF*sizeof(float));

		for(int j=0; j<NUM_MF; j++)
		{
			apMFH[i][j]=0;
			depAmpMFH[i][j]=1;
		}
	}
}

void InNet::initGRCUDA()
{
	cudaError_t error;

	gEGRGPU=new float*[numGPUs];
	gEGRGPUP=new size_t[numGPUs];
	gEGRSumGPU=new float*[numGPUs];
	gEDirectGPU=new float*[numGPUs];
	gESpilloverGPU=new float*[numGPUs];
	apMFtoGRGPU=new int*[numGPUs];
	numMFperGR=new int*[numGPUs];	
	numUBCperGR=new int*[numGPUs];	
	depAmpMFGRGPU=new float*[numGPUs];
	depAmpGOGRGPU=new float*[numGPUs];
	dynamicAmpGOGRGPU=new float*[numGPUs];
	gUBC_EGRGPU=new float*[numGPUs];
	gUBC_EGRGPUP=new size_t[numGPUs];
	gUBC_EDirectGPU=new float*[numGPUs];
	gUBC_ESpilloverGPU=new float*[numGPUs];

	gIGRGPU=new float*[numGPUs];
	gIGRGPUP=new size_t[numGPUs];
	gIGRSumGPU=new float*[numGPUs];
	gIDirectGPU=new float*[numGPUs];
	gISpilloverGPU=new float*[numGPUs];

	apBufGRGPU=new ct_uint32_t*[numGPUs];
	outputGRGPU=new ct_uint8_t*[numGPUs];
	apGRGPU=new ct_uint32_t*[numGPUs];

	threshGRGPU=new float*[numGPUs];
	vGRGPU=new float*[numGPUs];
	gKCaGRGPU=new float*[numGPUs];
	gLeakGRGPU=new float*[numGPUs];	
	gNMDAGRGPU=new float*[numGPUs];
	gNMDAIncGRGPU=new float*[numGPUs];
	historyGRGPU=new ct_uint64_t*[numGPUs];

	delayGOMasksGRGPU=new ct_uint32_t*[numGPUs];
	delayGOMasksGRGPUP=new size_t[numGPUs];
	delayBCPCSCMaskGRGPU=new ct_uint32_t*[numGPUs];
	
	
	delayBCMasksGRGPU=new ct_uint32_t*[numGPUs];
	delayBCMasksGRGPUP=new size_t[numGPUs];
	grConGROutBCGPU=new ct_uint32_t*[numGPUs];
	grConGROutBCGPUP=new size_t[numGPUs];
	numBCOutPerGRGPU=new ct_int32_t*[numGPUs];


	numGOOutPerGRGPU=new ct_int32_t*[numGPUs];
	grConGROutGOGPU=new ct_uint32_t*[numGPUs];
	grConGROutGOGPUP=new size_t[numGPUs];

	numGOInPerGRGPU=new ct_int32_t*[numGPUs];
	grConGOOutGRGPU=new ct_uint32_t*[numGPUs];
	grConGOOutGRGPUP=new size_t[numGPUs];

	numMFInPerGRGPU=new ct_int32_t*[numGPUs];
	numUBCInPerGRGPU=new ct_int32_t*[numGPUs];
	grConMFOutGRGPU=new ct_uint32_t*[numGPUs];
	grConUBCOutGRGPU=new ct_uint32_t*[numGPUs];
	grConMFOutGRGPUP=new size_t[numGPUs];
	grConUBCOutGRGPUP=new size_t[numGPUs];


	outputGRH=new ct_uint8_t[NUM_GR];
	std::fill(outputGRH, outputGRH + NUM_GR, 0);

	//allocate memory for GPU
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		cerr<<"setting device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&outputGRGPU[i], numGRPerGPU*sizeof(ct_uint8_t));
		cerr<<"allocating outputGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		cerr<<"allocating apGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc<float>(&(vGRGPU[i]), numGRPerGPU*sizeof(float));
		cerr<<"numGRPerGPU "<<numGRPerGPU<<endl;
		cerr<<"allocating vGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc<float>(&(gKCaGRGPU[i]), numGRPerGPU*sizeof(float));
		cerr<<"allocating gKCaGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc<float>(&(gLeakGRGPU[i]), numGRPerGPU*sizeof(float));
		cerr<<"allocating gLeakGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc<float>(&(gNMDAGRGPU[i]), numGRPerGPU*sizeof(float));
		cerr<<"allocating gNMDAGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc<float>(&(gNMDAIncGRGPU[i]), numGRPerGPU*sizeof(float));
		cerr<<"allocating gNMDAIncGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
	
		error=cudaMallocPitch((void **)&gEGRGPU[i], (size_t *)&gEGRGPUP[i],
				numGRPerGPU*sizeof(float), MAX_NUM_P_GR_FROM_MF_TO_GR);
		cerr<<"gEGRGPUP: "<<gEGRGPUP[i]<<endl;
		error=cudaMalloc((void **)&gEGRSumGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gEGRSumGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&gEDirectGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gEDirectGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&gESpilloverGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gESpilloverGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apMFtoGRGPU[i], numGRPerGPU*sizeof(int));
		cerr<<"allocating apMFtoGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&numMFperGR[i], numGRPerGPU*sizeof(int));
		cerr<<"allocating numMFperGR for device "<<i<<": "<<cudaGetErrorString(error)<<endl;	
		
		error=cudaMallocPitch((void **)&gUBC_EGRGPU[i], (size_t *)&gUBC_EGRGPUP[i],
				numGRPerGPU*sizeof(float), MAX_NUM_P_GR_FROM_MF_TO_GR);
		
		error=cudaMalloc((void **)&depAmpMFGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating depAmpMFGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&depAmpGOGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating depAmpGOGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&dynamicAmpGOGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating dynamicAmpGOGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		
		
		error=cudaMallocPitch((void **)&gIGRGPU[i], (size_t *)&gIGRGPUP[i],
				numGRPerGPU*sizeof(float), MAX_NUM_P_GR_FROM_GO_TO_GR);
		cerr<<"gEGRGPUP: "<<gIGRGPUP[i]<<endl;
		error=cudaMalloc((void **)&gIGRSumGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gIGRSumGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&gIDirectGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gIDirectGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&gISpilloverGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gISpilloverGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apBufGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		error=cudaMalloc((void **)&threshGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating threshGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		
		//variables for conduction delays
		error=cudaMalloc((void **)&delayBCPCSCMaskGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		error=cudaMallocPitch((void **)&delayGOMasksGRGPU[i], (size_t *)&delayGOMasksGRGPUP[i],
				numGRPerGPU*sizeof(ct_uint32_t), MAX_NUM_P_GR_FROM_GR_TO_GO);	
		//end conduction delay

		//New Basket Cell stuff
		error=cudaMallocPitch((void **)&grConGROutBCGPU[i], (size_t *)&grConGROutBCGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), NUM_BC);
		error=cudaMallocPitch((void **)&delayBCMasksGRGPU[i], (size_t *)&delayBCMasksGRGPUP[i],
				numGRPerGPU*sizeof(ct_uint32_t), NUM_BC);
		error=cudaMalloc((void **)&numBCOutPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));


		//connectivity
		error=cudaMallocPitch((void **)&grConGROutGOGPU[i], (size_t *)&grConGROutGOGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), MAX_NUM_P_GR_FROM_GR_TO_GO);
		error=cudaMalloc((void **)&numGOOutPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));

		error=cudaMallocPitch((void **)&grConGOOutGRGPU[i], (size_t *)&grConGOOutGRGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), MAX_NUM_P_GR_FROM_GO_TO_GR);
		error=cudaMalloc((void **)&numGOInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));

		error=cudaMallocPitch((void **)&grConMFOutGRGPU[i], (size_t *)&grConMFOutGRGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), MAX_NUM_P_GR_FROM_MF_TO_GR);
		error=cudaMalloc((void **)&numMFInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));
		
		//end connectivity

		error=cudaMalloc((void **)&historyGRGPU[i], numGRPerGPU*sizeof(ct_uint64_t));
		//end GPU memory allocation
		cudaDeviceSynchronize();

		cerr<<"GR GPU memory allocation: "<<cudaGetErrorString(error)<<endl;
	}
	cout << "NUMGPUS" << endl;
	cout << numGPUs;
	cout << "Passed" << endl;
	
	error=cudaGetLastError();
	cerr<<"MemAllocDone: "<<cudaGetErrorString(error)<<endl;

	//	create a transposed copy of the matrices from activity state and connectivity
	for (int i = 0; i < MAX_NUM_P_GR_FROM_GO_TO_GR; i++)
	{
		for (int j = 0; j < NUM_GR; j++)
		{
			gGOGRT[i][j] = as->gGOGR[j * NUM_GR + i];
			pGRfromGOtoGRT[i][j] = cs->pGRfromGOtoGR[j][i];
		}
	}

	for (int i = 0; i < MAX_NUM_P_GR_FROM_MF_TO_GR; i++)
	{
		for (int j = 0; j < NUM_GR; j++)
		{
			gMFGRT[i][j] 		 = as->gMFGR[j * NUM_GR + i];
			pGRfromMFtoGRT[i][j] = cs->pGRfromMFtoGR[j][i];
		}
	}
	
	for(int i=0; i<MAX_NUM_P_GR_FROM_GR_TO_GO; i++)
	{
		for(int j=0; j<NUM_GR; j++)
		{
			pGRDelayfromGRtoGOT[i][j]=cs->pGRDelayMaskfromGRtoGO[j][i];
			pGRfromGRtoGOT[i][j]=cs->pGRfromGRtoGO[j][i];
		}
	}

	//initialize GR GPU variables
	cerr<<"start GPU memory initialization"<<endl;
	
	for(int i=0; i<numGPUs; i++)
	{
		int cpyStartInd;
		int cpySize;

		cpyStartInd=numGRPerGPU*i;//numGR*i/numGPUs;
		cpySize=numGRPerGPU;
		cudaSetDevice(i+gpuIndStart);

		error=cudaMemcpy(gKCaGRGPU[i], &(as->gKCaGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);
	
		cerr<<"cuda memory copy vGRGPU, outputGRGPU, and gKCAGRGPU: "<<cudaGetErrorString(error)<<endl;

		for(int j=0; j<MAX_NUM_P_GR_FROM_MF_TO_GR; j++)
		{
			error=cudaMemcpy((void *)((char *)gEGRGPU[i]+j*gEGRGPUP[i]),
					&gMFGRT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice);	
			error=cudaMemcpy((void *)((char *)grConMFOutGRGPU[i]+j*grConMFOutGRGPUP[i]),
					&pGRfromMFtoGRT[j][cpyStartInd], cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		}
		cerr<<"cuda memory copy gEGRGPU and grConMFOutGRGPU: "<<cudaGetErrorString(error)<<endl;
	
		error=cudaMemcpy(vGRGPU[i], &(as->vGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);	
		error=cudaMemcpy(gEGRSumGPU[i], &(as->gMFSumGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);	
		error = cudaMemset(gEDirectGPU[i], 0.0, cpySize * sizeof(float));
		error = cudaMemset(gESpilloverGPU[i], 0.0, cpySize * sizeof(float));
		error=cudaMemcpy(apMFtoGRGPU[i], &(as->apMFtoGR[cpyStartInd]), cpySize*sizeof(int), cudaMemcpyHostToDevice);
		error=cudaMemcpy(numMFperGR[i], &(cs->numpGRfromMFtoGR[cpyStartInd]), cpySize*sizeof(int), cudaMemcpyHostToDevice);	
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
	
		error = cudaMemset(depAmpMFGRGPU[i], 1.0, cpySize * sizeof(float));	
		
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
		
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
		error = cudaMemset(depAmpGOGRGPU[i], 1.0, cpySize * sizeof(float));
		
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
		error = cudaMemset(dynamicAmpGOGRGPU[i], 0.0, cpySize * sizeof(float));
		
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
			
		for(int j=0; j<MAX_NUM_P_GR_FROM_GO_TO_GR; j++)
		{
			error=cudaMemcpy((void *)((char *)gIGRGPU[i] + j * gIGRGPUP[i]),
					&gGOGRT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice);
			error=cudaMemcpy((void *)((char *)grConGOOutGRGPU[i]+j*grConGOOutGRGPUP[i]),
					&pGRfromGOtoGRT[j][cpyStartInd], cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		}

		cout << "check" << endl;	

		error=cudaMemcpy(gIGRSumGPU[i], &(as->gGOSumGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error = cudaMemset(gIDirectGPU[i], 0.0, cpySize * sizeof(float));	
		error = cudaMemset(gISpilloverGPU[i], 0.0, cpySize * sizeof(float));

		error=cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]),
				cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);

		error=cudaMemcpy(threshGRGPU[i], &(as->threshGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);
		cout << "check" << endl;	

		// TODO: place initial value of gLeak into activityparams file and actually use that
		// (since its not default val of float)
		error = cudaMemset(gLeakGRGPU[i], 0.11, cpySize * sizeof(float));
		error = cudaMemset(gNMDAGRGPU[i], 0.0, cpySize * sizeof(float));
		error = cudaMemset(gNMDAIncGRGPU[i], 0.0, cpySize * sizeof(float));
		cout << "check" << endl;	

		for(int j=0; j<MAX_NUM_P_GR_FROM_GR_TO_GO; j++)
		{
			error=cudaMemcpy((void *)((char *)delayGOMasksGRGPU[i]+j*delayGOMasksGRGPUP[i]),
					&pGRDelayfromGRtoGOT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice );
			error=cudaMemcpy((void *)((char *)grConGROutGOGPU[i]+j*grConGROutGOGPUP[i]),
					&pGRfromGRtoGOT[j][cpyStartInd], cpySize*sizeof(unsigned int), cudaMemcpyHostToDevice);
		}
		error=cudaGetLastError();
		cerr<<"CUDA check: "<<cudaGetErrorString(error)<<endl;

		//Basket cell stuff
		error=cudaMemcpy(numGOOutPerGRGPU[i], &(cs->numpGRfromGRtoGO[cpyStartInd]),
				cpySize*sizeof(ct_int32_t), cudaMemcpyHostToDevice);
		cout << "	check" << endl;	

		error=cudaMemcpy(numGOInPerGRGPU[i], &(cs->numpGRfromGOtoGR[cpyStartInd]),
				cpySize*sizeof(ct_int32_t), cudaMemcpyHostToDevice);
		cout << "check" << endl;	
		
		
		error=cudaMemcpy(delayBCPCSCMaskGRGPU[i], &(cs->pGRDelayMaskfromGRtoBSP[cpyStartInd]),
				cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);

		error=cudaMemcpy(numMFInPerGRGPU[i], &(cs->numpGRfromMFtoGR[cpyStartInd]),
				cpySize*sizeof(int), cudaMemcpyHostToDevice);

		error=cudaMemcpy(historyGRGPU[i], &(as->historyGR[cpyStartInd]),
				cpySize*sizeof(ct_uint64_t), cudaMemcpyHostToDevice);


		cout << "check" << endl;	

		cudaMemset(outputGRGPU[i], 0, cpySize*sizeof(ct_uint8_t));
		cudaMemset(apGRGPU[i], 0, cpySize*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
	
	//end copying to GPU
	cerr<<"numGRPerGPU "<<numGRPerGPU<<endl;
}

void InNet::initGOCUDA()
{
	// seems like we do not have to "preallocate" host variables: we can do that w/ cudaMallocHost
	grInputGOSumH=new ct_uint32_t*[numGPUs];
	apGOH=new ct_uint32_t*[numGPUs];
	apGOGPU=new ct_uint32_t*[numGPUs];
	grInputGOGPU=new ct_uint32_t*[numGPUs];
	grInputGOGPUP=new size_t[numGPUs];
	grInputGOSumGPU=new ct_uint32_t*[numGPUs];
	depAmpGOH=new float*[numGPUs];
	depAmpGOGPU=new float*[numGPUs];	
	dynamicAmpGOH=new float*[numGPUs];
	dynamicAmpGOGPU=new float*[numGPUs];
	
	plasScalerEx = new float[80];
	plasScalerInh = new float[80];
	goExScaler = allocate2DArray<float>(NUM_GO, 1000);	
	std::fill(goExScaler[0], goExScaler[0] + NUM_GO * 1000, 0);

	goInhScaler = allocate2DArray<float>(NUM_GO, 1000);	
	std::fill(goInhScaler[0], goInhScaler[0] + NUM_GO * 1000, 0);
	
	goFRArray = allocate2DArray<float>(NUM_GO, 1000);	
	std::fill(goFRArray[0], goFRArray[0] + NUM_GO * 1000, 0);
		
	counterGOweight = 0;

	counter = new int[NUM_GO];
	std::fill(counter, counter + NUM_GO, 0);

	//initialize host memory
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMallocHost((void **)&grInputGOSumH[i], NUM_GO*sizeof(ct_uint32_t));
		cudaMallocHost((void **)&apGOH[i], NUM_GO*sizeof(ct_uint32_t));
		cudaMallocHost((void **)&depAmpGOH[i], NUM_GO*sizeof(float));
		cudaMallocHost((void **)&dynamicAmpGOH[i], NUM_GO*sizeof(float));
	
		for(int j=0; j<NUM_GO; j++)
		{
			grInputGOSumH[i][j]=0;
			apGOH[i][j]=0;
			depAmpGOH[i][j]=1;
			dynamicAmpGOH[i][j]=1;
			
		}
		//allocate gpu memory
		cudaMalloc((void **)&apGOGPU[i], NUM_GO*sizeof(ct_uint32_t));
		cudaMalloc((void **)&depAmpGOGPU[i], NUM_GO*sizeof(float));
		cudaMalloc((void **)&dynamicAmpGOGPU[i], NUM_GO*sizeof(float));

		cudaMallocPitch((void **)&grInputGOGPU[i], (size_t *)&grInputGOGPUP[i],
				NUM_GO*sizeof(ct_uint32_t), updateGRGOOutNumGRRows);
		cudaMalloc((void **)&grInputGOSumGPU[i], NUM_GO*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();

		for(int j=0; j<updateGRGOOutNumGRRows; j++)
		{
			cudaMemset(((char *)grInputGOGPU[i]+j*grInputGOGPUP[i]),
					0, NUM_GO*sizeof(ct_uint32_t));
		}

		cudaMemset(apGOGPU[i], 0, NUM_GO*sizeof(ct_uint32_t));
		cudaMemset(depAmpGOGPU[i], 1, NUM_GO*sizeof(float));
		cudaMemset(dynamicAmpGOGPU[i], 1, NUM_GO*sizeof(float));
		cudaMemset(grInputGOSumGPU[i], 0, NUM_GO*sizeof(ct_uint32_t));
		cudaDeviceSynchronize();
	}
}
void InNet::initBCCUDA()
{
	
	grInputBCGPU=new ct_uint32_t*[numGPUs];
	grInputBCGPUP=new size_t[numGPUs];
	grInputBCSumGPU=new ct_uint32_t*[numGPUs];
	grInputBCSumH=new ct_uint32_t*[numGPUs];
	
	inputPFBCGPU=new ct_uint32_t*[numGPUs];
	inputPFBCGPUP=new size_t[numGPUs];
	inputSumPFBCGPU=new ct_uint32_t*[numGPUs];

	//allocate host memory
	cudaSetDevice(gpuIndStart);
	cudaHostAlloc((void **)&inputSumPFBCH, NUM_BC*sizeof(ct_uint32_t), cudaHostAllocPortable);

	cudaDeviceSynchronize();

	//initialize host variables
	for(int i=0; i<NUM_BC; i++)
	{
		inputSumPFBCH[i]=0;
	}
	cout << "before GPU loop" << endl;
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMallocHost((void **)&grInputBCSumH[i], NUM_BC*sizeof(ct_uint32_t));
		for(int j=0; j<NUM_BC; j++)
		{
			grInputBCSumH[i][j]=0;
		}
	
		cudaMallocPitch((void **)&grInputBCGPU[i], (size_t *)&grInputBCGPUP[i],
				NUM_BC*sizeof(ct_uint32_t), updateGRBCOutNumGRRows);
		cudaMalloc((void **)&grInputBCSumGPU[i], NUM_BC*sizeof(ct_uint32_t));		
		//allocate GPU memory
		cudaMallocPitch((void **)&inputPFBCGPU[i], (size_t *)&inputPFBCGPUP[i],
				NUM_P_BC_FROM_GR_TO_BC*sizeof(ct_uint32_t), NUM_BC/numGPUs);
		
		cudaMalloc((void **)&inputSumPFBCGPU[i], NUM_BC/numGPUs*sizeof(ct_uint32_t));
		
		//end GPU allocation
		
		cudaDeviceSynchronize();
		
		for(int j=0; j<updateGRBCOutNumGRRows; j++)
		{
			cudaMemset(((char *)grInputBCGPU[i]+j*grInputBCGPUP[i]),
					0, NUM_BC*sizeof(ct_uint32_t));
		}
		
		cudaMemset(grInputBCSumGPU[i], 0, NUM_BC*sizeof(ct_uint32_t));
		
		for(int j=0; j<NUM_BC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFBCGPU[i]+j*inputPFBCGPUP[i]), 0,
					NUM_P_BC_FROM_GR_TO_BC*sizeof(ct_uint32_t));
		}

		cudaMemset(inputSumPFBCGPU[i], 0, NUM_BC/numGPUs*sizeof(ct_uint32_t));		
		cudaDeviceSynchronize();
	}
}

void InNet::initSCCUDA()
{
	inputPFSCGPU=new ct_uint32_t*[numGPUs];
	inputPFSCGPUP=new size_t[numGPUs];
	inputSumPFSCGPU=new ct_uint32_t*[numGPUs];

	//allocate host memory
	cudaSetDevice(gpuIndStart);
	cudaHostAlloc((void **)&inputSumPFSCH, NUM_SC * sizeof(ct_uint32_t), cudaHostAllocPortable);

	cudaDeviceSynchronize();
	
	//initialize host variables
	for(int i=0; i<NUM_SC; i++)
	{
		inputSumPFSCH[i]=0;
	}

	for(int i=0; i<numGPUs; i++)
	{
		//allocate GPU memory
		cudaSetDevice(i+gpuIndStart);
		cudaMallocPitch((void **)&inputPFSCGPU[i], (size_t *)&inputPFSCGPUP[i],
				NUM_P_SC_FROM_GR_TO_SC*sizeof(ct_uint32_t), NUM_SC/numGPUs);

		cudaMalloc((void **)&inputSumPFSCGPU[i], NUM_SC/numGPUs*sizeof(ct_uint32_t));
		//end GPU allocation

		for(int j=0; j<NUM_SC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFSCGPU[i]+j*inputPFSCGPUP[i]), 0,
					NUM_P_SC_FROM_GR_TO_SC*sizeof(ct_uint32_t));
		}

		cudaMemset(inputSumPFSCGPU[i], 0, NUM_SC/numGPUs*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
}

/* =========================== PRIVATE FUNCTIONS ============================= */

template<typename Type>
cudaError_t InNet::getGRGPUData(Type **gpuData, Type *hostData)
{
	//cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemcpy((void *)&hostData[i * numGRPerGPU], gpuData[i],
				numGRPerGPU * sizeof(Type), cudaMemcpyDeviceToHost);
	}
	return cudaGetLastError();
}

