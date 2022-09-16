/*
 *
 * innet.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#include "innetmodules/innet.h"

using namespace std;

InNet::InNet() {}

InNet::InNet(InNetConnectivityState *cs,
	InNetActivityState *as, int gpuIndStart, int numGPUs)
{
	// all of below are shallow-copying the pointers
	// thus we don't necessarily have ownership over
	// them, so caller deletes them. we assume the innet
	// object goes out of scope before or at the time of 
	// the calling class.
	this->cs = cs; 
	this->as = as; 

	this->gpuIndStart = gpuIndStart;
	this->numGPUs     = numGPUs;

	// why do we allocate these here???
	gGOGRT = allocate2DArray<float>(max_num_p_gr_from_go_to_gr, num_gr);
	gMFGRT = allocate2DArray<float>(max_num_p_gr_from_mf_to_gr, num_gr);

	pGRDelayfromGRtoGOT = allocate2DArray<ct_uint32_t>(max_num_p_gr_from_gr_to_go, num_gr);
	pGRfromMFtoGRT = allocate2DArray<ct_uint32_t>(max_num_p_gr_from_mf_to_gr, num_gr);
	pGRfromGOtoGRT = allocate2DArray<ct_uint32_t>(max_num_p_gr_from_go_to_gr, num_gr);
	pGRfromGRtoGOT = allocate2DArray<ct_uint32_t>(max_num_p_gr_from_gr_to_go, num_gr);

	apBufGRHistMask = (1 << (int)tsPerHistBinGR) - 1;

	sumGRInputGO           = new ct_uint32_t[num_go];
	sumInputGOGABASynDepGO = new float[num_go];

	initCUDA();
}

InNet::~InNet()
{
	std::cout << "[INFO]: Deleting innet gpu arrays." << std::endl;

	//gr external to initCUDA
	delete2DArray<float>(gGOGRT);
	delete2DArray<float>(gMFGRT);

	delete2DArray<ct_uint32_t>(pGRDelayfromGRtoGOT);
	delete2DArray<ct_uint32_t>(pGRfromMFtoGRT);
	delete2DArray<ct_uint32_t>(pGRfromGOtoGRT);
	delete2DArray<ct_uint32_t>(pGRfromGRtoGOT);

	// go, external to initCUDA
	delete[] sumGRInputGO;
	delete[] sumInputGOGABASynDepGO;

	// MF CUDA	
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		//mf variables
		cudaFree(apMFGPU[i]);
		cudaFreeHost(apMFH[i]);
		cudaFree(depAmpMFGPU[i]);
		cudaFreeHost(depAmpMFH[i]);

		cudaDeviceSynchronize();
	}

	delete[] apMFGPU;
	delete[] apMFH;
	delete[] depAmpMFH;
	delete[] depAmpMFGPU;

	// GR CUDA
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaFree(outputGRGPU[i]);
		cudaFree(apGRGPU[i]);
		cudaFree(vGRGPU[i]);
		cudaFree(gKCaGRGPU[i]);
		cudaFree(gLeakGRGPU[i]);
		cudaFree(gNMDAGRGPU[i]);
		cudaFree(gNMDAIncGRGPU[i]);
		cudaFree(gEGRGPU[i]);
		cudaFree(gEGRSumGPU[i]);
		cudaFree(gEDirectGPU[i]);
		cudaFree(gESpilloverGPU[i]);
		cudaFree(apMFtoGRGPU[i]);
		cudaFree(numMFperGR[i]);
		cudaFree(depAmpMFGRGPU[i]);
		cudaFree(depAmpGOGRGPU[i]);
		cudaFree(dynamicAmpGOGRGPU[i]); 
		cudaFree(gIGRGPU[i]);
		cudaFree(gIGRSumGPU[i]);
		cudaFree(gIDirectGPU[i]);
		cudaFree(gISpilloverGPU[i]);
		cudaFree(apBufGRGPU[i]);
		cudaFree(threshGRGPU[i]);
		cudaFree(delayBCPCSCMaskGRGPU[i]);
		cudaFree(delayGOMasksGRGPU[i]);
		
		cudaFree(grConGROutBCGPU[i]);
		cudaFree(delayBCMasksGRGPU[i]);
		cudaFree(numBCOutPerGRGPU[i]);
		cudaFree(grConGROutGOGPU[i]);
		cudaFree(numGOOutPerGRGPU[i]);
		cudaFree(grConGOOutGRGPU[i]);
		cudaFree(numGOInPerGRGPU[i]);
		cudaFree(grConMFOutGRGPU[i]);
		cudaFree(numMFInPerGRGPU[i]);
		cudaFree(historyGRGPU[i]);

		cudaDeviceSynchronize();
	}

	// GR CUDA
	delete[] gEGRGPU;
	delete[] gEGRGPUP;
	delete[] gEGRSumGPU;
	delete[] gEDirectGPU;
	delete[] gESpilloverGPU;
	delete[] apMFtoGRGPU;
	delete[] numMFperGR;
	delete[] depAmpMFGRGPU;
	delete[] depAmpGOGRGPU;
	delete[] dynamicAmpGOGRGPU;

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

	delete[] outputGRH;
	//cudaFreeHost(outputGRH);

	// GO CUDA
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);

		cudaFreeHost(grInputGOSumH[i]);
		cudaFreeHost(apGOH[i]);
		cudaFreeHost(depAmpGOH[i]);
		cudaFreeHost(dynamicAmpGOH[i]);

		cudaFree(apGOGPU[i]);
		cudaFree(depAmpGOGPU[i]);
		cudaFree(dynamicAmpGOGPU[i]);
		cudaFree(grInputGOGPU[i]);
		cudaFree(grInputGOSumGPU[i]);

		cudaDeviceSynchronize();
	}

	delete[] grInputGOSumH;
	delete[] apGOH;
	delete[] apGOGPU;
	delete[] grInputGOGPU;
	delete[] grInputGOGPUP;
	delete[] grInputGOSumGPU;
	delete[] depAmpGOH;
	delete[] depAmpGOGPU;
	delete[] dynamicAmpGOH;
	delete[] dynamicAmpGOGPU;

	delete[] counter;

	//bc
	cudaSetDevice(gpuIndStart);
	cudaFreeHost(inputSumPFBCH);

	cudaDeviceSynchronize();
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaFreeHost(grInputBCSumH[i]);
		cudaFree(grInputBCGPU[i]);
		cudaFree(grInputBCSumGPU[i]);
		
		cudaFree(inputPFBCGPU[i]);
		cudaFree(inputSumPFBCGPU[i]);

		cudaDeviceSynchronize();
	}

	delete[] grInputBCGPU;
	delete[] grInputBCGPUP;
	delete[] grInputBCSumGPU;
	delete[] grInputBCSumH;
	
	delete[] inputPFBCGPU;
	delete[] inputPFBCGPUP;
	delete[] inputSumPFBCGPU;

	//sc
	cudaSetDevice(gpuIndStart);
	cudaFreeHost(inputSumPFSCH);

	cudaDeviceSynchronize();
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);

		cudaFree(inputPFSCGPU[i]);
		cudaFree(inputSumPFSCGPU[i]);

		cudaDeviceSynchronize();
	}

	delete[] inputPFSCGPU;
	delete[] inputPFSCGPUP;
	delete[] inputSumPFSCGPU;

	std::cout << "[INFO]: Finished deleting innet gpu arrays." << std::endl;
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
	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd;
		int cpySize;

		cpyStartInd = numGRPerGPU*i;
		cpySize     = numGRPerGPU;

		cudaSetDevice(i + gpuIndStart);

		for (int j = 0; j < max_num_p_gr_from_mf_to_gr; j++)
		{
			error = cudaMemcpy(&gMFGRT[j][cpyStartInd], (void *)((char *)gEGRGPU[i]+j*gEGRGPUP[i]),
					cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		}

		for (int j = 0; j < max_num_p_gr_from_go_to_gr; j++)
		{
			error = cudaMemcpy(&gGOGRT[j][cpyStartInd], (void *)((char *)gIGRGPU[i] + j * gIGRGPUP[i]),
					cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		}
	}

	for (int i = 0; i < max_num_p_gr_from_mf_to_gr; i++)
	{
		for (int j = 0; j < num_gr; j++)
		{
			// NOTE: gMFGR now 1D array.
			as->gMFGR[j * max_num_p_gr_from_mf_to_gr + i] = gMFGRT[i][j];
		}
	}

	for (int i = 0; i < max_num_p_gr_from_go_to_gr; i++)
	{
		for (int j = 0; j < num_gr; j++)
		{
			as->gGOGR[j * max_num_p_gr_from_go_to_gr + i] = gGOGRT[i][j];
		}
	}
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

const ct_uint8_t* InNet::exportAPGO()
{
	return (const ct_uint8_t *)as->apGO.get();
}

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
	cudaError_t error = getGRGPUData<ct_uint8_t>(outputGRGPU, outputGRH);
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
	return (const ct_uint32_t *)inputSumPFBCH;
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

const float* InNet::exportGESumGR()
{
	getGRGPUData<float>(gEGRSumGPU, as->gMFSumGR.get());
	return (const float *)as->gMFSumGR.get();
}

const float* InNet::exportGISumGR()
{
	getGRGPUData<float>(gIGRSumGPU, as->gGOSumGR.get());
	return (const float *)as->gGOSumGR.get();
}

const float* InNet::exportgSum_MFGO()
{
	return (const float *)as->gSum_MFGO.get();
}

const float* InNet::exportgSum_GRGO()
{
	return (const float *)as->gGRGO.get();
}

void InNet::updateMFActivties(const ct_uint8_t *actInMF)
{
	apMFOut = actInMF;
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_mf; i++)
		{
			as->histMF[i] = as->histMF[i] || (actInMF[i] > 0);
			for (int j = 0; j < numGPUs; j++)
			{
				apMFH[j][i] = (actInMF[i] > 0);
			}
			as->apBufMF[i] = (as->apBufMF[i] << 1) | ((actInMF[i] > 0) * 0x00000001);
		}
	}
}

void InNet::calcGOActivities()
{
	for (int i = 0; i < num_go; i++)
	{
		sumGRInputGO[i] = 0;
		for (int j = 0; j < numGPUs; j++)
		{
			sumGRInputGO[i] += grInputGOSumH[j][i];
		}
	}

#pragma omp parallel for
	for (int i = 0; i < num_go; i++)
	{
		float tempVGO = as->vGO[i];

		//NMDA Low
		float gNMDAIncGRGO = (0.00000082263 * tempVGO * tempVGO * tempVGO)
						   + (0.00021653 * tempVGO * tempVGO)
						   + (0.0195 * tempVGO)
						   + 0.6117; 

		//NMDA High
		as->gNMDAIncMFGO[i] = (0.00000011969 * tempVGO * tempVGO * tempVGO)
							+ (0.000089369 * tempVGO * tempVGO)
							+ (0.0151 * tempVGO)
							+ 0.7713; 
	
		as->gSum_MFGO[i] = (as->inputMFGO[i] * mfgoW)
						 + as->gSum_MFGO[i] * gDecMFtoGO;
		as->gSum_GOGO[i] = (as->inputGOGO[i] * gogoW * as->synWscalerGOtoGO[i])
						 +  as->gSum_GOGO[i] * gGABADecGOtoGO;
		as->gNMDAMFGO[i] = as->inputMFGO[i] * (mfgoW * NMDA_AMPAratioMFGO * as->gNMDAIncMFGO[i])
						 + as->gNMDAMFGO[i] * gDecayMFtoGONMDA;
		
		as->gGRGO[i] = (sumGRInputGO[i] * grgoW * as->synWscalerGRtoGO[i])
					 + as->gGRGO[i] * gDecGRtoGO;
		as->gGRGO_NMDA[i] = sumGRInputGO[i] * ((grgoW * as->synWscalerGRtoGO[i]) * 0.6 * gNMDAIncGRGO)
						  + as->gGRGO_NMDA[i] * gDecayMFtoGONMDA;
		
		as->threshCurGO[i] += (threshRestGO - as->threshCurGO[i]) * threshDecGO;
		
		tempVGO += (gLeakGO * (eLeakGO - tempVGO))
				+ (as->gSum_GOGO[i] * (eGABAGO - tempVGO))
				- (as->gSum_MFGO[i] + as->gGRGO[i] + as->gNMDAMFGO[i]
					+ as->gGRGO_NMDA[i]) * tempVGO
				- (as->vCoupleGO[i] * tempVGO);

		tempVGO = threshMaxGO * (tempVGO > threshMaxGO) + tempVGO * (threshMaxGO > tempVGO); /* TODO: test whether gives same results as branched case */
		
		as->apGO[i]    = tempVGO > as->threshCurGO[i];
		as->apBufGO[i] = (as->apBufGO[i] << 1) | (as->apGO[i] * 0x00000001);

		as->threshCurGO[i] = as->apGO[i] * threshMaxGO
						   + (1-as->apGO[i]) * as->threshCurGO[i];

		as->inputMFGO[i]  = 0;
		as->inputGOGO[i]  = 0;
		as->vGO[i]        = tempVGO;
	}

	for (int i = 0; i < num_go; i++)
	{
		for (int j = 0; j < numGPUs; j++)
		{
			apGOH[j][i] = as->apGO[i];
		}
	}
}

void InNet::calcSCActivities()
{
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_sc; i++)
		{
			as->gPFSC[i] = as->gPFSC[i] + inputSumPFSCH[i] * gIncGRtoSC;
			as->gPFSC[i] = as->gPFSC[i] * gDecGRtoSC;

			as->vSC[i] = as->vSC[i] + gLeakSC * (eLeakSC - as->vSC[i]) - as->gPFSC[i] * as->vSC[i];

			as->apSC[i] = (as->vSC[i] > as->threshSC[i]);
			as->apBufSC[i] = (as->apBufSC[i] << 1) | (as->apSC[i] * 0x00000001);

			as->threshSC[i] = as->threshSC[i] + threshDecSC * (threshRestSC - as->threshSC[i]);
			as->threshSC[i] = as->apSC[i] * threshMaxSC + (!as->apSC[i]) * as->threshSC[i];
		}
	}
}

void InNet::updateMFtoGROut()
{
	float recoveryRate = 1 / recoveryTauMF;

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_mf; i++)
		{
			as->depAmpMFGR[i] = apMFH[0][i] * as->depAmpMFGR[i] * fracDepMF
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
	float recoveryRate = 1 / recoveryTauMF;

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_mf; i++)
		{
			as->gi_MFtoGO[i] = apMFH[0][i] * gIncMFtoGO * as->depAmpMFGO[i]
			   + as->gi_MFtoGO[i] * gDecMFtoGO; 
			as->depAmpMFGO[i] = apMFH[0][i] * as->depAmpMFGO[i] * fracDepMF
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

void InNet::updateGOtoGROutParameters(float spillFrac)
{

	// TODO: place these in the build file as well
	float scalerGOGR = gogrW * gIncFracSpilloverGOtoGR * 1.4;
	float halfShift = 12.0;//shift;
	float steepness = 20.0;//steep; 
	float recoveryRate = 1 / recoveryTauGO;
	float baselvl = spillFrac * gogrW;

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_go; i++)
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
	float gjCoupleScaler = coupleRiRjRatioGO;
	float recoveryRate = 1 / recoveryTauGO;

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_go; i++)
		{
			
			as->gi_GOtoGO[i] =  as->apGO[i] * gGABAIncGOtoGO * as->depAmpGOGO[i]
			   + as->gi_GOtoGO[i] * gGABADecGOtoGO; 
			as->depAmpGOGO[i] = 1;
		}

		// FIXME: Performance bottleneck: convert to branchless
		for (int i = 0; i < num_go; i++)
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
		for(int i=0; i<num_go; i++)
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
	// why unsigned long?
	if (t % (unsigned long)numTSinMFHist == 0)
	{
#pragma omp parallel
		{
#pragma omp for
			for(int i = 0; i < num_mf; i++)
			{
				as->histMF[i] = false;
			}
		}
	}
}

void InNet::runGRActivitiesCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	
	float gAMPAInc = gIncDirectMFtoGR + gIncDirectMFtoGR * gIncFracSpilloverMFtoGR; 

	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callGRActKernel(sts[i][streamN], calcGRActNumBlocks, calcGRActNumGRPerB,
				vGRGPU[i], gKCaGRGPU[i], gLeakGRGPU[i], gNMDAGRGPU[i], gNMDAIncGRGPU[i], threshGRGPU[i],
				apBufGRGPU[i], outputGRGPU[i], apGRGPU[i], apMFtoGRGPU[i], gEGRSumGPU[i], 
				gIGRSumGPU[i], eLeakGR, eGOGR, gAMPAInc, threshRestGR, threshMaxGR,
				threshDecGR);
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
				inputSumPFBCGPU[i], 1, num_bc/numGPUs, 1, num_p_bc_from_gr_to_bc);
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
				inputSumPFSCGPU[i], 1, num_sc/numGPUs, 1, num_p_sc_from_gr_to_sc);
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
			num_mf*sizeof(float), cudaMemcpyHostToDevice, sts[i][streamN]);
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
			num_mf*sizeof(ct_uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
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
			num_go*sizeof(float), cudaMemcpyHostToDevice, sts[i][streamN]);
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
			num_go*sizeof(ct_uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
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
				num_mf, apMFGPU[i], depAmpMFGRGPU[i] ,gEGRGPU[i], gEGRGPUP[i],   
				grConMFOutGRGPU[i], grConMFOutGRGPUP[i],
				numMFInPerGRGPU[i], apMFtoGRGPU[i], gEGRSumGPU[i], gEDirectGPU[i], gESpilloverGPU[i], 
				gDirectDecMFtoGR, gIncDirectMFtoGR, gSpilloverDecMFtoGR,
				gIncFracSpilloverMFtoGR);
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
				updateMFInGRNumGRPerB, num_mf, depAmpMFGPU[i], grConMFOutGRGPU[i],
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

void InNet::runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateInGROPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				num_go, apGOGPU[i], dynamicAmpGOGRGPU[i], gIGRGPU[i], gIGRGPUP[i],
				grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
				numGOInPerGRGPU[i], gIGRSumGPU[i], gIDirectGPU[i], gISpilloverGPU[i], 
				gDirectDecGOtoGR, gogrW, gIncFracSpilloverGOtoGR, gSpilloverDecGOtoGR);
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
				num_mf, depAmpGOGPU[i], grConGOOutGRGPU[i], grConGOOutGRGPUP[i], numGOInPerGRGPU[i],
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
				num_mf, dynamicAmpGOGPU[i], grConGOOutGRGPU[i], grConGOOutGRGPUP[i], numGOInPerGRGPU[i],
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
				inputPFBCGPU[i], inputPFBCGPUP[i], num_p_bc_from_gr_to_bc_p2, 
				inputPFSCGPU[i], inputPFSCGPUP[i], num_p_sc_from_gr_to_sc_p2); 
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
				num_go, apBufGRGPU[i], grInputGOGPU[i], grInputGOGPUP[i],
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
				num_bc, apBufGRGPU[i], grInputBCGPU[i], grInputBCGPUP[i],
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
		error=cudaMemcpyAsync(&inputSumPFBCH[num_bc * i / numGPUs], inputSumPFBCGPU[i],
				num_bc / numGPUs * sizeof(ct_uint32_t),
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
		error=cudaMemcpyAsync(&inputSumPFSCH[num_sc * i / numGPUs], inputSumPFSCGPU[i],
				num_sc / numGPUs * sizeof(ct_uint32_t),
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

		error=cudaMemcpyAsync(grInputGOSumHost[i], grInputGOSumGPU[i], num_go * sizeof(ct_uint32_t),
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

		error=cudaMemcpyAsync(grInputBCSumHost[i], grInputBCSumGPU[i], num_bc*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, unsigned long t)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error=cudaSetDevice(i + gpuIndStart);
		if (t % (unsigned long)tsPerHistBinGR == 0)
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

	numGRPerGPU=num_gr / numGPUs;
	calcGRActNumGRPerB = 512;
	calcGRActNumBlocks=numGRPerGPU/calcGRActNumGRPerB;

	updateGRGOOutNumGRPerR = 512 * (num_go > 512) + num_go * (num_go <= 512);
	updateGRGOOutNumGRRows=numGRPerGPU/updateGRGOOutNumGRPerR;

	sumGRGOOutNumGOPerB=1024*(num_go>1024)+num_go*(num_go<=1024);
	sumGRGOOutNumBlocks=num_go/sumGRGOOutNumGOPerB;

	updateMFInGRNumGRPerB=1024*(num_mf>1024)+(num_mf<=1024)*num_mf;
	updateMFInGRNumBlocks=numGRPerGPU/updateMFInGRNumGRPerB;

	updateUBCInGRNumGRPerB=1024*(num_ubc>1024)+(num_ubc<=1024)*num_ubc;
	updateUBCInGRNumBlocks=numGRPerGPU/updateUBCInGRNumGRPerB;

	updateGOInGRNumGRPerB=1024*(num_go>=1024)+(num_go<1024)*num_go;
	updateGOInGRNumBlocks=numGRPerGPU/updateGOInGRNumGRPerB;

	updateGRBCOutNumGRPerR=512*(num_bc>512)+num_bc*(num_bc<=512);
	updateGRBCOutNumGRRows=numGRPerGPU/updateGRBCOutNumGRPerR;
	
	sumGRBCOutNumBCPerB=1024*(num_bc>1024)+num_bc*(num_bc<=1024);
	sumGRBCOutNumBlocks=num_bc/sumGRBCOutNumBCPerB;
		
	updatePFBCSCNumGRPerB=512;
	updatePFBCSCNumBlocks=numGRPerGPU/updatePFBCSCNumGRPerB;

	updateGRHistNumGRPerB=1024;
	updateGRHistNumBlocks=numGRPerGPU/updateGRHistNumGRPerB;



	std::cout << "[INFO]: Initializing per-cell cuda vars..." << std::endl;
	
	initMFCUDA();
	std::cerr << "[INFO]: Initialized MF CUDA - Last error: "
	    	  << cudaGetErrorString(cudaGetLastError()) << std::endl;
	initGRCUDA();
	std::cerr << "[INFO]: Initialized GR CUDA - Last error: "
	    	  << cudaGetErrorString(cudaGetLastError()) << std::endl;
	initGOCUDA();
	std::cerr << "[INFO]: Initialized GO CUDA - Last error: "
	    	  << cudaGetErrorString(cudaGetLastError()) << std::endl;
	initBCCUDA();
	std::cerr << "[INFO]: Initialized BC CUDA - Last error: "
	    	  << cudaGetErrorString(cudaGetLastError()) << std::endl;
	initSCCUDA();
	std::cerr << "[INFO]: Initialized SC CUDA - Last error: "
	    	  << cudaGetErrorString(cudaGetLastError()) << std::endl;

	std::cout << "[INFO]: Finished initializing per-cell cuda vars." << std::endl;
}

void InNet::initMFCUDA()
{
	apMFH		= new ct_uint32_t*[numGPUs];
	depAmpMFH	= new float*[numGPUs];
	apMFGPU		= new ct_uint32_t*[numGPUs];
	depAmpMFGPU = new float*[numGPUs];

	std::cout << "[INFO]: Allocating MF cuda variables..." << std::endl;
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMalloc((void **)&apMFGPU[i], num_mf * sizeof(ct_uint32_t));
		cudaMallocHost((void **)&apMFH[i], num_mf * sizeof(ct_uint32_t));
		cudaMalloc((void **)&(depAmpMFGPU[i]), num_mf * sizeof(float));
		cudaMallocHost((void **)&depAmpMFH[i], num_mf * sizeof(float));
		cudaDeviceSynchronize();
	}
	std::cerr << "[INFO]: Finished MF variable cuda allocation - Last Error: "
	     	  << cudaGetErrorString(cudaGetLastError()) << std::endl;

	//initialize MF GPU variables
	std::cout << "[INFO]: Initializing MF cuda variables..." << std::endl;
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemset(apMFH[i], 0, num_mf * sizeof(ct_uint32_t));
		cudaMemset(depAmpMFH[i], 1, num_mf * sizeof(float));
		cudaMemset(apMFGPU[i], 0, num_mf*sizeof(ct_uint32_t));
		cudaMemset(depAmpMFGPU[i], 1, num_mf*sizeof(float));
		cudaDeviceSynchronize();
	}
	//end copying to GPU
	std::cout << "[INFO]: Finished initializing MF cuda variables..." << std::endl;
}

void InNet::initGRCUDA()
{
	gEGRGPU			  = new float*[numGPUs];
	gEGRGPUP		  = new size_t[numGPUs];
	gEGRSumGPU		  = new float*[numGPUs];
	gEDirectGPU		  = new float*[numGPUs];
	gESpilloverGPU	  = new float*[numGPUs];
	apMFtoGRGPU		  = new int*[numGPUs];
	numMFperGR		  = new int*[numGPUs];	
	depAmpMFGRGPU	  = new float*[numGPUs];
	depAmpGOGRGPU	  = new float*[numGPUs];
	dynamicAmpGOGRGPU = new float*[numGPUs];

	gIGRGPU		   = new float*[numGPUs];
	gIGRGPUP	   = new size_t[numGPUs];
	gIGRSumGPU     = new float*[numGPUs];
	gIDirectGPU	   = new float*[numGPUs];
	gISpilloverGPU = new float*[numGPUs];

	apBufGRGPU  = new ct_uint32_t*[numGPUs];
	outputGRGPU = new ct_uint8_t*[numGPUs];
	apGRGPU		= new ct_uint32_t*[numGPUs];

	threshGRGPU	  = new float*[numGPUs];
	vGRGPU		  = new float*[numGPUs];
	gKCaGRGPU	  = new float*[numGPUs];
	gLeakGRGPU	  = new float*[numGPUs];	
	gNMDAGRGPU	  = new float*[numGPUs];
	gNMDAIncGRGPU = new float*[numGPUs];
	historyGRGPU  = new ct_uint64_t*[numGPUs];

	delayGOMasksGRGPU	 = new ct_uint32_t*[numGPUs];
	delayGOMasksGRGPUP	 = new size_t[numGPUs];
	delayBCPCSCMaskGRGPU = new ct_uint32_t*[numGPUs];
	
	delayBCMasksGRGPU  = new ct_uint32_t*[numGPUs];
	delayBCMasksGRGPUP = new size_t[numGPUs];
	grConGROutBCGPU	   = new ct_uint32_t*[numGPUs];
	grConGROutBCGPUP   = new size_t[numGPUs];
	numBCOutPerGRGPU   = new ct_int32_t*[numGPUs];

	numGOOutPerGRGPU = new ct_int32_t*[numGPUs];
	grConGROutGOGPU  = new ct_uint32_t*[numGPUs];
	grConGROutGOGPUP = new size_t[numGPUs];

	numGOInPerGRGPU  = new ct_int32_t*[numGPUs];
	grConGOOutGRGPU  = new ct_uint32_t*[numGPUs];
	grConGOOutGRGPUP = new size_t[numGPUs];

	numMFInPerGRGPU  = new ct_int32_t*[numGPUs];
	grConMFOutGRGPU  = new ct_uint32_t*[numGPUs];
	grConMFOutGRGPUP = new size_t[numGPUs];

	// NOTE: debating whether to make this page-locked mem or not (06/25/2022)
	outputGRH = new ct_uint8_t[num_gr];
	std::fill(outputGRH, outputGRH + num_gr, 0);

	std::cout << "[INFO]: Allocating GR cuda variables..." << std::endl;
	//cudaMallocHost((void **)&outputGRH, NUM_GR * sizeof(ct_uint8_t));
	//cudaMemset(outputGRH, 0, NUM_GR * sizeof(ct_uint8_t));

	//allocate memory for GPU
	for( int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMallocPitch((void **)&gEGRGPU[i], (size_t *)&gEGRGPUP[i],
			numGRPerGPU * sizeof(float), max_num_p_gr_from_mf_to_gr);
		cudaMalloc((void **)&gEGRSumGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&gEDirectGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&gESpilloverGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&apMFtoGRGPU[i], numGRPerGPU*sizeof(int));
		cudaMalloc((void **)&numMFperGR[i], numGRPerGPU*sizeof(int));
		cudaMalloc((void **)&depAmpMFGRGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&depAmpGOGRGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&dynamicAmpGOGRGPU[i], numGRPerGPU*sizeof(float));

		cudaMallocPitch((void **)&gIGRGPU[i], (size_t *)&gIGRGPUP[i],
			numGRPerGPU*sizeof(float), max_num_p_gr_from_go_to_gr);
		cudaMalloc((void **)&gIGRSumGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&gIDirectGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&gISpilloverGPU[i], numGRPerGPU*sizeof(float));

		cudaMalloc((void **)&apGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		cudaMalloc((void **)&apBufGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		cudaMalloc((void **)&outputGRGPU[i], numGRPerGPU*sizeof(ct_uint8_t));

		cudaMalloc((void **)&threshGRGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(vGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(gKCaGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(gLeakGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(gNMDAGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(gNMDAIncGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&historyGRGPU[i], numGRPerGPU*sizeof(ct_uint64_t));
		
		//variables for conduction delays
		cudaMallocPitch((void **)&delayGOMasksGRGPU[i], (size_t *)&delayGOMasksGRGPUP[i],
			numGRPerGPU * sizeof(ct_uint32_t), max_num_p_gr_from_gr_to_go);
		cudaMalloc((void **)&delayBCPCSCMaskGRGPU[i], numGRPerGPU * sizeof(ct_uint32_t));
		//end conduction delay

		//New Basket Cell stuff
		cudaMallocPitch((void **)&delayBCMasksGRGPU[i], (size_t *)&delayBCMasksGRGPUP[i],
			numGRPerGPU * sizeof(ct_uint32_t), num_bc);
		cudaMallocPitch((void **)&grConGROutBCGPU[i], (size_t *)&grConGROutBCGPUP[i],
			numGRPerGPU * sizeof(ct_uint32_t), num_bc);
		cudaMalloc((void **)&numBCOutPerGRGPU[i], numGRPerGPU * sizeof(ct_int32_t));

		//connectivity
		cudaMalloc((void **)&numGOOutPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));
		cudaMallocPitch((void **)&grConGROutGOGPU[i], (size_t *)&grConGROutGOGPUP[i],
		  	numGRPerGPU*sizeof(ct_uint32_t), max_num_p_gr_from_gr_to_go);

		cudaMalloc((void **)&numGOInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));
		cudaMallocPitch((void **)&grConGOOutGRGPU[i], (size_t *)&grConGOOutGRGPUP[i],
		  	numGRPerGPU*sizeof(ct_uint32_t), max_num_p_gr_from_go_to_gr);

		cudaMalloc((void **)&numMFInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));
		cudaMallocPitch((void **)&grConMFOutGRGPU[i], (size_t *)&grConMFOutGRGPUP[i],
			numGRPerGPU*sizeof(ct_uint32_t), max_num_p_gr_from_mf_to_gr);
		//end connectivity

		cudaDeviceSynchronize();
	}
	std::cerr << "[INFO]: Finished GR variable cuda allocation - Last Error: "
	     	  << cudaGetErrorString(cudaGetLastError()) << std::endl;

	std::cout << "[INFO]: Initializing transposed copies of act state and con state vars..." << std::endl;

	// create a transposed copy of the matrices from activity state and connectivity
	for (int i = 0; i < max_num_p_gr_from_go_to_gr; i++)
	{
		for (int j = 0; j < num_gr; j++)
		{
			gGOGRT[i][j]         = as->gGOGR[j * max_num_p_gr_from_go_to_gr + i];
			pGRfromGOtoGRT[i][j] = cs->pGRfromGOtoGR[j][i];
		}
	}

	for (int i = 0; i < max_num_p_gr_from_mf_to_gr; i++)
	{
		for (int j = 0; j < num_gr; j++)
		{
			gMFGRT[i][j]         = as->gMFGR[j * max_num_p_gr_from_mf_to_gr + i];
			pGRfromMFtoGRT[i][j] = cs->pGRfromMFtoGR[j][i];
		}
	}
	
	for (int i = 0; i < max_num_p_gr_from_gr_to_go; i++)
	{
		for (int j = 0; j < num_gr; j++)
		{
			pGRDelayfromGRtoGOT[i][j] = cs->pGRDelayMaskfromGRtoGO[j][i];
			pGRfromGRtoGOT[i][j]      = cs->pGRfromGRtoGO[j][i];
		}
	}

	std::cout << "[INFO]: Finished transposition of act state and con state vars." << std::endl;

	//initialize GR GPU variables
	std::cout << "[INFO]: Initializing GR cuda variables..." << std::endl;
	
	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd = numGRPerGPU * i;
		int cpySize		= numGRPerGPU;
		cudaSetDevice(i + gpuIndStart);

		cudaMemcpy(gKCaGRGPU[i], &(as->gKCaGR[cpyStartInd]), cpySize * sizeof(float),
			cudaMemcpyHostToDevice);

		for(int j = 0; j < max_num_p_gr_from_mf_to_gr; j++)
		{
			cudaMemcpy((void *)((char *)gEGRGPU[i] + j * gEGRGPUP[i]), &gMFGRT[j][cpyStartInd],
				cpySize * sizeof(float), cudaMemcpyHostToDevice);	
			cudaMemcpy((void *)((char *)grConMFOutGRGPU[i]+ j * grConMFOutGRGPUP[i]),
				&pGRfromMFtoGRT[j][cpyStartInd], cpySize * sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		}
	
		cudaMemcpy(vGRGPU[i], &(as->vGR[cpyStartInd]), cpySize * sizeof(float), cudaMemcpyHostToDevice);	
		cudaMemcpy(gEGRSumGPU[i], &(as->gMFSumGR[cpyStartInd]), cpySize * sizeof(float),
			cudaMemcpyHostToDevice);	
		cudaMemset(gEDirectGPU[i], 0.0, cpySize * sizeof(float));
		cudaMemset(gESpilloverGPU[i], 0.0, cpySize * sizeof(float));
		cudaMemcpy(apMFtoGRGPU[i], &(as->apMFtoGR[cpyStartInd]), cpySize * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(numMFperGR[i], &(cs->numpGRfromMFtoGR[cpyStartInd]), cpySize * sizeof(int),
			cudaMemcpyHostToDevice);	
		cudaMemset(depAmpMFGRGPU[i], 1.0, cpySize * sizeof(float));	
		cudaMemset(depAmpGOGRGPU[i], 1.0, cpySize * sizeof(float));
		cudaMemset(dynamicAmpGOGRGPU[i], 0.0, cpySize * sizeof(float));
		
		for (int j = 0; j < max_num_p_gr_from_go_to_gr; j++)
		{
			cudaMemcpy((void *)((char *)gIGRGPU[i] + j * gIGRGPUP[i]), &gGOGRT[j][cpyStartInd],
				cpySize * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy((void *)((char *)grConGOOutGRGPU[i]+j*grConGOOutGRGPUP[i]),
					&pGRfromGOtoGRT[j][cpyStartInd], cpySize * sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		}

		cudaMemcpy(gIGRSumGPU[i], &(as->gGOSumGR[cpyStartInd]), cpySize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(gIDirectGPU[i], 0.0, cpySize * sizeof(float));	
		cudaMemset(gISpilloverGPU[i], 0.0, cpySize * sizeof(float));

		cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]), cpySize * sizeof(ct_uint32_t),
			cudaMemcpyHostToDevice);
		cudaMemcpy(threshGRGPU[i], &(as->threshGR[cpyStartInd]), cpySize * sizeof(float),
			cudaMemcpyHostToDevice);

		// TODO: place initial value of gLeak into activityparams file and actually use that
		// (since its not default val of float)
		cudaMemset(gLeakGRGPU[i], 0.11, cpySize * sizeof(float));
		cudaMemset(gNMDAGRGPU[i], 0.0, cpySize * sizeof(float));
		cudaMemset(gNMDAIncGRGPU[i], 0.0, cpySize * sizeof(float));

		for (int j = 0; j < max_num_p_gr_from_gr_to_go; j++)
		{
			cudaMemcpy((void *)((char *)delayGOMasksGRGPU[i] + j * delayGOMasksGRGPUP[i]),
					&pGRDelayfromGRtoGOT[j][cpyStartInd], cpySize * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy((void *)((char *)grConGROutGOGPU[i] + j * grConGROutGOGPUP[i]),
					&pGRfromGRtoGOT[j][cpyStartInd], cpySize * sizeof(unsigned int), cudaMemcpyHostToDevice);
		}

		//Basket cell stuff
		cudaMemcpy(numGOOutPerGRGPU[i], &(cs->numpGRfromGRtoGO[cpyStartInd]),
			cpySize * sizeof(ct_int32_t), cudaMemcpyHostToDevice);

		cudaMemcpy(numGOInPerGRGPU[i], &(cs->numpGRfromGOtoGR[cpyStartInd]),
			cpySize * sizeof(ct_int32_t), cudaMemcpyHostToDevice);
		
		cudaMemcpy(delayBCPCSCMaskGRGPU[i], &(cs->pGRDelayMaskfromGRtoBSP[cpyStartInd]),
			cpySize * sizeof(ct_uint32_t), cudaMemcpyHostToDevice);

		cudaMemcpy(numMFInPerGRGPU[i], &(cs->numpGRfromMFtoGR[cpyStartInd]),
			cpySize * sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(historyGRGPU[i], &(as->historyGR[cpyStartInd]),
			cpySize * sizeof(ct_uint64_t), cudaMemcpyHostToDevice);

		cudaMemset(outputGRGPU[i], 0, cpySize * sizeof(ct_uint8_t));
		cudaMemset(apGRGPU[i], 0, cpySize * sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
	//end copying to GPU
	std::cout << "[INFO]: Finished initializing GR cuda variables..." << std::endl;
}

void InNet::initGOCUDA()
{
	//FIXME: change the types of some of these arrays (see joe's biasManip sim)
	grInputGOSumH   = new ct_uint32_t*[numGPUs];
	apGOH		    = new ct_uint32_t*[numGPUs];
	apGOGPU		    = new ct_uint32_t*[numGPUs];
	grInputGOGPU    = new ct_uint32_t*[numGPUs];
	grInputGOGPUP   = new size_t[numGPUs];
	grInputGOSumGPU = new ct_uint32_t*[numGPUs];
	depAmpGOH		= new float*[numGPUs];
	depAmpGOGPU		= new float*[numGPUs];	
	dynamicAmpGOH	= new float*[numGPUs];
	dynamicAmpGOGPU = new float*[numGPUs];

	std::cout << "[INFO]: Allocating GO cuda variables..." << std::endl;
	counter = new int[num_go];
	std::fill(counter, counter + num_go, 0);

	// allocate host and device memory	
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMallocHost((void **)&grInputGOSumH[i], num_go*sizeof(ct_uint32_t));
		cudaMallocHost((void **)&apGOH[i], num_go*sizeof(ct_uint32_t));
		cudaMallocHost((void **)&depAmpGOH[i], num_go*sizeof(float));
		cudaMallocHost((void **)&dynamicAmpGOH[i], num_go*sizeof(float));
	
		//allocate gpu memory
		cudaMalloc((void **)&apGOGPU[i], num_go*sizeof(ct_uint32_t));
		cudaMalloc((void **)&depAmpGOGPU[i], num_go*sizeof(float));
		cudaMalloc((void **)&dynamicAmpGOGPU[i], num_go*sizeof(float));

		cudaMallocPitch((void **)&grInputGOGPU[i], (size_t *)&grInputGOGPUP[i],
				num_go*sizeof(ct_uint32_t), updateGRGOOutNumGRRows);
		cudaMalloc((void **)&grInputGOSumGPU[i], num_go * sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
	std::cerr << "[INFO]: Finished GO variable cuda allocation - Last Error: "
	     	  << cudaGetErrorString(cudaGetLastError()) << std::endl;

	// initialize GO vars
	std::cout << "[INFO]: Initializing GO cuda variables..." << std::endl;
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemset(apGOH[i], 0, num_go * sizeof(ct_uint32_t));
		cudaMemset(depAmpGOH[i], 1, num_go * sizeof(float));
		cudaMemset(dynamicAmpGOH[i], 1, num_go * sizeof(float));
		cudaMemset(grInputGOSumH[i], 0, num_go * sizeof(ct_uint32_t));

		cudaMemset(apGOGPU[i], 0, num_go*sizeof(ct_uint32_t));
		cudaMemset(depAmpGOGPU[i], 1, num_go*sizeof(float));
		cudaMemset(dynamicAmpGOGPU[i], 1, num_go*sizeof(float));

		for (int j = 0; j < updateGRGOOutNumGRRows; j++)
		{
			cudaMemset(((char *)grInputGOGPU[i]+j * grInputGOGPUP[i]),
					0, num_go * sizeof(ct_uint32_t));
		}
		cudaMemset(grInputGOSumGPU[i], 0, num_go*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
	std::cout << "[INFO]: Finished initializing GO cuda variables..." << std::endl;
}
void InNet::initBCCUDA()
{
	grInputBCGPU    = new ct_uint32_t*[numGPUs];
	grInputBCGPUP   = new size_t[numGPUs];
	grInputBCSumGPU = new ct_uint32_t*[numGPUs];
	grInputBCSumH   = new ct_uint32_t*[numGPUs];
	
	inputPFBCGPU    = new ct_uint32_t*[numGPUs];
	inputPFBCGPUP   = new size_t[numGPUs];
	inputSumPFBCGPU = new ct_uint32_t*[numGPUs];

	//allocate host memory
	std::cout << "[INFO]: Allocating BC cuda variables..." << std::endl;
	cudaSetDevice(gpuIndStart);
	cudaHostAlloc((void **)&inputSumPFBCH, num_bc*sizeof(ct_uint32_t), cudaHostAllocPortable);
	cudaMemset(inputSumPFBCH, 0, num_bc * sizeof(ct_uint32_t));

	cudaDeviceSynchronize();

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMallocHost((void **)&grInputBCSumH[i], num_bc * sizeof(ct_uint32_t));
		cudaMallocPitch((void **)&grInputBCGPU[i], (size_t *)&grInputBCGPUP[i],
			num_bc * sizeof(ct_uint32_t), updateGRBCOutNumGRRows);
		cudaMalloc((void **)&grInputBCSumGPU[i], num_bc * sizeof(ct_uint32_t));

		cudaMallocPitch((void **)&inputPFBCGPU[i], (size_t *)&inputPFBCGPUP[i],
			num_p_bc_from_gr_to_bc * sizeof(ct_uint32_t), num_bc / numGPUs);
		cudaMalloc((void **)&inputSumPFBCGPU[i], num_bc / numGPUs * sizeof(ct_uint32_t));
		cudaDeviceSynchronize();
	}		
	std::cerr << "[INFO]: Finished BC variable cuda allocation - Last Error: "
	     	  << cudaGetErrorString(cudaGetLastError()) << std::endl;

	// initialize BC vars
	std::cout << "[INFO]: Initializing BC cuda variables..." << std::endl;
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemset(grInputBCSumH[i], 0, num_bc * sizeof(ct_uint32_t));
		for (int j = 0; j < updateGRBCOutNumGRRows; j++)
		{
			cudaMemset(((char *)grInputBCGPU[i]+j*grInputBCGPUP[i]),
				0, num_bc * sizeof(ct_uint32_t));
		}
		cudaMemset(grInputBCSumGPU[i], 0, num_bc * sizeof(ct_uint32_t));

		for (int j = 0; j < num_bc / numGPUs; j++)
		{
			cudaMemset(((char *)inputPFBCGPU[i] + j * inputPFBCGPUP[i]), 0,
				num_p_bc_from_gr_to_bc * sizeof(ct_uint32_t));
		}
		cudaMemset(inputSumPFBCGPU[i], 0, num_bc / numGPUs*sizeof(ct_uint32_t));
		cudaDeviceSynchronize();
	}
	std::cout << "[INFO]: Finished initializing BC cuda variables..." << std::endl;
}

void InNet::initSCCUDA()
{
	inputPFSCGPU    = new ct_uint32_t*[numGPUs];
	inputPFSCGPUP   = new size_t[numGPUs];
	inputSumPFSCGPU = new ct_uint32_t*[numGPUs];

	//allocate host memory
	std::cout << "[INFO]: Allocating SC cuda variables..." << std::endl;
	cudaSetDevice(gpuIndStart);
	cudaHostAlloc((void **)&inputSumPFSCH, num_sc * sizeof(ct_uint32_t), cudaHostAllocPortable);
	cudaMemset(inputSumPFSCH, 0, num_sc * sizeof(ct_uint32_t));

	cudaDeviceSynchronize();

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMallocPitch((void **)&inputPFSCGPU[i], (size_t *)&inputPFSCGPUP[i],
				num_p_sc_from_gr_to_sc * sizeof(ct_uint32_t), num_sc / numGPUs);
		cudaMalloc((void **)&inputSumPFSCGPU[i], num_sc / numGPUs * sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
	std::cerr << "[INFO]: Finished SC variable cuda allocation - Last Error: "
	     	  << cudaGetErrorString(cudaGetLastError()) << std::endl;

	// initialize SC vars
	std::cout << "[INFO]: Initializing SC cuda variables..." << std::endl;
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		for(int j =0; j < num_sc / numGPUs; j++)
		{
			cudaMemset(((char *)inputPFSCGPU[i] + j * inputPFSCGPUP[i]), 0,
					num_p_sc_from_gr_to_sc * sizeof(ct_uint32_t));
		}
		cudaMemset(inputSumPFSCGPU[i], 0, num_sc / numGPUs * sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
	std::cout << "[INFO]: Finished initializing SC cuda variables..." << std::endl;
}

/* =========================== PRIVATE FUNCTIONS ============================= */

template<typename Type>
cudaError_t InNet::getGRGPUData(Type **gpuData, Type *hostData)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemcpy((void *)&hostData[i * numGRPerGPU], gpuData[i],
				numGRPerGPU * sizeof(Type), cudaMemcpyDeviceToHost);
	}
	return cudaGetLastError();
}

