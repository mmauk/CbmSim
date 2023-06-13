/*
 *
 * innet.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#include <math.h>
#include <iostream>

#include "logger.h"
#include "connectivityparams.h" 
#include "activityparams.h"
#include "dynamic2darray.h"
#include "innet.h"

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

	gGOGRT = allocate2DArray<float>(max_num_p_gr_from_go_to_gr, num_gr);
	gMFGRT = allocate2DArray<float>(max_num_p_gr_from_mf_to_gr, num_gr);

	pGRDelayfromGRtoGOT = allocate2DArray<uint32_t>(max_num_p_gr_from_gr_to_go, num_gr);
	pGRfromMFtoGRT = allocate2DArray<uint32_t>(max_num_p_gr_from_mf_to_gr, num_gr);
	pGOfromMFtoGOT = allocate2DArray<uint32_t>(max_num_p_go_from_mf_to_go, num_go);
	pGOGABAInGOGOT = allocate2DArray<uint32_t>(num_con_go_to_go, num_go);
	pGOCoupInGOGOT = allocate2DArray<uint32_t>(num_p_go_to_go_gj, num_go);
	pGOCoupInGOGOCCoeffT = allocate2DArray<float>(num_p_go_to_go_gj, num_go);
	pGRfromGOtoGRT = allocate2DArray<uint32_t>(max_num_p_gr_from_go_to_gr, num_gr);
	pGRfromGRtoGOT = allocate2DArray<uint32_t>(max_num_p_gr_from_gr_to_go, num_gr);

	apBufGRHistMask = (1 << (int)tsPerHistBinGR) - 1;

	sumGRInputGO           = new uint32_t[num_go];
	sumInputGOGABASynDepGO = new float[num_go];

	initCUDA();
}

InNet::~InNet()
{
	LOG_DEBUG("Deleting innet gpu arrays.");

	//gr external to initCUDA
	delete2DArray<float>(gGOGRT);
	delete2DArray<float>(gMFGRT);

	delete2DArray<uint32_t>(pGRDelayfromGRtoGOT);
	delete2DArray<uint32_t>(pGRfromMFtoGRT);
	delete2DArray<uint32_t>(pGOfromMFtoGOT);
	delete2DArray<uint32_t>(pGOGABAInGOGOT);
	delete2DArray<uint32_t>(pGOCoupInGOGOT);
	delete2DArray<float>(pGOCoupInGOGOCCoeffT);
	delete2DArray<uint32_t>(pGRfromGOtoGRT);
	delete2DArray<uint32_t>(pGRfromGRtoGOT);

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
		cudaFree(delayGOMasksGRGPU[i]);
		
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

	// GO CUDA
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);

		cudaFreeHost(grInputGOSumH[i]);

		// new vars

		cudaFree(numMFInPerGOGPU[i]);
		cudaFree(goConMFOutGOGPU[i]);

		cudaFree(numGOInPerGOGPU[i]);
		cudaFree(goConGOOutGOGPU[i]);

		cudaFree(numGOCoupInPerGOGPU[i]);
		cudaFree(goConCoupGOInGOGPU[i]);
		cudaFree(goCoupCoeffInGOGPU[i]);

		cudaFree(vGOOutGPU[i]);
		cudaFree(vGOInGPU[i]);
		cudaFree(goIsiCounterGPU[i]);
		cudaFree(vCoupleGOGOGPU[i]);
		cudaFree(threshGOGPU[i]);
		cudaFree(apBufGOGPU[i]);
		cudaFree(inputMFGOGPU[i]);
		cudaFree(inputGOGOGPU[i]);
		cudaFree(inputGRGOGPU[i]);
		cudaFree(gSumMFGOGPU[i]);
		cudaFree(gSumGOGOGPU[i]);
		cudaFree(synWScalerGOGOGPU[i]);
		cudaFree(synWScalerGRGOGPU[i]);
		cudaFree(gNMDAMFGOGPU[i]);
		cudaFree(gNMDAIncMFGOGPU[i]);
		cudaFree(gGRGOGPU[i]);
		cudaFree(gGRGO_NMDAGPU[i]);

		// end new vars

		cudaFree(apGOOutGPU[i]);
		cudaFree(apGOInGPU[i]);
		cudaFree(dynamicAmpGOOutGPU[i]);
		cudaFree(dynamicAmpGOInGPU[i]);
		cudaFree(grInputGOGPU[i]);
		cudaFree(grInputGOSumGPU[i]);

		cudaDeviceSynchronize();
	}

	delete[] grInputGOSumH;
	delete[] grInputGOSumHost;
	delete[] apGOH;
	delete[] vGOH;

	// new vars
	delete[] numMFInPerGOGPU;
	delete[] goConMFOutGOGPU;
	delete[] goConMFOutGOGPUP;

	delete[] numGOInPerGOGPU;
	delete[] goConGOOutGOGPU;
	delete[] goConGOOutGOGPUP;

	delete[] numGOCoupInPerGOGPU;
	delete[] goConCoupGOInGOGPU;
	delete[] goConCoupGOInGOGPUP;
	delete[] goCoupCoeffInGOGPU;
	delete[] goCoupCoeffInGOGPUP;

	delete[] vGOOutGPU;
	delete[] vGOInGPU;
	delete[] goIsiCounterGPU;
	delete[] vCoupleGOGOGPU;
	delete[] threshGOGPU;
	delete[] apBufGOGPU;
	delete[] inputMFGOGPU;
	delete[] inputGOGOGPU;
	delete[] inputGRGOGPU;
	delete[] gSumMFGOGPU;
	delete[] gSumGOGOGPU;
	delete[] synWScalerGOGOGPU;
	delete[] synWScalerGRGOGPU;
	delete[] gNMDAMFGOGPU;
	delete[] gNMDAIncMFGOGPU;
	delete[] gGRGOGPU;
	delete[] gGRGO_NMDAGPU;

	// end new vars

	delete[] apGOOutGPU;
	delete[] apGOInGPU;
	delete[] grInputGOGPU;
	delete[] grInputGOGPUP;
	delete[] grInputGOSumGPU;
	delete[] dynamicAmpGOH;
	delete[] dynamicAmpGOOutGPU;
	delete[] dynamicAmpGOInGPU;
	delete[] counter;
	delete[] counter_maxes;

	LOG_DEBUG("Finished deleting innet gpu arrays.");
}

void InNet::writeToState()
{
	cudaError_t error;
	//GR variables
	// WARNING THIS IS A HORRIBLE IDEA. IF YOU GET BUGS CONSIDER THIS!
	// Reason: the apGR is a unique_ptr. it should only be modifed in the scope
	// that it is defined in.
	getGPUData<uint8_t>(outputGRGPU, as->apGR.get());
	getGPUData<uint32_t>(apBufGRGPU, as->apBufGR.get());
	getGPUData<float>(gEGRSumGPU, as->gMFSumGR.get());
	getGPUData<float>(gIGRSumGPU, as->gGOSumGR.get());

	getGPUData<float>(threshGRGPU, as->threshGR.get());
	getGPUData<float>(vGRGPU, as->vGR.get());
	getGPUData<float>(gKCaGRGPU, as->gKCaGR.get());
	getGPUData<uint64_t>(historyGRGPU, as->historyGR.get());
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
//	getGPUData<uint8_t>(outputGRGPU, as->apGR.get());
//	getGPUData<uint32_t>(apBufGRGPU, as->apBufGR.get());
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
//				cpySize*sizeof(uint32_t), cudaMemcpyHostToDevice);
//		cudaMemcpy(outputGRGPU[i], &outputGRH[cpyStartInd],
//				cpySize*sizeof(uint8_t), cudaMemcpyHostToDevice);
//	}
//}

const uint8_t* InNet::exportAPGO()
{
	return (const uint8_t *)apGOH; // keep in mind this is an explicit downcast
}

const uint8_t* InNet::exportAPMF()
{
	return (const uint8_t *)apMFOut;
}

const uint8_t* InNet::exportHistMF()
{
	return (const uint8_t *)as->histMF.get();
}

const uint8_t* InNet::exportAPGR()
{
	cudaError_t error = getGPUData<uint8_t>(outputGRGPU, outputGRH);
	return (const uint8_t *)outputGRH;
}

const uint32_t* InNet::exportSumGRInputGO()
{
	return (const uint32_t *)sumGRInputGO;
}

const float* InNet::exportSumGOInputGO()
{
	return (const float *)sumInputGOGABASynDepGO;
}

uint32_t** InNet::getApBufGRGPUPointer()
{
	return apBufGRGPU;
}

// YIKES this should be deprecated: control class should not be able
// to interact directly with the gpu pointers here!!!
uint64_t** InNet::getHistGRGPUPointer()
{
	return historyGRGPU;
}

uint32_t** InNet::getGRInputGOSumHPointer()
{
	return grInputGOSumH;
}

const float* InNet::exportGESumGR()
{
	getGPUData<float>(gEGRSumGPU, as->gMFSumGR.get());
	return (const float *)as->gMFSumGR.get();
}

const float* InNet::exportGISumGR()
{
	getGPUData<float>(gIGRSumGPU, as->gGOSumGR.get());
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

void InNet::updateMFActivties(const uint8_t *actInMF)
{
	apMFOut = actInMF;
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

void InNet::runGOActivitiesCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	
	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);

		callGOActKernel(sts[i][streamN], calcGOActNumBlocks, calcGOActNumGOPerB,
			vGOOutGPU[i], vCoupleGOGOGPU[i], threshGOGPU[i], apBufGOGPU[i], apGOOutGPU[i],
			inputMFGOGPU[i], inputGOGOGPU[i], inputGRGOGPU[i], gSumMFGOGPU[i], gSumGOGOGPU[i],
			synWScalerGOGOGPU[i], synWScalerGRGOGPU[i], NMDA_AMPAratioMFGO, gNMDAMFGOGPU[i], 
			gNMDAIncMFGOGPU[i], gGRGOGPU[i], gGRGO_NMDAGPU[i], gLeakGO, eLeakGO, threshRestGO, threshMaxGO,
			threshDecGO, mfgoW, gogoW, grgoW, gDecMFtoGO, gGABADecGOtoGO, gDecayMFtoGONMDA,
			gDecGRtoGO, eGABAGO);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"goActivityCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyApGODevicetoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd = numGOPerGPU * i;
		int cpySize     = numGOPerGPU;
		error = cudaSetDevice(i + gpuIndStart);
		error = cudaMemcpyAsync(apGOH + cpyStartInd, apGOOutGPU[i],
			cpySize * sizeof(uint32_t), cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::cpyVGODevicetoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd = numGOPerGPU * i;
		int cpySize     = numGOPerGPU;
		error = cudaSetDevice(i + gpuIndStart);
		error = cudaMemcpyAsync(vGOH + cpyStartInd, vGOOutGPU[i],
			cpySize * sizeof(float), cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::updateMFtoGROut()
{
	float recoveryRate = 1 / recoveryTauMF;

	for (int i = 0; i < num_mf; i++)
	{
		as->depAmpMFGR[i] = apMFH[0][i] * as->depAmpMFGR[i] * fracDepMF
		   + (!apMFH[0][i]) * (as->depAmpMFGR[i] + recoveryRate * (1 - as->depAmpMFGR[i])); 

		for (int j = 0; j < numGPUs; j++)
		{
			depAmpMFH[j][i] = as->depAmpMFGR[i];
		}
	}
}

void InNet::runUpdateMFInGOCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
		callUpdateMFInGOOPKernel(sts[i][streamN], calcGOActNumBlocks, calcGOActNumGOPerB,
				num_mf, apMFGPU[i], goConMFOutGOGPU[i], goConMFOutGOGPUP[i],
				numMFInPerGOGPU[i], inputMFGOGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGOCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::runUpdateGOInGOCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
		callUpdateGOInGOOPKernel(sts[i][streamN], updateGOInGONumBlocks, updateGOInGONumGOPerB,
				num_go, apGOInGPU[i], goConGOOutGOGPU[i], goConGOOutGOGPUP[i],
				numGOInPerGOGPU[i], inputGOGOGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGOCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

// for now just going with same num blocks and go per block as runUpdateGOInGOCUDA
// as this fn could have easily been in that one
void InNet::runUpdateGOCoupInGOCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
		callUpdateGOCoupInGOOPKernel(sts[i][streamN], updateGOInGONumBlocks, updateGOInGONumGOPerB,
				num_go, vGOInGPU[i], goConCoupGOInGOGPU[i], goConCoupGOInGOGPUP[i], numGOCoupInPerGOGPU[i],
				coupleRiRjRatioGO, goCoupCoeffInGOGPU[i], goCoupCoeffInGOGPUP[i], vCoupleGOGOGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGOCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

// again, for now just going with same num blocks and go per block as runUpdateGOInGOCUDA. can
// make different values in future to test out block/thread combinations

// FIXME: need to think about dynamicAmpGOGPU var as its used when updating dynamicAmpGOGRGPU input
// variable due to array alloc size diffs! (dynamicAmpGOGPU should have length numGOPerGPU per device,
// and dynamicAmpGOGRGPU should have length num_go as we need all go available in GR device kernels)
void InNet::runUpdateGOOutGRDynamicSpillCUDA(cudaStream_t **sts, int streamN, float spillFrac)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
		callUpdateGOOutGRDynamicSpillOPKernel(sts[i][streamN], updateGOInGONumBlocks, updateGOInGONumGOPerB,
			spillFrac, gIncFracSpilloverGOtoGR, gogrW, apGOOutGPU[i],
			goIsiCounterGPU[i], dynamicAmpGOOutGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateGOOutGRDynamicSpillCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyGOGRDynamicSpillGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd = numGOPerGPU * i;
		int cpySize     = numGOPerGPU;
		error = cudaSetDevice(i + gpuIndStart);
		error=cudaMemcpyAsync(dynamicAmpGOH + cpyStartInd, dynamicAmpGOOutGPU[i],
			cpySize * sizeof(float), cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
} 

void InNet::resetMFHist(uint32_t t)
{
	if (t % (uint32_t)numTSinMFHist == 0)
	{
		for(int i = 0; i < num_mf; i++)
		{
			as->histMF[i] = false;
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
			num_mf*sizeof(uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPMFHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

//void InNet::cpyDepAmpUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN) {}
//void InNet::cpyAPUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN) {}

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

//void InNet::runUpdateUBCInGRDepressionCUDA(cudaStream_t **sts, int streamN) {}
//void InNet::runUpdateUBCInGRCUDA(cudaStream_t **sts, int streamN) {}

void InNet::runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateInGROPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				num_go, apGOInGPU[i], dynamicAmpGOGRGPU[i], gIGRGPU[i], gIGRGPUP[i],
				grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
				numGOInPerGRGPU[i], gIGRSumGPU[i], gIDirectGPU[i], gISpilloverGPU[i], 
				gDirectDecGOtoGR, gogrW, gSpilloverDecGOtoGR, gIncFracSpilloverGOtoGR);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateGOInGRCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyGOGRDynamicSpillHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(dynamicAmpGOInGPU[i], dynamicAmpGOH, num_go * sizeof(float),
				cudaMemcpyHostToDevice, sts[i][streamN]);
	}
}

void InNet::cpyApGOHosttoDeviceCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
		error = cudaMemcpyAsync(apGOInGPU[i], apGOH, num_go * sizeof(uint32_t),
				cudaMemcpyHostToDevice, sts[i][streamN]);
	}
}

void InNet::cpyVGOHosttoDeviceCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
		error = cudaMemcpyAsync(vGOInGPU[i], vGOH, num_go * sizeof(float),
				cudaMemcpyHostToDevice, sts[i][streamN]);
	}
}

void InNet::runUpdateGOInGRDynamicSpillCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdateGOInGRDynamicSpillOPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				num_go, dynamicAmpGOInGPU[i], grConGOOutGRGPU[i], grConGOOutGRGPUP[i], numGOInPerGRGPU[i],
				dynamicAmpGOGRGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGRDynamicSpillCUDA: kernel launch for gpu #"<<i<<
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

void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);

		error=cudaMemcpyAsync(grInputGOSumH[i], grInputGOSumGPU[i], num_go * sizeof(uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::runSumReductionGRGOInputHost()
{
	for (int i = 0; i < numGPUs; i++)
	{
		for (int j = 0; j < num_go; j++)
		{
			grInputGOSumHost[j] += grInputGOSumH[i][j];
		}
	}
}

void InNet::cpyGRGOSumHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd = numGOPerGPU * i;
		int cpySize     = numGOPerGPU;
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(inputGRGOGPU[i], grInputGOSumHost + cpyStartInd, cpySize * sizeof(uint32_t),
				cudaMemcpyHostToDevice, sts[i][streamN]);
	}
}

void InNet::runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, uint32_t t)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error=cudaSetDevice(i + gpuIndStart);
		if (t % (uint32_t)tsPerHistBinGR == 0)
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
	LOG_DEBUG("Maximum number of CUDA devices: %d", maxNumGPUs);
	LOG_DEBUG("%s", cudaGetErrorString(error));
	LOG_DEBUG("Number of CUDA devices actually used: %d", numGPUs);
	LOG_DEBUG("Lowest CUDA device index: %d", gpuIndStart);

	numGRPerGPU = num_gr / numGPUs;
	calcGRActNumGRPerB = 512;
	calcGRActNumBlocks = numGRPerGPU / calcGRActNumGRPerB;

	// new vars

	numGOPerGPU = num_go / numGPUs;
	calcGOActNumGOPerB = 512;
	calcGOActNumBlocks = numGOPerGPU / calcGOActNumGOPerB;

	// end new vars

	updateGRGOOutNumGRPerR = 512 * (num_go > 512) + num_go * (num_go <= 512);
	updateGRGOOutNumGRRows = numGRPerGPU / updateGRGOOutNumGRPerR;

	sumGRGOOutNumGOPerB = 1024 * (num_go > 1024) + num_go * (num_go <= 1024);
	sumGRGOOutNumBlocks = num_go / sumGRGOOutNumGOPerB;

	// new vars

	updateMFInGOnumGOPerB = 512 * (num_go > 512) + num_go * (num_go <= 512); 
	updateMFInGONumBlocks = numGOPerGPU / updateMFInGOnumGOPerB;

	updateGOInGONumGOPerB = 512 * (num_go > 512) + num_go * (num_go <= 512);
	updateGOInGONumBlocks = numGOPerGPU / updateGOInGONumGOPerB;

	// end new vars

	updateMFInGRNumGRPerB = 1024 * (num_mf > 1024) + (num_mf <= 1024) * num_mf;
	updateMFInGRNumBlocks = numGRPerGPU / updateMFInGRNumGRPerB;

	updateUBCInGRNumGRPerB = 1024 * (num_ubc > 1024) + (num_ubc <= 1024) * num_ubc;
	updateUBCInGRNumBlocks = numGRPerGPU / updateUBCInGRNumGRPerB;

	updateGOInGRNumGRPerB = 1024 * (num_go >= 1024) + (num_go < 1024) * num_go;
	updateGOInGRNumBlocks = numGRPerGPU / updateGOInGRNumGRPerB;

	updateGRHistNumGRPerB = 1024;
	updateGRHistNumBlocks = numGRPerGPU / updateGRHistNumGRPerB;

	LOG_DEBUG("Initializing per-cell cuda vars...");
	
	initMFCUDA();
	LOG_DEBUG("Initialized MF CUDA");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));
	initGRCUDA();
	LOG_DEBUG("Initialized GR CUDA");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));
	initGOCUDA();
	LOG_DEBUG("Initialized GO CUDA");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));
	LOG_DEBUG("Finished initializing per-cell cuda variables.");
}

void InNet::initMFCUDA()
{
	apMFH		= new uint32_t*[numGPUs];
	depAmpMFH	= new float*[numGPUs];
	apMFGPU		= new uint32_t*[numGPUs];
	depAmpMFGPU = new float*[numGPUs];

	LOG_DEBUG("Allocating MF cuda variables...");
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMalloc((void **)&apMFGPU[i], num_mf * sizeof(uint32_t));
		cudaMallocHost((void **)&apMFH[i], num_mf * sizeof(uint32_t));
		cudaMalloc((void **)&(depAmpMFGPU[i]), num_mf * sizeof(float));
		cudaMallocHost((void **)&depAmpMFH[i], num_mf * sizeof(float));
		cudaDeviceSynchronize();
	}
	LOG_DEBUG("Finished MF variable cuda allocation");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

	//initialize MF GPU variables
	LOG_DEBUG("Initializing MF cuda variables...");
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemset(apMFH[i], 0, num_mf * sizeof(uint32_t));
		cudaMemset(depAmpMFH[i], 1, num_mf * sizeof(float));
		cudaMemset(apMFGPU[i], 0, num_mf*sizeof(uint32_t));
		cudaMemset(depAmpMFGPU[i], 1, num_mf*sizeof(float));
		cudaDeviceSynchronize();
	}
	//end copying to GPU
	LOG_DEBUG("Finished initializing MF cuda variables.");
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

	apBufGRGPU  = new uint32_t*[numGPUs];
	outputGRGPU = new uint8_t*[numGPUs];
	apGRGPU		= new uint32_t*[numGPUs];

	threshGRGPU	  = new float*[numGPUs];
	vGRGPU		  = new float*[numGPUs];
	gKCaGRGPU	  = new float*[numGPUs];
	gLeakGRGPU	  = new float*[numGPUs];	
	gNMDAGRGPU	  = new float*[numGPUs];
	gNMDAIncGRGPU = new float*[numGPUs];
	historyGRGPU  = new uint64_t*[numGPUs];

	delayGOMasksGRGPU	 = new uint32_t*[numGPUs];
	delayGOMasksGRGPUP	 = new size_t[numGPUs];
	
	numGOOutPerGRGPU = new int32_t*[numGPUs];
	grConGROutGOGPU  = new uint32_t*[numGPUs];
	grConGROutGOGPUP = new size_t[numGPUs];

	numGOInPerGRGPU  = new int32_t*[numGPUs];
	grConGOOutGRGPU  = new uint32_t*[numGPUs];
	grConGOOutGRGPUP = new size_t[numGPUs];

	numMFInPerGRGPU  = new int32_t*[numGPUs];
	grConMFOutGRGPU  = new uint32_t*[numGPUs];
	grConMFOutGRGPUP = new size_t[numGPUs];

	// NOTE: debating whether to make this page-locked mem or not (06/25/2022)
	outputGRH = new uint8_t[num_gr];
	memset(outputGRH, 0, num_gr * sizeof(uint8_t));

	LOG_DEBUG("Allocating GR cuda variables...");

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

		cudaMalloc((void **)&apGRGPU[i], numGRPerGPU*sizeof(uint32_t));
		cudaMalloc((void **)&apBufGRGPU[i], numGRPerGPU*sizeof(uint32_t));
		cudaMalloc((void **)&outputGRGPU[i], numGRPerGPU*sizeof(uint8_t));

		cudaMalloc((void **)&threshGRGPU[i], numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(vGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(gKCaGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(gLeakGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(gNMDAGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc<float>(&(gNMDAIncGRGPU[i]), numGRPerGPU*sizeof(float));
		cudaMalloc((void **)&historyGRGPU[i], numGRPerGPU*sizeof(uint64_t));
		
		//variables for conduction delays
		cudaMallocPitch((void **)&delayGOMasksGRGPU[i], (size_t *)&delayGOMasksGRGPUP[i],
			numGRPerGPU * sizeof(uint32_t), max_num_p_gr_from_gr_to_go);
		//end conduction delay

		//connectivity
		cudaMalloc((void **)&numGOOutPerGRGPU[i], numGRPerGPU*sizeof(int32_t));
		cudaMallocPitch((void **)&grConGROutGOGPU[i], (size_t *)&grConGROutGOGPUP[i],
		  	numGRPerGPU*sizeof(uint32_t), max_num_p_gr_from_gr_to_go);

		cudaMalloc((void **)&numGOInPerGRGPU[i], numGRPerGPU*sizeof(int32_t));
		cudaMallocPitch((void **)&grConGOOutGRGPU[i], (size_t *)&grConGOOutGRGPUP[i],
		  	numGRPerGPU*sizeof(uint32_t), max_num_p_gr_from_go_to_gr);

		cudaMalloc((void **)&numMFInPerGRGPU[i], numGRPerGPU*sizeof(int32_t));
		cudaMallocPitch((void **)&grConMFOutGRGPU[i], (size_t *)&grConMFOutGRGPUP[i],
			numGRPerGPU*sizeof(uint32_t), max_num_p_gr_from_mf_to_gr);
		//end connectivity

		cudaDeviceSynchronize();
	}
	LOG_DEBUG("Finished GR variable cuda allocation");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

	LOG_DEBUG("Initializing transposed copies of act state and con state vars...");

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

	LOG_DEBUG("Finished transposition of act state and con state vars.");

	//initialize GR GPU variables
	LOG_DEBUG("Initializing GR cuda variables...");
	
	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd = numGRPerGPU * i;
		int cpySize     = numGRPerGPU;
		cudaSetDevice(i + gpuIndStart);

		cudaMemcpy(gKCaGRGPU[i], &(as->gKCaGR[cpyStartInd]), cpySize * sizeof(float),
			cudaMemcpyHostToDevice);

		for(int j = 0; j < max_num_p_gr_from_mf_to_gr; j++)
		{
			cudaMemcpy((void *)((char *)gEGRGPU[i] + j * gEGRGPUP[i]), &gMFGRT[j][cpyStartInd],
				cpySize * sizeof(float), cudaMemcpyHostToDevice);	
			cudaMemcpy((void *)((char *)grConMFOutGRGPU[i]+ j * grConMFOutGRGPUP[i]),
				&pGRfromMFtoGRT[j][cpyStartInd], cpySize * sizeof(uint32_t), cudaMemcpyHostToDevice);
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
					&pGRfromGOtoGRT[j][cpyStartInd], cpySize * sizeof(uint32_t), cudaMemcpyHostToDevice);
		}

		cudaMemcpy(gIGRSumGPU[i], &(as->gGOSumGR[cpyStartInd]), cpySize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(gIDirectGPU[i], 0.0, cpySize * sizeof(float));	
		cudaMemset(gISpilloverGPU[i], 0.0, cpySize * sizeof(float));

		cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]), cpySize * sizeof(uint32_t),
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
			cpySize * sizeof(int32_t), cudaMemcpyHostToDevice);

		cudaMemcpy(numGOInPerGRGPU[i], &(cs->numpGRfromGOtoGR[cpyStartInd]),
			cpySize * sizeof(int32_t), cudaMemcpyHostToDevice);
		
		cudaMemcpy(numMFInPerGRGPU[i], &(cs->numpGRfromMFtoGR[cpyStartInd]),
			cpySize * sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(historyGRGPU[i], &(as->historyGR[cpyStartInd]),
			cpySize * sizeof(uint64_t), cudaMemcpyHostToDevice);

		cudaMemset(outputGRGPU[i], 0, cpySize * sizeof(uint8_t));
		cudaMemset(apGRGPU[i], 0, cpySize * sizeof(uint32_t));

		cudaDeviceSynchronize();
	}
	//end copying to GPU
	LOG_DEBUG("Finished initializing GR cuda variables.");
}

void InNet::initGOCUDA()
{

	// new vars

	numMFInPerGOGPU  = new int32_t*[numGPUs];
	goConMFOutGOGPU  = new uint32_t*[numGPUs];
	goConMFOutGOGPUP = new size_t[numGPUs];

	numGOInPerGOGPU  = new int32_t*[numGPUs];
	goConGOOutGOGPU  = new uint32_t*[numGPUs];
	goConGOOutGOGPUP = new size_t[numGPUs];

	numGOCoupInPerGOGPU = new int32_t*[numGPUs];
	goConCoupGOInGOGPU = new uint32_t*[numGPUs];
	goConCoupGOInGOGPUP = new size_t[numGPUs];
	goCoupCoeffInGOGPU = new float*[numGPUs];
	goCoupCoeffInGOGPUP = new size_t[numGPUs];

	vGOOutGPU            = new float*[numGPUs];
	vGOInGPU            = new float*[numGPUs];
	goIsiCounterGPU   = new uint32_t*[numGPUs];
	vCoupleGOGOGPU    = new float*[numGPUs];
	threshGOGPU       = new float*[numGPUs];
	apBufGOGPU        = new uint32_t*[numGPUs];
	inputMFGOGPU      = new uint32_t*[numGPUs];
	inputGOGOGPU      = new uint32_t*[numGPUs];
	inputGRGOGPU      = new uint32_t*[numGPUs];
	gSumMFGOGPU       = new float*[numGPUs];
	gSumGOGOGPU       = new float*[numGPUs];
	synWScalerGOGOGPU = new float*[numGPUs];
	synWScalerGRGOGPU = new float*[numGPUs];
	gNMDAMFGOGPU      = new float*[numGPUs];
	gNMDAIncMFGOGPU   = new float*[numGPUs];
	gGRGOGPU          = new float*[numGPUs];
	gGRGO_NMDAGPU     = new float*[numGPUs];

	// end new vars

	grInputGOSumH   = new uint32_t*[numGPUs];
	grInputGOSumHost = new uint32_t[num_go];
	memset(grInputGOSumHost, 0, num_go * sizeof(uint32_t));

	apGOH		    = new uint32_t[num_go];
	memset(apGOH, 0, num_go * sizeof(uint32_t));
	vGOH		    = new float[num_go];
	memset(vGOH, 0, num_go * sizeof(float)); // this arr is copied to *first* before its read so can init to 0
	apGOOutGPU		    = new uint32_t*[numGPUs];
	apGOInGPU		    = new uint32_t*[numGPUs];
	grInputGOGPU    = new uint32_t*[numGPUs];
	grInputGOGPUP   = new size_t[numGPUs];
	grInputGOSumGPU = new uint32_t*[numGPUs];
	dynamicAmpGOH	= new float[num_go];
	memset(dynamicAmpGOH, 1, num_go * sizeof(uint32_t));
	dynamicAmpGOOutGPU = new float*[numGPUs];
	dynamicAmpGOInGPU = new float*[numGPUs];

	LOG_DEBUG("Allocating GO cuda variables...");
	counter = new int[num_go];
	memset(counter, 0, num_go * sizeof(int));
	counter_maxes = new int[num_go];
	memset(counter_maxes, 0, num_go * sizeof(int));

	// allocate host and device memory
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMallocHost((void **)&grInputGOSumH[i], num_go * sizeof(uint32_t));

		// new vars

		cudaMalloc((void **)&numMFInPerGOGPU[i], numGOPerGPU*sizeof(int32_t));
		cudaMallocPitch((void **)&goConMFOutGOGPU[i], (size_t *)&goConMFOutGOGPUP[i],
			numGOPerGPU * sizeof(uint32_t), max_num_p_go_from_mf_to_go);

		cudaMalloc((void **)&numGOInPerGOGPU[i], numGOPerGPU*sizeof(int32_t));
		cudaMallocPitch((void **)&goConGOOutGOGPU[i], (size_t *)&goConGOOutGOGPUP[i],
			numGOPerGPU * sizeof(uint32_t), num_con_go_to_go);

		cudaMalloc((void **)&numGOCoupInPerGOGPU[i], numGOPerGPU * sizeof(int32_t));
		cudaMallocPitch((void **)&goConCoupGOInGOGPU[i], (size_t *)&goConCoupGOInGOGPUP[i],
			numGOPerGPU * sizeof(uint32_t), num_p_go_to_go_gj);
		cudaMallocPitch((void **)&goCoupCoeffInGOGPU[i], (size_t *)&goCoupCoeffInGOGPUP[i],
			numGOPerGPU * sizeof(uint32_t), num_p_go_to_go_gj);

		cudaMalloc((void **)&vGOOutGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&vGOInGPU[i], num_go * sizeof(float));
		cudaMalloc((void **)&goIsiCounterGPU[i], numGOPerGPU * sizeof(uint32_t));
		cudaMalloc((void **)&vCoupleGOGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&threshGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&apBufGOGPU[i], numGOPerGPU * sizeof(uint32_t));
		cudaMalloc((void **)&inputMFGOGPU[i], numGOPerGPU * sizeof(uint32_t));
		cudaMalloc((void **)&inputGOGOGPU[i], numGOPerGPU * sizeof(uint32_t));
		cudaMalloc((void **)&inputGRGOGPU[i], numGOPerGPU * sizeof(uint32_t));
		cudaMalloc((void **)&gSumMFGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&gSumGOGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&synWScalerGOGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&synWScalerGRGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&gNMDAMFGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&gNMDAIncMFGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&gGRGOGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&gGRGO_NMDAGPU[i], numGOPerGPU * sizeof(float));

		// end new vars

		//allocate gpu memory
		cudaMalloc((void **)&apGOOutGPU[i], numGOPerGPU*sizeof(uint32_t));
		cudaMalloc((void **)&apGOInGPU[i], num_go*sizeof(uint32_t));
		cudaMalloc((void **)&dynamicAmpGOOutGPU[i], numGOPerGPU * sizeof(float));
		cudaMalloc((void **)&dynamicAmpGOInGPU[i], num_go * sizeof(float));

		cudaMallocPitch((void **)&grInputGOGPU[i], (size_t *)&grInputGOGPUP[i],
				num_go*sizeof(uint32_t), updateGRGOOutNumGRRows);
		cudaMalloc((void **)&grInputGOSumGPU[i], num_go * sizeof(uint32_t));

		cudaDeviceSynchronize();
	}
	LOG_DEBUG("Finished GO variable cuda allocation");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

	// initialize GO vars
	LOG_DEBUG("Initializing GO cuda variables...");

	for (int i = 0; i < max_num_p_go_from_mf_to_go; i++)
	{
		for (int j = 0; j < num_go; j++)
		{
			pGOfromMFtoGOT[i][j] = cs->pGOfromMFtoGO[j][i];
		}
	}

	for (int i = 0; i < num_con_go_to_go; i++)
	{
		for (int j = 0; j < num_go; j++)
		{
			pGOGABAInGOGOT[i][j] = cs->pGOGABAInGOGO[j][i];
		}
	}

	for (int i = 0; i < num_p_go_to_go_gj; i++)
	{
		for (int j = 0; j < num_go; j++)
		{
			pGOCoupInGOGOT[i][j] = cs->pGOCoupInGOGO[j][i];
		}
	}

	for (int i = 0; i < num_p_go_to_go_gj; i++)
	{
		for (int j = 0; j < num_go; j++)
		{
			pGOCoupInGOGOCCoeffT[i][j] = cs->pGOCoupInGOGOCCoeff[j][i];
		}
	}

	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd = numGOPerGPU * i;
		int cpySize     = numGOPerGPU;

		cudaSetDevice(i + gpuIndStart);
		cudaMemset(grInputGOSumH[i], 0, num_go * sizeof(uint32_t));

		// new vars
		cudaMemset(goIsiCounterGPU[i], 0, numGOPerGPU * sizeof(uint32_t));
		cudaMemset(vCoupleGOGOGPU[i], 0, numGOPerGPU * sizeof(float));
		cudaMemset(apBufGOGPU[i], 0, numGOPerGPU * sizeof(uint32_t));
		cudaMemset(inputMFGOGPU[i], 0, numGOPerGPU * sizeof(uint32_t));
		cudaMemset(inputGOGOGPU[i], 0, numGOPerGPU * sizeof(uint32_t));
		cudaMemset(inputGRGOGPU[i], 0, numGOPerGPU * sizeof(uint32_t));
		cudaMemset(gSumMFGOGPU[i], 0, numGOPerGPU * sizeof(float));
		cudaMemset(gSumGOGOGPU[i], 0, numGOPerGPU * sizeof(float));
		cudaMemset(gNMDAMFGOGPU[i], 0, numGOPerGPU * sizeof(float));
		cudaMemset(gNMDAIncMFGOGPU[i], 0, numGOPerGPU * sizeof(float));
		cudaMemset(gGRGOGPU[i], 0, numGOPerGPU * sizeof(float));
		cudaMemset(gGRGO_NMDAGPU[i], 0, numGOPerGPU * sizeof(float));

		cudaMemcpy(vGOOutGPU[i], as->vGO.get() + cpyStartInd,
			cpySize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemset(vGOInGPU[i], 0, num_go * sizeof(float));

		cudaMemcpy(threshGOGPU[i], as->threshCurGO.get() + cpyStartInd,
			cpySize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(synWScalerGOGOGPU[i], as->synWscalerGOtoGO.get() + cpyStartInd,
			cpySize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(synWScalerGRGOGPU[i], as->synWscalerGRtoGO.get() + cpyStartInd,
			cpySize * sizeof(float), cudaMemcpyHostToDevice);

		cudaMemcpy(numMFInPerGOGPU[i], &(cs->numpGOfromMFtoGO[cpyStartInd]),
			cpySize * sizeof(int32_t), cudaMemcpyHostToDevice);

		for (int j = 0; j < max_num_p_go_from_mf_to_go; j++)
		{
			cudaMemcpy((void *)((char *)goConMFOutGOGPU[i]+ j * goConMFOutGOGPUP[i]),
				&pGOfromMFtoGOT[j][cpyStartInd], cpySize * sizeof(uint32_t), cudaMemcpyHostToDevice);
		}

		cudaMemcpy(numGOInPerGOGPU[i], &(cs->numpGOGABAInGOGO[cpyStartInd]),
			cpySize * sizeof(int32_t), cudaMemcpyHostToDevice);

		for (int j = 0; j < num_con_go_to_go; j++)
		{
			cudaMemcpy((void *)((char *)goConGOOutGOGPU[i]+ j * goConGOOutGOGPUP[i]),
				&pGOGABAInGOGOT[j][cpyStartInd], cpySize * sizeof(uint32_t), cudaMemcpyHostToDevice);
		}

		cudaMemcpy(numGOCoupInPerGOGPU[i], &(cs->numpGOCoupInGOGO[cpyStartInd]),
			cpySize * sizeof(int32_t), cudaMemcpyHostToDevice);

		for (int j = 0; j < num_p_go_to_go_gj; j++)
		{
			cudaMemcpy((void *)((char *)goConCoupGOInGOGPU[i] + j * goConCoupGOInGOGPUP[i]),
				&pGOCoupInGOGOT[j][cpyStartInd], cpySize * sizeof(uint32_t), cudaMemcpyHostToDevice);
		}

		for (int j = 0; j < num_p_go_to_go_gj; j++)
		{
			cudaMemcpy((void *)((char *)goCoupCoeffInGOGPU[i] + j * goCoupCoeffInGOGPUP[i]),
				&pGOCoupInGOGOCCoeffT[j][cpyStartInd], cpySize * sizeof(float), cudaMemcpyHostToDevice);
		}

		// end new vars

		cudaMemset(apGOOutGPU[i], 0, numGOPerGPU*sizeof(uint32_t));
		cudaMemset(apGOInGPU[i], 0, num_go*sizeof(uint32_t));
		cudaMemset(dynamicAmpGOOutGPU[i], 1, numGOPerGPU*sizeof(float));
		cudaMemset(dynamicAmpGOInGPU[i], 1, num_go * sizeof(float));

		for (int j = 0; j < updateGRGOOutNumGRRows; j++)
		{
			cudaMemset(((char *)grInputGOGPU[i]+j * grInputGOGPUP[i]),
					0, num_go * sizeof(uint32_t));
		}
		cudaMemset(grInputGOSumGPU[i], 0, num_go*sizeof(uint32_t));

		cudaDeviceSynchronize();
	}
	LOG_DEBUG("Finished initializing GO cuda variables.");
}

/* =========================== PRIVATE FUNCTIONS ============================= */

template<typename Type>
cudaError_t InNet::getGPUData(Type **gpuData, Type *hostData)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemcpy((void *)&hostData[i * numGRPerGPU], gpuData[i],
				numGRPerGPU * sizeof(Type), cudaMemcpyDeviceToHost);
	}
	return cudaGetLastError();
}

