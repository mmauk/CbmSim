/*
 * innetallgrmfgo.cpp
 *
 *  Created on: Nov 25, 2013
 *      Author: consciousness
 */

//#include "innetmodules/innetallgrmfgo.h"
//
//using namespace std;
//
//InNetAllGRMFGO::InNetAllGRMFGO(ConnectivityParams *conParams, ActivityParams *actParams,
//			InNetConnectivityState *conState, InNetActivityState *actState,
//			int gpuIndStart, int numGPUs)
//			:InNet(conParams, actParams, conState, actState, gpuIndStart, numGPUs)
//{
//	initAddGOCUDA();
//}
//
//InNetAllGRMFGO::~InNetAllGRMFGO()
//{
//	for(int i=0; i<numGPUs; i++)
//	{
//		cudaSetDevice(i+gpuIndStart);
//
//		cudaFree(totalGRGPU[i]);
//	}
//
//	delete[] totalGRGPU;
//}
//
//void InNetAllGRMFGO::initAddGOCUDA()
//{
//	totalGRGPU=new ct_uint32_t *[numGPUs];
//
//	for(int i=0; i<numGPUs; i++)
//	{
//		cudaSetDevice(i+gpuIndStart);
//
//		cudaMalloc((void **)&totalGRGPU[i], sizeof(ct_uint32_t));
//
//		cudaMemset(totalGRGPU[i], 0, sizeof(ct_uint32_t));
//
//		cudaDeviceSynchronize();
//	}
//}
//
//void InNetAllGRMFGO::updateMFtoGOOut()
//{
//	ct_uint32_t mfTotal;
//
//	mfTotal=0;
//
//	for(int i=0; i<cp->numMF; i++)
//	{
//		mfTotal+=apMFH[0][i];
//	}
//
//	for(int i=0; i<cp->numGO; i++)
//	{
//		as->inputMFGO[i]=mfTotal;
//	}
//}
//
//void InNetAllGRMFGO::runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN)
//{
//	cudaError_t error;
//
//	for(int i=0; i<numGPUs; i++)
//	{
//		error=cudaSetDevice(i+gpuIndStart);
//
//		callSumKernel<ct_uint32_t, false, false>(sts[i][streamN], apGRGPU[i],
//				1, grInputGOSumGPU[i], 1, 1, cp->numGO, numGRPerGPU);
//		callSumKernel<ct_uint32_t, false, false>(sts[i][streamN], grInputGOSumGPU[i],
//				1, totalGRGPU[i], 1, 1, 1, cp->numGO);
//	}
//}
//
//void InNetAllGRMFGO::runSumGRGOOutCUDA(cudaStream_t **sts, int streamN)
//{
//	cudaError_t error;
//	for(int i=0; i<numGPUs; i++)
//	{
//		error=cudaSetDevice(i+gpuIndStart);
//		callBroadcastKernel<ct_uint32_t>(sts[i][streamN], totalGRGPU[i], grInputGOSumGPU[i],
//				2, cp->numGO);
//	}
//}

