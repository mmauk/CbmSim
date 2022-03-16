/*
 * innet.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */


#include "innetmodules/innet.h"

using namespace std;

InNet::InNet(ConnectivityParams *conParams, ActivityParams *actParams,
		InNetConnectivityState *conState, InNetActivityState *actState,
		int gpuIndStart, int numGPUs)
{
	cout << "Constructor entered" << endl;
	
	randGen=new CRandomSFMT0(time(NULL));

	cp=conParams;
	ap=actParams;
	cs=conState;
	as=actState;

	this->gpuIndStart=gpuIndStart;
	this->numGPUs=numGPUs;

	gGOGRT=allocate2DArray<float>(cp->maxnumpGRfromGOtoGR, cp->numGR);
	gMFGRT=allocate2DArray<float>(cp->maxnumpGRfromMFtoGR, cp->numGR);
	//gUBCGRT=allocate2DArray<float>(cp->maxnumpGRfromMFtoGR, cp->numGR);
	pGRDelayfromGRtoGOT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromGRtoGO, cp->numGR);
	pGRfromMFtoGRT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromMFtoGR, cp->numGR);
	pGRfromGOtoGRT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromGOtoGR, cp->numGR);
	//pGRfromUBCtoGRT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromMFtoGR, cp->numGR);
	
	pGRfromGRtoGOT=allocate2DArray<ct_uint32_t>(cp->maxnumpGRfromGRtoGO, cp->numGR);
	
	//pGRDelayfromPFtoBCT=allocate2DArray<ct_uint32_t>(cp->numBC, cp->numGR);	
	//pGRfromPFtoBCT=allocate2DArray<ct_uint32_t>(cp->numBC, cp->numGR);

	apBufGRHistMask=(1<<(ap->tsPerHistBinGR))-1;
	cout<<"apBufGRHistMask "<<hex<<apBufGRHistMask<<dec<<endl;

	sumGRInputGO=new ct_uint32_t[cp->numGO];
	sumInputGOGABASynDepGO=new float[cp->numGO];

	tempGIncGRtoGO=actParams->gIncGRtoGO;
	
	initCUDA();
}

InNet::~InNet()
{
	cout << "**********************************************CUDA DESTRUCTOR ENTERED**********************************************" << endl;
	delete randGen;

	//gpu related host variables
	cudaFreeHost(outputGRH);
	cudaFreeHost(inputSumPFBCH);
	cudaFreeHost(inputSumPFSCH);

	//gpu variables
	cout << numGPUs << endl;
	for(int i=0; i<numGPUs; i++)
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
		//cudaFree(numUBCperGR[i]);
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
		
		//cudaFree(numUBCInPerGRGPU[i]);
		cudaFree(grConMFOutGRGPU[i]);
		//cudaFree(grConUBCOutGRGPU[i]);
		//cudaFree(gUBC_EGRGPU[i]);
		cudaFree(gUBC_EGRSumGPU[i]);
		//cudaFree(gUBC_EDirectGPU[i]);
		//cudaFree(gUBC_ESpilloverGPU[i]);
		
		
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
		cudaFree(dynamicAmpGOGRGPU[i]); /**********/

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

		//UBC
		//cudaFree(apUBCGPU[i]);
		//cudaFreeHost(apUBCH[i]);
		//cudaFreeHost(depAmpUBCH[i]);
		//cudaFree(depAmpUBCGPU[i]);
		cudaFree(apUBCtoGRGPU[i]); /******/
		//cudaFree(depAmpUBCGRGPU[i]); /********/		
		
		
		cudaDeviceSynchronize();
	cout << "***************GPU DELETED************" << endl;
	
	}

	
	delete[] apUBCtoGRGPU;
	//delete[] depAmpUBCGRGPU; 
	
	//mf
	delete[] apMFH;
	delete[] apMFGPU;
	delete[] depAmpMFH;
	delete[] depAmpMFGPU;
	delete[] depAmpMFGRGPU;
	//gr
	delete2DArray<float>(gMFGRT);
//	delete2DArray<float>(gUBCGRT);
	delete2DArray<float>(gGOGRT);
	delete2DArray<ct_uint32_t>(pGRDelayfromGRtoGOT);
	delete2DArray<ct_uint32_t>(pGRfromMFtoGRT);
//	delete2DArray<ct_uint32_t>(pGRfromUBCtoGRT);
	delete2DArray<ct_uint32_t>(pGRfromGOtoGRT);
	delete2DArray<ct_uint32_t>(pGRfromGRtoGOT);
	
	//delete2DArray<ct_uint32_t>(pGRDelayfromPFtoBCT);	
	//delete2DArray<ct_uint32_t>(pGRfromPFtoBCT);

	delete[] gEGRGPU;
	delete[] gEGRGPUP;
	delete[] gEGRSumGPU;
	delete[] gEDirectGPU;
	delete[] gESpilloverGPU;
	delete[] apMFtoGRGPU;
	delete[] numMFperGR;
	//delete[] numUBCperGR;
	
	//delete[] gUBC_EGRGPU;
	//delete[] gUBC_EGRGPUP;
	delete[] gUBC_EGRSumGPU;
	//delete[] gUBC_EDirectGPU;
	//delete[] gUBC_ESpilloverGPU;

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
	//delete[] numUBCInPerGRGPU;
	delete[] grConMFOutGRGPU;
	//delete[] grConUBCOutGRGPU;
	delete[] grConMFOutGRGPUP;
	//delete[] grConUBCOutGRGPUP;


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

	//delete[] apUBCH;
	//delete[] apUBCGPU;
	//delete[] depAmpUBCH;
	//delete[] depAmpUBCGPU;

	delete[] plasScalerEx;
	delete[] plasScalerInh;
	delete2DArray<float>(goExScaler);	
	delete2DArray<float>(goInhScaler);	
	delete2DArray<float>(goFRArray);	

}

void getnumGPUs()
{
	cout << "NUMGPUS = " << endl;
//	cout << numGPUs << endl;
}

void InNet::initCUDA()
{
	cudaError_t error;
	int maxNumGPUs;

	error = cudaGetDeviceCount(&maxNumGPUs);
	cerr<<"CUDA number of devices: "<<maxNumGPUs<<", "<<cudaGetErrorString(error)<<endl;
	cerr<<"number of devices used: "<<numGPUs<<" starting at GPU# "<<gpuIndStart<<endl;

	cout << cp->numGR << endl;
	numGRPerGPU=cp->numGR/numGPUs;
	calcGRActNumGRPerB=512;
	calcGRActNumBlocks=numGRPerGPU/calcGRActNumGRPerB;

	updateGRGOOutNumGRPerR=512*(cp->numGO>512)+cp->numGO*(cp->numGO<=512);
	updateGRGOOutNumGRRows=numGRPerGPU/updateGRGOOutNumGRPerR;

	sumGRGOOutNumGOPerB=1024*(cp->numGO>1024)+cp->numGO*(cp->numGO<=1024);
	sumGRGOOutNumBlocks=cp->numGO/sumGRGOOutNumGOPerB;

	updateMFInGRNumGRPerB=1024*(cp->numMF>1024)+(cp->numMF<=1024)*cp->numMF;
	updateMFInGRNumBlocks=numGRPerGPU/updateMFInGRNumGRPerB;

	updateUBCInGRNumGRPerB=1024*(cp->numUBC>1024)+(cp->numUBC<=1024)*cp->numUBC;
	updateUBCInGRNumBlocks=numGRPerGPU/updateUBCInGRNumGRPerB;

	updateGOInGRNumGRPerB=1024*(cp->numGO>=1024)+(cp->numGO<1024)*cp->numGO;
	updateGOInGRNumBlocks=numGRPerGPU/updateGOInGRNumGRPerB;

	updateGRBCOutNumGRPerR=512*(cp->numBC>512)+cp->numBC*(cp->numBC<=512);
	updateGRBCOutNumGRRows=numGRPerGPU/updateGRBCOutNumGRPerR;
	
	sumGRBCOutNumBCPerB=1024*(cp->numBC>1024)+cp->numBC*(cp->numBC<=1024);
	sumGRBCOutNumBlocks=cp->numBC/sumGRBCOutNumBCPerB;
		
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



//	cudaSetDevice(gpuIndStart);
//	error=cudaMalloc<float>(&testA, 1024*sizeof(float));
//	error=cudaMalloc<float>(&testB, 1024*sizeof(float));
//	error=cudaMalloc<float>(&testC, 1024*sizeof(float));

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

	//apUBCH=new ct_uint32_t*[numGPUs];
	//apUBCGPU=new ct_uint32_t*[numGPUs];
	//depAmpUBCH=new float*[numGPUs];
	//depAmpUBCGPU=new float*[numGPUs];
	

	for(int i=0; i<numGPUs; i++)
	{
			
		error=cudaSetDevice(i+gpuIndStart);
	//	cerr<<"setting device "<<i<<": "<<cudaGetErrorString(error)<<endl;	
	//	error=cudaMallocHost((void **)&apUBCH[i], cp->numUBC*sizeof(ct_uint32_t));
	//	cerr<<"Allocating apUBCH for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
	//	error=cudaMalloc((void **)&apUBCGPU[i], cp->numUBC*sizeof(ct_uint32_t));
	//	cerr<<"Allocating apUBCGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		
	//	error=cudaMallocHost((void **)&depAmpUBCH[i], cp->numUBC*sizeof(float));
	//	cerr<<"Allocating depAmpUBCH for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
	//	error=cudaMalloc<float>(&(depAmpUBCGPU[i]), cp->numUBC*sizeof(float));
	//	cerr<<"Allocating depAmpUBCGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;

		cudaDeviceSynchronize();

	//	cudaMemset(apUBCGPU[i], 0, cp->numUBC*sizeof(ct_uint32_t));
	//	cudaMemset(depAmpUBCGPU[i], 1, cp->numUBC*sizeof(float));

	//	for(int j=0; j<cp->numUBC; j++)
	//	{
	//		apUBCH[i][j]=0;
	//		depAmpUBCH[i][j]=1;
	//	}
	}

}



void InNet::initMFCUDA()
{
	cudaError_t error;

	apMFGPU=new ct_uint32_t*[numGPUs];
	apMFH=new ct_uint32_t*[numGPUs];
	depAmpMFH=new float*[numGPUs];
	depAmpMFGPU=new float*[numGPUs];
//	cudaSetDevice(gpuIndStart);
//	error=cudaHostAlloc((void **)&apMFH, cp->numMF*sizeof(unsigned int), cudaHostAllocPortable);
//	cerr<<"apMFH alloc: "<<cudaGetErrorString(error)<<endl;
//	cudaMallocHost((void **)&apMFH, cp->numMF*sizeof(unsigned int));

	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		cerr<<"setting device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apMFGPU[i], cp->numMF*sizeof(ct_uint32_t));
		cerr<<"Allocating apMFGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMallocHost((void **)&apMFH[i], cp->numMF*sizeof(ct_uint32_t));
		cerr<<"Allocating apMFH for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
			
		error=cudaMallocHost((void **)&depAmpMFH[i], cp->numMF*sizeof(float));
		cerr<<"Allocating depAmpMFH for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc<float>(&(depAmpMFGPU[i]), cp->numMF*sizeof(float));
		cerr<<"Allocating depAmpMFGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;

		cudaDeviceSynchronize();

		cudaMemset(apMFGPU[i], 0, cp->numMF*sizeof(ct_uint32_t));
		cudaMemset(depAmpMFGPU[i], 1, cp->numMF*sizeof(float));

		for(int j=0; j<cp->numMF; j++)
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
	apUBCtoGRGPU=new int*[numGPUs];
	depAmpUBCGRGPU=new float*[numGPUs];
	gUBC_EGRGPU=new float*[numGPUs];
	gUBC_EGRGPUP=new size_t[numGPUs];
	gUBC_EGRSumGPU=new float*[numGPUs];
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


//	error=cudaSetDevice(gpuIndStart);
//	cerr<<"setting device 0: "<<cudaGetErrorString(error)<<endl;
//	error=cudaHostAlloc((void **)&outputGRH, cp->numGR*sizeof(ct_uint8_t), cudaHostAllocPortable);
//	cerr<<"allocating outputGRH: "<<cudaGetErrorString(error)<<endl;
	outputGRH=new ct_uint8_t[cp->numGR];

//	cudaMallocHost((void **)&outputGRH, numGR*sizeof(unsigned char))

	for(int i=0; i<cp->numGR; i++)
	{
		outputGRH[i]=0;
	}

	//allocate memory for GPU
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		cerr<<"setting device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&outputGRGPU[i], numGRPerGPU*sizeof(ct_uint8_t));
		cerr<<"allocating outputGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&apGRGPU[i], numGRPerGPU*sizeof(ct_uint32_t));
		cerr<<"allocating apGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
//		error=cudaMalloc((void **)&vGRGPU[i], numGRPerGPU*sizeof(float));
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
				numGRPerGPU*sizeof(float), cp->maxnumpGRfromMFtoGR);
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
		
		//error=cudaMalloc((void **)&numUBCperGR[i], numGRPerGPU*sizeof(int));
		//cerr<<"allocating numUBCperGR for device "<<i<<": "<<cudaGetErrorString(error)<<endl;	
		error=cudaMalloc((void **)&apUBCtoGRGPU[i], numGRPerGPU*sizeof(int));
		cerr<<"allocating apUBCtoGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMallocPitch((void **)&gUBC_EGRGPU[i], (size_t *)&gUBC_EGRGPUP[i],
				numGRPerGPU*sizeof(float), cp->maxnumpGRfromMFtoGR);
		//cerr<<"gUBC_EGRGPUP: "<<gUBC_EGRGPUP[i]<<endl;
		error=cudaMalloc((void **)&gUBC_EGRSumGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating gUBC_EGRSumGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		//error=cudaMalloc((void **)&gUBC_EDirectGPU[i], numGRPerGPU*sizeof(float));
		//cerr<<"allocating gUBC_EDirectGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		//error=cudaMalloc((void **)&gUBC_ESpilloverGPU[i], numGRPerGPU*sizeof(float));
		//cerr<<"allocating gUBC_ESpilloverGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		
		error=cudaMalloc((void **)&depAmpMFGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating depAmpMFGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&depAmpGOGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating depAmpGOGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaMalloc((void **)&dynamicAmpGOGRGPU[i], numGRPerGPU*sizeof(float));
		cerr<<"allocating dynamicAmpGOGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		//error=cudaMalloc((void **)&depAmpUBCGRGPU[i], numGRPerGPU*sizeof(float));
		//cerr<<"allocating depAmpUBCGRGPU for device "<<i<<": "<<cudaGetErrorString(error)<<endl;
		
		
		
		error=cudaMallocPitch((void **)&gIGRGPU[i], (size_t *)&gIGRGPUP[i],
				numGRPerGPU*sizeof(float), cp->maxnumpGRfromGOtoGR);
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
				numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromGRtoGO);	
		//end conduction delay

		//New Basket Cell stuff
		error=cudaMallocPitch((void **)&grConGROutBCGPU[i], (size_t *)&grConGROutBCGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), cp->numBC);
		error=cudaMallocPitch((void **)&delayBCMasksGRGPU[i], (size_t *)&delayBCMasksGRGPUP[i],
				numGRPerGPU*sizeof(ct_uint32_t), cp->numBC);
		error=cudaMalloc((void **)&numBCOutPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));




		//connectivity
		error=cudaMallocPitch((void **)&grConGROutGOGPU[i], (size_t *)&grConGROutGOGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromGRtoGO);
		error=cudaMalloc((void **)&numGOOutPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));

		error=cudaMallocPitch((void **)&grConGOOutGRGPU[i], (size_t *)&grConGOOutGRGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromGOtoGR);
		error=cudaMalloc((void **)&numGOInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));

		error=cudaMallocPitch((void **)&grConMFOutGRGPU[i], (size_t *)&grConMFOutGRGPUP[i],
					numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromMFtoGR);
		error=cudaMalloc((void **)&numMFInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));
		

		//error=cudaMallocPitch((void **)&grConUBCOutGRGPU[i], (size_t *)&grConUBCOutGRGPUP[i],
		//			numGRPerGPU*sizeof(ct_uint32_t), cp->maxnumpGRfromMFtoGR);	
		//error=cudaMalloc((void **)&numUBCInPerGRGPU[i], numGRPerGPU*sizeof(ct_int32_t));
		
		
		//end connectivity

		error=cudaMalloc((void **)&historyGRGPU[i], numGRPerGPU*sizeof(ct_uint64_t));
		//end GPU memory allocation
		cudaDeviceSynchronize();

		cerr<<"GR GPU memory allocation: "<<cudaGetErrorString(error)<<endl;
	}
	cout << "NUMGPUS" << endl;
	cout << "WTF" << endl;
	cout << numGPUs;
	cout << "Passed" << endl;
	
	error=cudaGetLastError();
	cerr<<"MemAllocDone: "<<cudaGetErrorString(error)<<endl;



//	create a transposed copy of the matrices from activity state and connectivity
	
	for(int i=0; i<cp->maxnumpGRfromGOtoGR; i++)
	{
		cout << "	" << i << endl;
		for(int j=0; j<cp->numGR; j++)
		{
			gGOGRT[i][j]=as->gGOGR[j][i];
			pGRfromGOtoGRT[i][j]=cs->pGRfromGOtoGR[j][i];
		}
	}

	//cout << "here" << endl;

	for(int i=0; i<cp->maxnumpGRfromMFtoGR; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			gMFGRT[i][j]=as->gMFGR[j][i];
			pGRfromMFtoGRT[i][j]=cs->pGRfromMFtoGR[j][i];
		
			//gUBCGRT[i][j]=as->gUBCGR[j][i];
			//pGRfromUBCtoGRT[i][j]=cs->pGRfromUBCtoGR[j][i];
			
		
		}
	}
	

	for(int i=0; i<cp->maxnumpGRfromGRtoGO; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			pGRDelayfromGRtoGOT[i][j]=cs->pGRDelayMaskfromGRtoGO[j][i];
			pGRfromGRtoGOT[i][j]=cs->pGRfromGRtoGO[j][i];
		}
	}

	/*for(int i=0; i<cp->numBC; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			pGRDelayfromPFtoBCT[i][j]=cs->pGRDelayMaskfromGRtoBC[j][i];
			pGRfromPFtoBCT[i][j]=cs->pGRfromPFtoBC[j][i];	
		}
	}*/

	/*for(int i=0; i<20; i++)
	{
		for(int j=0; j<20; j++)
		{	
			cout << pGRfromPFtoBCT[i][j] << " ";
		}
		cout << endl;
	}*/
	//initialize GR GPU variables
	cerr<<"start GPU memory initialization"<<endl;
	
	
	
	for(int i=0; i<numGPUs; i++)
	{
		int cpyStartInd;
		int cpySize;

		cpyStartInd=numGRPerGPU*i;//cp->numGR*i/numGPUs;
		cpySize=numGRPerGPU;
		cudaSetDevice(i+gpuIndStart);

	//	error=cudaMemcpy(vGRGPU[i], &(as->vGR[cpyStartInd]),
	//			cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaMemcpy(gKCaGRGPU[i], &(as->gKCaGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);
	
	//Trying to print out vGRGPU
	/*	float *tempCpy[numGPUs];
		cudaMemcpy(tempCpy[i], vGRGPU[i], cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		cout << "Temp Copy" << endl;
	*/	//cout << *tempCpy[i] << endl;

//		error=cudaMemcpy(outputGRGPU[i], &outputGRH[cpyStartInd],
//				cpySize*sizeof(ct_uint8_t), cudaMemcpyHostToDevice);

		cerr<<"cuda memory copy vGRGPU, outputGRGPU, and gKCAGRGPU: "<<cudaGetErrorString(error)<<endl;

		for(int j=0; j<cp->maxnumpGRfromMFtoGR; j++)
		{
			error=cudaMemcpy((void *)((char *)gEGRGPU[i]+j*gEGRGPUP[i]),
					&gMFGRT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice);	
			error=cudaMemcpy((void *)((char *)grConMFOutGRGPU[i]+j*grConMFOutGRGPUP[i]),
					&pGRfromMFtoGRT[j][cpyStartInd], cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		


			//error=cudaMemcpy((void *)((char *)gUBC_EGRGPU[i]+j*gUBC_EGRGPUP[i]),
			//		&gUBCGRT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice);	
			//error=cudaMemcpy((void *)((char *)grConUBCOutGRGPU[i]+j*grConUBCOutGRGPUP[i]),
			//		&pGRfromUBCtoGRT[j][cpyStartInd], cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		
		
		}
		cerr<<"cuda memory copy gEGRGPU and grConMFOutGRGPU: "<<cudaGetErrorString(error)<<endl;
	
		/************/


		error=cudaMemcpy(vGRGPU[i], &(as->vGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);	
		error=cudaMemcpy(gEGRSumGPU[i], &(as->gMFSumGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);	
		error=cudaMemcpy(gEDirectGPU[i], &(as->gMFDirectGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaMemcpy(gESpilloverGPU[i], &(as->gMFSpilloverGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaMemcpy(apMFtoGRGPU[i], &(as->apMFtoGR[cpyStartInd]), cpySize*sizeof(int), cudaMemcpyHostToDevice);
		error=cudaMemcpy(numMFperGR[i], &(cs->numpGRfromMFtoGR[cpyStartInd]), cpySize*sizeof(int), cudaMemcpyHostToDevice);	
		//error=cudaMemcpy(numUBCperGR[i], &(cs->numpGRfromUBCtoGR[cpyStartInd]), cpySize*sizeof(int), cudaMemcpyHostToDevice);	
		error=cudaMemcpy(apUBCtoGRGPU[i], &(as->apUBCtoGR[cpyStartInd]), cpySize*sizeof(int), cudaMemcpyHostToDevice);	
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
		
		error=cudaMemcpy(gUBC_EGRSumGPU[i], &(as->gUBCSumGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);	
		//error=cudaMemcpy(gUBC_EDirectGPU[i], &(as->gUBCDirectGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		//error=cudaMemcpy(gUBC_ESpilloverGPU[i], &(as->gUBCSpilloverGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
		
		error=cudaMemcpy(depAmpMFGRGPU[i], &(as->depAmpMFtoGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
		//error=cudaMemcpy(depAmpUBCGRGPU[i], &(as->depAmpUBCtoGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpy(depAmpGOGRGPU[i], &(as->depAmpGOtoGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpy(dynamicAmpGOGRGPU[i], &(as->dynamicAmpGOtoGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		
		error=cudaGetLastError();
		cerr<<"		CUDA check: "<<cudaGetErrorString(error)<<endl;
			
		/********/

		for(int j=0; j<cp->maxnumpGRfromGOtoGR; j++)
		{
			error=cudaMemcpy((void *)((char *)gIGRGPU[i]+j*gIGRGPUP[i]),
					&gGOGRT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice);
			error=cudaMemcpy((void *)((char *)grConGOOutGRGPU[i]+j*grConGOOutGRGPUP[i]),
					&pGRfromGOtoGRT[j][cpyStartInd], cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		}

		cout << "check" << endl;	

		error=cudaMemcpy(gIGRSumGPU[i], &(as->gGOSumGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaMemcpy(gIDirectGPU[i], &(as->gGODirectGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaMemcpy(gISpilloverGPU[i], &(as->gGOSpilloverGR[cpyStartInd]), cpySize*sizeof(float), cudaMemcpyHostToDevice);

		error=cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]),
				cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);

		error=cudaMemcpy(threshGRGPU[i], &(as->threshGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);
		cout << "check" << endl;	
	
	
		error=cudaMemcpy(gLeakGRGPU[i], &(as->gLeakGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaMemcpy(gNMDAGRGPU[i], &(as->gNMDAGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);
		error=cudaMemcpy(gNMDAIncGRGPU[i], &(as->gNMDAIncGR[cpyStartInd]),
				cpySize*sizeof(float), cudaMemcpyHostToDevice);
		cout << "check" << endl;	
		

		for(int j=0; j<cp->maxnumpGRfromGRtoGO; j++)
		{
			error=cudaMemcpy((void *)((char *)delayGOMasksGRGPU[i]+j*delayGOMasksGRGPUP[i]),
					&pGRDelayfromGRtoGOT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice );
			error=cudaMemcpy((void *)((char *)grConGROutGOGPU[i]+j*grConGROutGOGPUP[i]),
					&pGRfromGRtoGOT[j][cpyStartInd], cpySize*sizeof(unsigned int), cudaMemcpyHostToDevice);
		}
		error=cudaGetLastError();
		cerr<<"CUDA check: "<<cudaGetErrorString(error)<<endl;

		//Basket cell stuff
		/*for(int j=0; j<cp->numBC; j++)
		{
			error=cudaMemcpy((void *)((char *)delayBCMasksGRGPU[i]+j*delayBCMasksGRGPUP[i]),
					&pGRDelayfromPFtoBCT[j][cpyStartInd], cpySize*sizeof(float), cudaMemcpyHostToDevice );
			error=cudaMemcpy((void *)((char *)grConGROutBCGPU[i]+j*grConGROutBCGPUP[i]),
					&pGRfromPFtoBCT[j][cpyStartInd], cpySize*sizeof(unsigned int), cudaMemcpyHostToDevice);
		}*/
		cout << "	check" << endl;	
		//error=cudaMemcpy(numBCOutPerGRGPU[i], &(cs->numpGRfromPFtoBC[cpyStartInd]),
		//		cpySize*sizeof(ct_int32_t), cudaMemcpyHostToDevice);
		cout << "	check" << endl;	
		
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

		//error=cudaMemcpy(numUBCInPerGRGPU[i], &(cs->numpGRfromUBCtoGR[cpyStartInd]),
		//		cpySize*sizeof(int), cudaMemcpyHostToDevice);



		cudaMemset(outputGRGPU[i], 0, cpySize*sizeof(ct_uint8_t));
		cudaMemset(apGRGPU[i], 0, cpySize*sizeof(ct_uint32_t));
	

		cudaDeviceSynchronize();
	
	}
	//end copying to GPU
	cerr<<"numGRPerGPU "<<numGRPerGPU<<endl;

}

void InNet::grStim(int startGRStim, int numGRStim)
{
	exportAPGR(); //getGRGPUData<ct_uint8_t>(outputGRGPU, as->apGR);
	exportAPBufGR();  //getGRGPUData<ct_uint32_t>(apBufGRGPU, as->apBufGR);
	for (int j=startGRStim; j<=startGRStim+numGRStim; j++)
	{
		as->apBufGR[j] = as->apBufGR[j]|1u; 
		outputGRH[j] = true;
	}
	
	for(int i=0; i<numGPUs; i++)
	{
		int cpyStartInd;
		int cpySize;

		cpyStartInd=numGRPerGPU*i;//cp->numGR*i/numGPUs;
		cpySize=numGRPerGPU;
		cudaSetDevice(i+gpuIndStart);

		cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]),
				cpySize*sizeof(ct_uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(outputGRGPU[i], &outputGRH[cpyStartInd],
				cpySize*sizeof(ct_uint8_t), cudaMemcpyHostToDevice);
	}
}

void InNet::initGOCUDA()
{
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
	goExScaler = allocate2DArray<float>(cp->numGO, 1000);	
	arrayInitialize<float>(goExScaler[0], 0, cp->numGO*1000);
	goInhScaler = allocate2DArray<float>(cp->numGO, 1000);	
	arrayInitialize<float>(goInhScaler[0], 0, cp->numGO*1000);
	
	//goGRGOScaler=allocate2DArray<float>(cp->numGO, 40*10000);
	//arrayInitialize<float>(goGRGOScaler[0], 0, cp->numGO*40*10000);

	
	goFRArray= allocate2DArray<float>(cp->numGO, 1000);	
	arrayInitialize<float>(goFRArray[0], 0, cp->numGO*1000);
		
	counterGOweight = 0;
		

	counter=new int[cp->numGO];
	for(int i=0; i<cp->numGO; i++)
	{
		counter[i] = 0;
	}

//	cudaSetDevice(gpuIndStart);
//	cudaHostAlloc((void **)&apGOH, cp->numGO*sizeof(unsigned int), cudaHostAllocPortable);
//	cudaMallocHost((void **)&apGOH, numGO*sizeof(unsigned int));

	//initialize host memory
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMallocHost((void **)&grInputGOSumH[i], cp->numGO*sizeof(ct_uint32_t));
		cudaMallocHost((void **)&apGOH[i], cp->numGO*sizeof(ct_uint32_t));
		cudaMallocHost((void **)&depAmpGOH[i], cp->numGO*sizeof(float));
		cudaMallocHost((void **)&dynamicAmpGOH[i], cp->numGO*sizeof(float));
	
		for(int j=0; j<cp->numGO; j++)
		{
			grInputGOSumH[i][j]=0;
			apGOH[i][j]=0;
			depAmpGOH[i][j]=1;
			dynamicAmpGOH[i][j]=1;
			
		}
		//allocate gpu memory
		cudaMalloc((void **)&apGOGPU[i], cp->numGO*sizeof(ct_uint32_t));
		cudaMalloc((void **)&depAmpGOGPU[i], cp->numGO*sizeof(float));
		cudaMalloc((void **)&dynamicAmpGOGPU[i], cp->numGO*sizeof(float));

		cudaMallocPitch((void **)&grInputGOGPU[i], (size_t *)&grInputGOGPUP[i],
				cp->numGO*sizeof(ct_uint32_t), updateGRGOOutNumGRRows);
		cudaMalloc((void **)&grInputGOSumGPU[i], cp->numGO*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();

		for(int j=0; j<updateGRGOOutNumGRRows; j++)
		{
			cudaMemset(((char *)grInputGOGPU[i]+j*grInputGOGPUP[i]),
					0, cp->numGO*sizeof(ct_uint32_t));
		}

		cudaMemset(apGOGPU[i], 0, cp->numGO*sizeof(ct_uint32_t));
		cudaMemset(depAmpGOGPU[i], 1, cp->numGO*sizeof(float));
		cudaMemset(dynamicAmpGOGPU[i], 1, cp->numGO*sizeof(float));
		cudaMemset(grInputGOSumGPU[i], 0, cp->numGO*sizeof(ct_uint32_t));
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
	cudaHostAlloc((void **)&inputSumPFBCH, cp->numBC*sizeof(ct_uint32_t), cudaHostAllocPortable);

	cudaDeviceSynchronize();
//	cudaMallocHost((void **)&inputSumPFBCH, numBC*sizeof(unsigned int));

	//initialize host variables
	for(int i=0; i<cp->numBC; i++)
	{
		inputSumPFBCH[i]=0;
	}
	cout << "before GPU loop" << endl;
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMallocHost((void **)&grInputBCSumH[i], cp->numBC*sizeof(ct_uint32_t));
		for(int j=0; j<cp->numBC; j++)
		{
			grInputBCSumH[i][j]=0;
		}
	
		cudaMallocPitch((void **)&grInputBCGPU[i], (size_t *)&grInputBCGPUP[i],
				cp->numBC*sizeof(ct_uint32_t), updateGRBCOutNumGRRows);
		cudaMalloc((void **)&grInputBCSumGPU[i], cp->numBC*sizeof(ct_uint32_t));		
		//allocate GPU memory
		cudaMallocPitch((void **)&inputPFBCGPU[i], (size_t *)&inputPFBCGPUP[i],
				cp->numpBCfromGRtoBC*sizeof(ct_uint32_t), cp->numBC/numGPUs);
		
		cudaMalloc((void **)&inputSumPFBCGPU[i], cp->numBC/numGPUs*sizeof(ct_uint32_t));
		//end GPU allocation
		
		
		cudaDeviceSynchronize();
		
		
		for(int j=0; j<updateGRBCOutNumGRRows; j++)
		{
			cudaMemset(((char *)grInputBCGPU[i]+j*grInputBCGPUP[i]),
					0, cp->numBC*sizeof(ct_uint32_t));
		}
		
		cudaMemset(grInputBCSumGPU[i], 0, cp->numBC*sizeof(ct_uint32_t));
		
		for(int j=0; j<cp->numBC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFBCGPU[i]+j*inputPFBCGPUP[i]), 0,
					cp->numpBCfromGRtoBC*sizeof(ct_uint32_t));
		}

		cudaMemset(inputSumPFBCGPU[i], 0, cp->numBC/numGPUs*sizeof(ct_uint32_t));		
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
	cudaHostAlloc((void **)&inputSumPFSCH, cp->numSC*sizeof(ct_uint32_t), cudaHostAllocPortable);

	cudaDeviceSynchronize();
//	cudaMallocHost((void **)&inputSumPFSCH, numSC*sizeof(ct_uint32_t));
	//initialize host variables
	for(int i=0; i<cp->numSC; i++)
	{
		inputSumPFSCH[i]=0;
	}

	for(int i=0; i<numGPUs; i++)
	{
		//allocate GPU memory
		cudaSetDevice(i+gpuIndStart);
		cudaMallocPitch((void **)&inputPFSCGPU[i], (size_t *)&inputPFSCGPUP[i],
				cp->numpSCfromGRtoSC*sizeof(ct_uint32_t), cp->numSC/numGPUs);

		cudaMalloc((void **)&inputSumPFSCGPU[i], cp->numSC/numGPUs*sizeof(ct_uint32_t));
		//end GPU allocation

		for(int j=0; j<cp->numSC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFSCGPU[i]+j*inputPFSCGPUP[i]), 0,
					cp->numpSCfromGRtoSC*sizeof(ct_uint32_t));
		}

		cudaMemset(inputSumPFSCGPU[i], 0, cp->numSC/numGPUs*sizeof(ct_uint32_t));

		cudaDeviceSynchronize();
	}
}

void InNet::writeToState()
{
	cudaError_t error;
	//GR variables
	getGRGPUData<ct_uint8_t>(outputGRGPU, as->apGR);
	getGRGPUData<ct_uint32_t>(apBufGRGPU, as->apBufGR);
	getGRGPUData<float>(gEGRSumGPU, as->gMFSumGR);
	getGRGPUData<float>(gIGRSumGPU, as->gGOSumGR);

	getGRGPUData<float>(threshGRGPU, as->threshGR);
	getGRGPUData<float>(vGRGPU, as->vGR);
	getGRGPUData<float>(gKCaGRGPU, as->gKCaGR);
	getGRGPUData<ct_uint64_t>(historyGRGPU, as->historyGR);
	for(int i=0; i<numGPUs; i++)
	{
		int cpyStartInd;
		int cpySize;

		cpyStartInd=numGRPerGPU*i;
		cpySize=numGRPerGPU;

		cudaSetDevice(i+gpuIndStart);



		for(int j=0; j<cp->maxnumpGRfromMFtoGR; j++)
		{
			error=cudaMemcpy(&gMFGRT[j][cpyStartInd], (void *)((char *)gEGRGPU[i]+j*gEGRGPUP[i]),
					cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		}

		for(int j=0; j<cp->maxnumpGRfromGOtoGR; j++)
		{
			error=cudaMemcpy(&gGOGRT[j][cpyStartInd], (void *)((char *)gIGRGPU[i]+j*gIGRGPUP[i]),
					cpySize*sizeof(float), cudaMemcpyDeviceToHost);
		}
	}

	for(int i=0; i<cp->maxnumpGRfromMFtoGR; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			as->gMFGR[j][i]=gMFGRT[i][j];
		}
	}

	for(int i=0; i<cp->maxnumpGRfromGOtoGR; i++)
	{
		for(int j=0; j<cp->numGR; j++)
		{
			as->gGOGR[j][i]=gGOGRT[i][j];
		}
	}

	//stellate variables
	//for(int i=0; i<cp->numSC; i++)
	//{
	//	as->inputSumPFSC[i]=inputSumPFSCH[i];
	//}
}

void InNet::setGIncGRtoGO(float inc)
{
	tempGIncGRtoGO=inc;
}

void InNet::resetGIncGRtoGO()
{
	tempGIncGRtoGO=ap->gIncGRtoGO;
}

const ct_uint8_t* InNet::exportAPMF()
{
	return (const ct_uint8_t *)apMFOut;
}
const ct_uint8_t* InNet::exportAPSC()
{
	return (const ct_uint8_t *)as->apSC;
}

const ct_uint8_t* InNet::exportAPGO()
{
	return (const ct_uint8_t *)as->apGO;
}
/*const int* InNet::exportSPKGO()
{
	return (const int *)as->spkGO;
}*/

const ct_uint8_t* InNet::exportAPUBC()
{
	return (const ct_uint8_t *)as->apUBC;
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

template<typename Type>cudaError_t InNet::getGRGPUData(Type **gpuData, Type *hostData)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaMemcpy((void *)&hostData[i*numGRPerGPU], gpuData[i],
				numGRPerGPU*sizeof(Type), cudaMemcpyDeviceToHost);
	}
	return cudaGetLastError();
}

const float* InNet::exportVmGR()
{
	getGRGPUData<float>(vGRGPU, as->vGR);
	return (const float *)as->vGR;
}

const float* InNet::exportVmGO()
{
	return (const float *)as->vGO;
}


const float* InNet::exportExGOInput()
{
	return (const float *)as->exGOInput;
}
const float* InNet::exportInhGOInput()
{
	return (const float *)as->inhGOInput;
}






const float* InNet::exportVGOGOcouple()
{
	return (const float *)as->vCoupleGO;
}
const float* InNet::exportgSum_MFGO()
{
	return (const float *)as->gSum_MFGO;
}
const float* InNet::exportgSum_GRGO()
{
	return (const float *)as->gGRGO;
}
const float* InNet::exportgSum_GOGO()
{
	return (const float *)as->gSum_GOGO;
}

const float* InNet::exportvSum_GOGO()
{
	return (const float *)as->vSum_GOGO;
}
const float* InNet::exportvSum_GRGO()
{
	return (const float *)as->vSum_GRGO;
}
const float* InNet::exportvSum_MFGO()
{
	return (const float *)as->vSum_MFGO;
}




const float* InNet::exportVmSC()
{
	return (const float *)as->vSC;
}

const float* InNet::exportGESumGR()
{
	getGRGPUData<float>(gEGRSumGPU, as->gMFSumGR);
	return (const float *)as->gMFSumGR;
}
/*
const float* InNet::exportDepSumMFGR()
{
	getGRGPUData<float>(depAmpMFGRGPU, as->depAmpMFtoGR);
	return (const float *)as->depAmpMFtoGR;
}
*/



const float* InNet::exportGUBCESumGR()
{
	getGRGPUData<float>(gUBC_EGRSumGPU, as->gUBCSumGR);
	return (const float *)as->gUBCSumGR;
}

const float* InNet::exportDepSumUBCGR()
{
	getGRGPUData<float>(depAmpUBCGRGPU, as->depAmpUBCtoGR);
	return (const float *)as->depAmpUBCtoGR;
}





const float* InNet::exportDepSumGOGR()
{
	getGRGPUData<float>(depAmpGOGRGPU, as->depAmpGOtoGR);
	return (const float *)as->depAmpGOtoGR;
}


const float* InNet::exportDynamicSpillSumGOGR()
{
	getGRGPUData<float>(dynamicAmpGOGRGPU, as->dynamicAmpGOtoGR);
	return (const float *)as->dynamicAmpGOtoGR;
}




const float* InNet::exportgNMDAGR()
{
	getGRGPUData<float>(gNMDAGRGPU, as->gNMDAGR);
	return (const float *)as->gNMDAGR;
}
const int* InNet::exportAPfromMFtoGR()
{
	getGRGPUData<int>(apMFtoGRGPU, as->apMFtoGR);
	return (const int *)as->apMFtoGR;
}



const float* InNet::exportGISumGR()
{
	getGRGPUData<float>(gIGRSumGPU, as->gGOSumGR);
	return (const float *)as->gGOSumGR;
}

const ct_uint32_t* InNet::exportSumGRInputGO()
{
	return (const ct_uint32_t *)sumGRInputGO;
}

const float* InNet::exportgGOGO()
{
	return (const float *)as->gGOGO;
}


const float* InNet::exportSumGOInputGO()
{
	return (const float *)sumInputGOGABASynDepGO;
}

const float* InNet::exportGOOutSynScaleGOGO()
{
	return (const float *)as->goGABAOutSynScaleGOGO;
}

const ct_uint8_t* InNet::exportHistMF()
{
	return (const ct_uint8_t *)as->histMF;
}
const ct_uint32_t* InNet::exportAPBufMF()
{
	return (const ct_uint32_t *)as->apBufMF;
}
const ct_uint32_t* InNet::exportAPBufGO()
{
	return (const ct_uint32_t *)as->apBufGO;
}
const ct_uint32_t* InNet::exportAPBufGR()
{
	getGRGPUData<ct_uint32_t>(apBufGRGPU, as->apBufGR);
	return (const ct_uint32_t *)as->apBufGR;
}
const unsigned int* InNet::exportAPBufSC()
{
	return (const unsigned int *)as->apBufSC;
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
		for(int i=0; i<cp->numMF; i++)
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

as->goTimeStep++;


float gConstGO = ap->gConstGO;

//50ms
float gLeakGO = 0.02;
float mGluDecay = exp(-1.0/100);
float gNMDAIncGRGO;


	for(int i=0; i<cp->numGO; i++){
		sumGRInputGO[i] = 0;
		for(int j=0; j<numGPUs; j++){
			sumGRInputGO[i]+=grInputGOSumH[j][i];
		}		
	}



#pragma omp parallel for
		for(int i=0; i<cp->numGO; i++)
		{

			//totalGRIn=0;
			//for(int j=0; j<numGPUs; j++)
			//{
			//	totalGRIn+=grInputGOSumH[j][i];
			//}
			//if(i==0){ cout << totalGRIn << endl; }
			//sumGRInputGO[i]=totalGRIn;
			//sumInputGOGABASynDepGO[i]=as->inputGOGABASynDepGO[i];

			
			
			//NMDA Low
			gNMDAIncGRGO=(0.00000082263*as->vGO[i]*as->vGO[i]*as->vGO[i])+(0.00021653*as->vGO[i]*as->vGO[i])+(0.0195*as->vGO[i])+0.6117; 
			//NMDA High
			as->gNMDAIncMFGO[i]=(0.00000011969*as->vGO[i]*as->vGO[i]*as->vGO[i])+(0.000089369*as->vGO[i]*as->vGO[i])+(0.0151*as->vGO[i])+0.7713; 
		
			as->gSum_MFGO[i] = (as->inputMFGO[i]/*ap->gIncMFtoGO*/*MFGO) + as->gSum_MFGO[i]*ap->gDecMFtoGO;
			as->gSum_GOGO[i] = 0;//(as->inputGOGO[i]*gogoW/*ap->gGABAIncGOtoGO*/*as->synWscalerGOtoGO[i]) + as->gSum_GOGO[i]*ap->gGABADecGOtoGO;

			as->gNMDAMFGO[i]=as->inputMFGO[i]*(/*ap->gIncMFtoGO*/MFGO*ap->NMDA_AMPAratioMFGO*as->gNMDAIncMFGO[i])+as->gNMDAMFGO[i]*ap->gDecayMFtoGONMDA;	
			as->gNMDAUBCGO[i]=as->inputUBCGO[i]*(ap->gIncUBCtoGO*1.0*as->gNMDAIncMFGO[i])+as->gNMDAMFGO[i]*ap->gDecayMFtoGONMDA;	
			
			as->gGRGO[i]=(sumGRInputGO[i]*GRGO/*tempGIncGRtoGO*/)*as->synWscalerGRtoGO[i]+as->gGRGO[i]*ap->gDecGRtoGO;
			//as->gMGluRGO[i]=sumGRInputGO[i]*(/*tempGIncGRtoGO*/GRGO*0.01) + as->gMGluRGO[i]*mGluDecay;
			as->gGRGO_NMDA[i]=sumGRInputGO[i]*(/*tempGIncGRtoGO*/(GRGO*as->synWscalerGRtoGO[i])*0.6*gNMDAIncGRGO) + as->gGRGO_NMDA[i]*ap->gDecayMFtoGONMDA;
			
			//as->gGRGO[i]=(totalGRIn*GRGO/*tempGIncGRtoGO*/)*as->synWscalerGRtoGO[i]+as->gGRGO[i]*ap->gDecGRtoGO;
			//as->gMGluRGO[i]=totalGRIn*(/*tempGIncGRtoGO*/GRGO*0.01) + as->gMGluRGO[i]*mGluDecay;
			//as->gGRGO_NMDA[i]=totalGRIn*(/*tempGIncGRtoGO*/GRGO*0.6*gNMDAIncGRGO) + as->gGRGO_NMDA[i]*ap->gDecayMFtoGONMDA;
			
			as->threshCurGO[i]=as->threshCurGO[i]+(ap->threshRestGO-as->threshCurGO[i])*ap->threshDecGO;
			
		//	as->vSum_GOGO[i] = as->gSum_GOGO[i]*(ap->eGABAGO-as->vGO[i]);	
		//	as->vSum_GRGO[i] = (as->gGRGO[i]+as->gGRGO_NMDA[i])*as->vGO[i];	
		//	as->vSum_MFGO[i] = (as->gSum_MFGO[i]+as->gNMDAMFGO[i])*as->vGO[i];	
			
			as->vGO[i]=as->vGO[i]+(gLeakGO*(ap->eLeakGO-as->vGO[i]))
					+(as->gSum_GOGO[i]*(ap->eGABAGO-as->vGO[i]))
					-(as->gSum_MFGO[i]+as->gGRGO[i]+as->gNMDAUBCGO[i]+as->gNMDAMFGO[i]+as->gSum_UBCtoGO[i]+as->gGRGO_NMDA[i])*as->vGO[i]
					- (as->vCoupleGO[i]*as->vGO[i]);
			
			if(as->vGO[i] > (ap->threshMaxGO)){
				as->vGO[i] = (ap->threshMaxGO);
			}
			
			
			
			as->apGO[i]=as->vGO[i]>as->threshCurGO[i];
			//for(int j=0; j<numGPUs; j++)
			//{
			//	apGOH[j][i]=as->apGO[i];
			//}
			as->apBufGO[i]=(as->apBufGO[i]<<1)|(as->apGO[i]*0x00000001);

			as->threshCurGO[i]=as->apGO[i]*ap->threshMaxGO+(!as->apGO[i])*as->threshCurGO[i];

			as->inputMFGO[i]=0;
			as->inputUBCGO[i]=0;	
			as->inputGOGO[i]=0;

			//as->inputGOGABASynDepGO[i]=0;
		
			//if(as->goTimeStep < 60*5000 && as->synWscalerGRtoGO[i] > 0.2 && as->synWscalerGRtoGO[i] < 1.6){
			//	as->synWscalerGRtoGO[i] = as->synWscalerGRtoGO[i] - (as->apGO[i]*exDecrease) + ((1-as->apGO[i])*exIncrease);	
			//}
			//if(as->synWscalerGOtoGO[i] > 0.98 && as->goTimeStep <= 200*10000 && as->synWscalerGOtoGO[i] < 1.02){
			//	as->synWscalerGOtoGO[i] = as->synWscalerGOtoGO[i] + (as->apGO[i]*inhIncrease) - ((1-as->apGO[i])*inhDecrease);
			//}
			//goGRGOScaler[i][as->goTimeStep] = as->synWscalerGRtoGO[i];
		
		}

	for(int i=0; i<cp->numGO; i++){
		for(int j=0; j<numGPUs; j++){
			apGOH[j][i]=as->apGO[i];		
		}
	}
	//cout << as->vGO[0] << "		" << sumGRInputGO[0]<< endl;
	
	/*if(as->goTimeStep == 60*5000){
		
		ofstream fileGRW;
		fileGRW.open("GRGOWeights_recip100_10Con_GJ0_conv1000_"+to_string(simNum)+".txt");
		for(int i=0; i<cp->numGO; i++){
			fileGRW << as->synWscalerGRtoGO[i] << " ";
		}
	}*/



}


void InNet::calcSCActivities()
{
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numSC; i++)
		{
			as->gPFSC[i]=as->gPFSC[i]+(inputSumPFSCH[i]*ap->gIncGRtoSC);
			as->gPFSC[i]=as->gPFSC[i]*ap->gDecGRtoSC;

			as->vSC[i]=as->vSC[i]+(ap->gLeakSC*(ap->eLeakSC-as->vSC[i]))-as->gPFSC[i]*as->vSC[i];

			as->apSC[i]=as->vSC[i]>as->threshSC[i];
			as->apBufSC[i]=(as->apBufSC[i]<<1)|(as->apSC[i]*0x00000001);

			as->threshSC[i]=as->threshSC[i]+ap->threshDecSC*(ap->threshRestSC-as->threshSC[i]);
			as->threshSC[i]=as->apSC[i]*ap->threshMaxSC+(!as->apSC[i])*(as->threshSC[i]);
		}
	}
}



void InNet::updateMFtoUBCOut()
{
/*
int mfInputInd;
int fakeV;
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numUBC; i++)
		{			

			
			for(int j=0; j<cs->numpUBCfromMFtoUBC[i]; j++)
			{
				
				mfInputInd = cs->pUBCfromMFtoUBC[i];
				as->inputMFUBC[i] = apMFH[0][ mfInputInd ];

				as->gRise_MFtoUBC[i] = ( as->inputMFUBC[i]*(as->gRise_MFtoUBC[i]*0.9923+ap->gIncMFtoUBC) ) + (1-as->inputMFUBC[i])*(as->gRise_MFtoUBC[i]*0.9923); 
				as->gDecay_MFtoUBC[i] = ( as->inputMFUBC[i]*(as->gDecay_MFtoUBC[i]*0.9941+ap->gIncMFtoUBC ) ) + (1-as->inputMFUBC[i])*(as->gDecay_MFtoUBC[i]*0.9941); 
				as->gSum_MFtoUBC[i] =  as->gDecay_MFtoUBC[i] - as->gRise_MFtoUBC[i];
			
			}
		}
	}
*/
}


void InNet::updateGOtoUBCOut()
{
/*
float gIncGOtoUBC = 1;
int goInputInd;
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numUBC; i++)
		{			

			
			for(int j=0; j<cs->numpUBCfromGOtoUBC[i]; j++)
			{
				
				goInputInd = cs->pUBCfromGOtoUBC[i][j];
				as->inputGOUBC[i] = as->apGO[ goInputInd ];
			
				as->gSum_GOtoUBC[i] = as->inputGOUBC[i]*ap->gIncGOtoUBC + as->gSum_GOtoUBC[i]*0.91;//ap->gGABADecGOtoGO;		
			}
		}	
	}
*/
}


void InNet::calcUBCActivities()
{
/*
float gLeakUBC = 0.1;
float eLeakUBC= -70;
float eGOUBC= -70;
float NMDAampScaler = 0.5;

float gKInc = ap->gKIncUBC;
float gKDecay = exp(-1.0/ap->gKTauUBC);

float gConst = ap->gConstUBC;

float threshDecUBC = 1-exp(-1/ap->threshTauUBC);//ap->threshDecGO;

float threshRestUBC = -40;
float threshMaxUBC = 0;
float tempNMDAInc;
float tempV;
#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numUBC; i++)
		{

			
			tempV = as->vUBC[i];
			//tempNMDAInc = (0.00000011969*tempV*tempV*tempV) + (0.000089369*tempV*tempV) + (0.0151*tempV) + 0.7713; 
			//as->gRise_UBCNMDA[i] = (  (as->inputUBCtoUBC[i]+as->inputMFUBC[i])*(as->gRise_UBCNMDA[i]*0.9923+(ap->gIncMFtoUBC*NMDAampScaler) ) ) + (1 - (as->inputUBCtoUBC[i]+as->inputMFUBC[i]))*(as->gRise_UBCNMDA[i]*0.9923); 
			//as->gDecay_UBCNMDA[i] = ( (as->inputUBCtoUBC[i]+as->inputMFUBC[i])*(as->gDecay_UBCNMDA[i]*0.9941+(ap->gIncMFtoUBC*NMDAampScaler) ) ) + (1 -  (as->inputUBCtoUBC[i]+as->inputMFUBC[i])  )*(as->gDecay_UBCNMDA[i]*0.9941); 
			//as->gSum_UBCNMDA[i] =  as->gDecay_UBCNMDA[i] - as->gRise_UBCNMDA[i];		
			//as->gK_UBC[i] = as->apUBC[i]*(as->gK_UBC[i]*gKDecay+gKInc) + (1-as->apUBC[i])*(as->gK_UBC[i]*gKDecay);


			as->vUBC[i]=as->vUBC[i]+gLeakUBC*(eLeakUBC-as->vUBC[i])  + as->gK_UBC[i]*(eLeakUBC-as->vUBC[i]) - gConst*as->vUBC[i] - as->gSum_MFtoUBC[i]*as->vUBC[i] - as->gSumInUBCtoUBC[i]*as->vUBC[i] +as->gSum_GOtoUBC[i]*(eLeakUBC-as->vUBC[i]);
			//- as->gSum_UBCNMDA[i]*as->vUBC[i];
			//as->vUBC[i] = -60;

			as->apUBC[i]=as->vUBC[i]>as->threshUBC[i];
			as->threshUBC[i]=as->threshUBC[i]+threshDecUBC*(threshRestUBC-as->threshUBC[i]);
			as->threshUBC[i]=as->apUBC[i]*threshMaxUBC+(!as->apUBC[i])*(as->threshUBC[i]);
		
			for(int j=0; j<numGPUs; j++)
			{
				apUBCH[j][i] = as->apUBC[i];
			}
		
		}
	}
*/
}



void InNet::updateUBCtoUBCOut()
{
/*
float gIncUBCtoUBC = 1;
int ubcInputInd;

float gDecUBCtoGO = ap->gDecMFtoGO;
float fracDepUBC = 0.1;
float recoveryRate = 1.0/50.0;//ap->recoveryTauMF;

#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numUBC; i++)
		{						
			
			as->gRise_UBCtoUBC[i] = ( as->apUBC[i]*(as->gRise_UBCtoUBC[i]*0.9923+ap->gIncUBCtoUBC)) + (1-as->apUBC[i])*(as->gRise_UBCtoUBC[i]*0.9923); 
			as->gDecay_UBCtoUBC[i] = ( as->apUBC[i]*(as->gDecay_UBCtoUBC[i]*0.9941+ap->gIncUBCtoUBC)) + (1-as->apUBC[i])*(as->gDecay_UBCtoUBC[i]*0.9941); 
				
			as->gSumOutUBCtoUBC[i] =  as->gDecay_UBCtoUBC[i] - as->gRise_UBCtoUBC[i];		

		}
	
	
		for(int i=0; i<cp->numUBC; i++)
		{
			if(cs->numpUBCfromUBCInUBC[i] == 1)
			{
				as->gSumInUBCtoUBC[i] = as->gSumOutUBCtoUBC[ cs->pUBCfromUBCInUBC[i][0]  ]; 
				
				if(as->apUBC[cs->pUBCfromUBCInUBC[i][0]] == 1)
				{
					as->inputUBCtoUBC[i] = 1;
				}
			
			}		
		}

		
	
	}
*/
}




void InNet::updateUBCtoGOOut()
{
/*
//float gIncUBCtoGO = 0.01;
float gDecUBCtoGO = ap->gDecMFtoGO;
float fracDepUBC = ap->fracDepMF;
float recoveryRate = 1/ap->recoveryTauMF;


#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numUBC; i++)
		{
			as->gi_UBCtoGO[i] = as->apUBC[i]*ap->gIncUBCtoGO*as->depAmpUBCGO[i] + as->gi_UBCtoGO[i]*gDecUBCtoGO; 
			as->depAmpUBCGO[i] = as->apUBC[i]*as->depAmpUBCGO[i]*fracDepUBC + (!as->apUBC[i])*( as->depAmpUBCGO[i]+recoveryRate*(1-as->depAmpUBCGO[i]) );
			
			if(as->apUBC[i])
			{
					for(int j=0; j<cs->numpUBCfromUBCtoGO[i]; j++)
					{
						as->inputUBCGO[ cs->pUBCfromUBCtoGO[i][j] ]++;	
					}
			}
		
			
		}

#pragma omp critical 
		{
			for(int i=0; i<cp->numGO; i++)
			{
				as->gSum_UBCtoGO[i] = 0;
				
				for(int j=0; j<cs->numpGOfromUBCtoGO[i]; j++)
				{
					as->gSum_UBCtoGO[i] += as->gi_UBCtoGO[ cs->pGOfromUBCtoGO[i][j] ];
				}
			}
		}
	}
*/
}







void InNet::updateUBCtoGROut()
{
/*
float recoveryRate = 1/ap->recoveryTauMF;
float fracDepUBC = ap->fracDepMF;


#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numUBC; i++)
		{			
			as->depAmpUBCGR[i] = as->apUBC[i]*as->depAmpUBCGR[i]*fracDepUBC + (!as->apUBC[i])*( as->depAmpUBCGR[i]+recoveryRate*(1-as->depAmpUBCGR[i]) ); 
			
#pragma omp critical	
			{	
				for(int j=0; j<numGPUs; j++)
				{
					depAmpUBCH[j][i] = as->depAmpUBCGR[i];
				}
			}	
		}	
	}
*/
}







void InNet::updateMFtoGROut()
{

float recoveryRate = 1/ap->recoveryTauMF;

#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numMF; i++)
		{			
			as->depAmpMFGR[i] = apMFH[0][i]*as->depAmpMFGR[i]*ap->fracDepMF + (!apMFH[0][i])*( as->depAmpMFGR[i]+recoveryRate*(1-as->depAmpMFGR[i]) ); 
			
#pragma omp critical	
			{	
				for(int j=0; j<numGPUs; j++)
				{
					depAmpMFH[j][i] = as->depAmpMFGR[i];
				}
			}	
		}	
	}
}












void InNet::updateMFtoGOOut()
{

float recoveryRate = 1/ap->recoveryTauMF;

#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numMF; i++)
		{			
			as->gi_MFtoGO[i] = apMFH[0][i]*ap->gIncMFtoGO*as->depAmpMFGO[i] + as->gi_MFtoGO[i]*ap->gDecMFtoGO; 
			as->depAmpMFGO[i] = apMFH[0][i]*as->depAmpMFGO[i]*ap->fracDepMF + (!apMFH[0][i])*( as->depAmpMFGO[i]+recoveryRate*(1-as->depAmpMFGO[i]) ); 

			if(apMFH[0][i])
			{
#pragma omp critical
				{
					for(int j=0; j<cs->numpMFfromMFtoGO[i]; j++)
					{
						as->inputMFGO[ cs->pMFfromMFtoGO[i][j] ]++;	
					}
				}
			
			}

		}
	
	}

	/*for(int i=0; i<cp->numGO; i++)
	{
		//as->gSum_MFGO[i] = 0;

		for(int j=0; j<cs->numpGOfromMFtoGO[i]; j++)
		{
			//as->gSum_MFGO[i] = as->gSum_MFGO[i] + as->gi_MFtoGO[ cs->pGOfromMFtoGO[i][j] ];	
		}	
	}*/


}





void InNet::updateGOtoGROutParameters(float GOGR, float spillFrac)
{

float scalerGOGR = /*ap->gIncDirectGOtoGR*/GOGR*ap->gIncFracSpilloverGOtoGR*1.4;
float halfShift = 12.0;//shift;
float steepness = 20.0;//steep; 
float recoveryRate = 1/ap->recoveryTauGO;
float baselvl = spillFrac*GOGR;


#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numGO; i++)
		{			
			as->depAmpGOGR[i] = 1;//as->apGO[i]*as->depAmpGOGR[i]*1/*ap->fracDepGO*/ + (!as->apGO[i])*( as->depAmpGOGR[i]+recoveryRate*(1-as->depAmpGOGR[i]) ); 


			as->dynamicAmpGOGR[i] = baselvl + ( scalerGOGR * ( 1/ (1+ (  exp((counter[i]-halfShift)/steepness)  )) ) );
			counter[i] = (1-as->apGO[i])*counter[i] + 1; 

#pragma omp critical
			{
				for(int j=0; j<numGPUs; j++)
				{
					depAmpGOH[j][i] = 1;//as->depAmpGOGR[i];
					dynamicAmpGOH[j][i] = as->dynamicAmpGOGR[i];
				}
			}
		
		}
	
	}



}





void InNet::updateGOtoGOOut()
{
//float gjCoupleScaler = 0.00004;
float gjCoupleScaler = ap->coupleRiRjRatioGO;

float recoveryRate = 1/ap->recoveryTauGO;

#pragma omp parallel
	{
#pragma omp for
		for(int i=0; i<cp->numGO; i++)
		{
			
			as->gi_GOtoGO[i] =  as->apGO[i]*ap->gGABAIncGOtoGO*as->depAmpGOGO[i]  +  as->gi_GOtoGO[i]*ap->gGABADecGOtoGO  ; 
			as->depAmpGOGO[i] = 1;//as->apGO[i]*as->depAmpGOGO[i]*1/*ap->fracDepGO*/ + (!as->apGO[i])*( as->depAmpGOGO[i]+recoveryRate*(1-as->depAmpGOGO[i]) ); 
		}

		for(int i=0; i<cp->numGO; i++)
		{
			/*as->gSum_GOGO[i] = 0;
			for(int j=0; j<cs->numpGOGABAInGOGO[i]; j++)
			{
				as->gSum_GOGO[i] += as->gi_GOtoGO[ cs->pGOGABAInGOGO[i][j]  ];
			}*/
		
			if(as->apGO[i]){
				
				for(int j=0; j<cs->numpGOGABAOutGOGO[i]; j++)
				{
					as->inputGOGO[ cs->pGOGABAOutGOGO[i][j] ]++;	
				}
			
			}
		}


#pragma omp for
		for(int i=0; i<cp->numGO; i++)
		{
			float threshCoupleGO;

			as->vCoupleGO[i]=0;

			threshCoupleGO=0;
			for(int j=0; j<cs->numpGOCoupInGOGO[i]; j++)
			{
				//as->vCoupleGO[i]=as->vCoupleGO[i]+(cs->pGOCoupInGOGOCCoeff[i][j]*ap->coupleRiRjRatioGO)*
				//		( as->vGO[cs->pGOCoupInGOGO[i][j]]-as->vGO[i] );
				
				as->vCoupleGO[i]= as->vCoupleGO[i] +( (as->vGO[cs->pGOCoupInGOGO[i][j]]-as->vGO[i])*(gjCoupleScaler*cs->pGOCoupInGOGOCCoeff[i][j]) );

			//	threshCoupleGO=threshCoupleGO+ap->coupleRiRjRatioGO*
			//			(as->threshCurGO[cs->pGOCoupInGOGO[i][j]]-as->threshCurGO[i]);
			}
	//		as->threshCurGO[i]=as->threshCurGO[i]+threshCoupleGO;
		}

	
	}

		

}

void InNet::resetMFHist(unsigned long t)
{
	if(t%ap->numTSinMFHist==0)
	{
#pragma omp parallel
		{
#pragma omp for
			for(int i=0; i<cp->numMF; i++)
			{
				as->histMF[i]=false;
			}
		}
	}
}
void InNet::runGRActivitiesCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	
	float gAMPAInc = (ap->gIncDirectMFtoGR)+(ap->gIncDirectMFtoGR*ap->gIncFracSpilloverMFtoGR); 

	for(int i=0; i<numGPUs; i++)
	{	
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"grActivityCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callGRActKernel(sts[i][streamN], calcGRActNumBlocks, calcGRActNumGRPerB,
				vGRGPU[i], gKCaGRGPU[i],  gLeakGRGPU[i], gNMDAGRGPU[i], gNMDAIncGRGPU[i], threshGRGPU[i],
				apBufGRGPU[i], outputGRGPU[i], apGRGPU[i],
				apMFtoGRGPU[i], apUBCtoGRGPU[i], gEGRSumGPU[i], gUBC_EGRSumGPU[i], gIGRSumGPU[i],
				ap->eLeakGR, ap->eGOGR, gAMPAInc,  
				ap->threshRestGR, ap->threshMaxGR, ap->threshDecGR);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"grActivityCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
		}

		//cout << "		FUCK		"<< endl;
}

void InNet::runSumPFBCCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runSumPFBCCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callSumKernel<ct_uint32_t, true, false>
		(sts[i][streamN], inputPFBCGPU[i], inputPFBCGPUP[i],
				inputSumPFBCGPU[i], 1, cp->numBC/numGPUs, 1, cp->numpBCfromGRtoBC);
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
//		cerr<<"runSumPFSCCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callSumKernel<ct_uint32_t, true, false>
		(sts[i][streamN], inputPFSCGPU[i], inputPFSCGPUP[i],
				inputSumPFSCGPU[i], 1, cp->numSC/numGPUs, 1, cp->numpSCfromGRtoSC);
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
//		cerr<<"runSumGRGOOutCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callSumGRGOOutKernel(sts[i][streamN], sumGRGOOutNumBlocks, sumGRGOOutNumGOPerB,
				updateGRGOOutNumGRRows, grInputGOGPU[i], grInputGOGPUP[i], grInputGOSumGPU[i]);
#ifdef DEBUGOUT
//		error=cudaGetLastError();
//		cerr<<"runSumGRGOOutCUDA: kernel launch for gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::runSumGRBCOutCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runSumGRGOOutCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callSumGRBCOutKernel(sts[i][streamN], sumGRBCOutNumBlocks, sumGRBCOutNumBCPerB,
				updateGRBCOutNumGRRows, grInputBCGPU[i], grInputBCGPUP[i], grInputBCSumGPU[i]);
#ifdef DEBUGOUT
//		error=cudaGetLastError();
//		cerr<<"runSumGRGOOutCUDA: kernel launch for gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void InNet::cpyDepAmpMFHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyAPMFHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(depAmpMFGPU[i], depAmpMFH[i], cp->numMF*sizeof(float), cudaMemcpyHostToDevice, sts[i][streamN]);
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
//		cerr<<"cpyAPMFHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(apMFGPU[i], apMFH[i], cp->numMF*sizeof(ct_uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPMFHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}





void InNet::cpyDepAmpUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
/*	
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyAPMFHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(depAmpUBCGPU[i], depAmpUBCH[i], cp->numUBC*sizeof(float), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPUBCHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
*/
}


void InNet::cpyAPUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
/*
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyAPMFHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(apUBCGPU[i], apUBCH[i], cp->numUBC*sizeof(ct_int32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyAPUBCHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
*/
}






void InNet::cpyDepAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	/*cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyAPMFHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(depAmpGOGPU[i], depAmpGOH[i], cp->numGO*sizeof(float), cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyDepAmpGOGRHosttoGPUCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}*/
}





void InNet::cpyDynamicAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"cpyAPMFHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(dynamicAmpGOGPU[i], dynamicAmpGOH[i], cp->numGO*sizeof(float), cudaMemcpyHostToDevice, sts[i][streamN]);
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
//		cerr<<"cpyAPGOHosttoGPUCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(apGOGPU[i], apGOH[i], cp->numGO*sizeof(ct_uint32_t), cudaMemcpyHostToDevice, sts[i][streamN]);
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
//		cerr<<"runUpdateMFInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateMFInGROPKernel(sts[i][streamN], updateMFInGRNumBlocks, updateMFInGRNumGRPerB,
				cp->numMF, apMFGPU[i], depAmpMFGRGPU[i] ,gEGRGPU[i], gEGRGPUP[i],   
				grConMFOutGRGPU[i], grConMFOutGRGPUP[i],
				numMFInPerGRGPU[i], apMFtoGRGPU[i], gEGRSumGPU[i], gEDirectGPU[i], gESpilloverGPU[i], 
				ap->gDirectDecMFtoGR, ap->gIncDirectMFtoGR, ap->gSpilloverDecMFtoGR, ap->gIncFracSpilloverMFtoGR);
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
//		cerr<<"runUpdateMFInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateMFInGRDepressionOPKernel(sts[i][streamN], updateMFInGRNumBlocks, updateMFInGRNumGRPerB,
				cp->numMF, depAmpMFGPU[i],
				grConMFOutGRGPU[i], grConMFOutGRGPUP[i],
				numMFInPerGRGPU[i], numMFperGR[i], depAmpMFGRGPU[i]);
#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateMFInGRDepressionCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}







void InNet::runUpdateUBCInGRCUDA(cudaStream_t **sts, int streamN)
{
/*	
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdateMFInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateUBCInGROPKernel(sts[i][streamN], updateUBCInGRNumBlocks, updateUBCInGRNumGRPerB,
				cp->numUBC, apUBCGPU[i], depAmpUBCGRGPU[i], gUBC_EGRGPU[i], gUBC_EGRGPUP[i],   
				grConUBCOutGRGPU[i], grConUBCOutGRGPUP[i],
				numUBCInPerGRGPU[i], apUBCtoGRGPU[i], gUBC_EGRSumGPU[i], gUBC_EDirectGPU[i], gUBC_ESpilloverGPU[i], 
				ap->gDirectDecMFtoGR, ap->gIncUBCtoGR, ap->gSpilloverDecMFtoGR, ap->gIncFracSpilloverMFtoGR);

#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateUBCInGRCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
*/
}


void InNet::runUpdateUBCInGRDepressionCUDA(cudaStream_t **sts, int streamN)
{
/*	
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdateMFInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateUBCInGRDepressionOPKernel(sts[i][streamN], updateUBCInGRNumBlocks, updateUBCInGRNumGRPerB,
				cp->numUBC, depAmpUBCGPU[i],
				grConUBCOutGRGPU[i], grConUBCOutGRGPUP[i],
				numUBCInPerGRGPU[i], numUBCperGR[i], depAmpUBCGRGPU[i]);

#ifdef DEBUGOUT
		error=cudaGetLastError();
		cerr<<"runUpdateUBCInGRDepressionCUDA: kernel launch for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
*/
}












void InNet::runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN, float GOGR)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdateGOInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateInGROPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				cp->numGO, apGOGPU[i], dynamicAmpGOGRGPU[i], gIGRGPU[i], gIGRGPUP[i],
				grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
				numGOInPerGRGPU[i], gIGRSumGPU[i], gIDirectGPU[i], gISpilloverGPU[i], 
				ap->gDirectDecGOtoGR, GOGR, ap->gIncFracSpilloverGOtoGR, ap->gSpilloverDecGOtoGR);
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
//		cerr<<"runUpdateMFInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateGOInGRDepressionOPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				cp->numMF, depAmpGOGPU[i],
				grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
				numGOInPerGRGPU[i], depAmpGOGRGPU[i]);
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
//		cerr<<"runUpdateMFInGRCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateGOInGRDynamicSpillOPKernel(sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB,
				cp->numMF, dynamicAmpGOGPU[i],
				grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
				numGOInPerGRGPU[i], dynamicAmpGOGRGPU[i]);
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
//		cerr<<"runUpdatePFBCSCOutCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdatePFBCSCOutKernel(sts[i][streamN], updatePFBCSCNumBlocks, updatePFBCSCNumGRPerB,
				apBufGRGPU[i], delayBCPCSCMaskGRGPU[i],
				inputPFBCGPU[i], inputPFBCGPUP[i], cp->numpBCfromGRtoBCP2,
				inputPFSCGPU[i], inputPFSCGPUP[i], cp->numpSCfromGRtoSCP2);
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
//		cerr<<"runUpdateGROutGOCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateGROutGOKernel(sts[i][streamN], updateGRGOOutNumGRRows, updateGRGOOutNumGRPerR,
				cp->numGO, apBufGRGPU[i], grInputGOGPU[i], grInputGOGPUP[i],
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
//		cerr<<"runUpdateGROutGOCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		callUpdateGROutBCKernel(sts[i][streamN], updateGRBCOutNumGRRows, updateGRBCOutNumGRPerR,
				cp->numBC, apBufGRGPU[i], grInputBCGPU[i], grInputBCGPUP[i],
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
//		cerr<<"cpyPFBCSumGPUtoHostCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(&inputSumPFBCH[cp->numBC*i/numGPUs], inputSumPFBCGPU[i],
				cp->numBC/numGPUs*sizeof(ct_uint32_t),
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
//		cerr<<"cpyPFSCSumGPUtoHostCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		error=cudaMemcpyAsync(&inputSumPFSCH[cp->numSC*i/numGPUs], inputSumPFSCGPU[i],
				cp->numSC/numGPUs*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
#ifdef DEBUGOUT
		cerr<<"cpyPFSCSumGPUtoHostCUDA: async copy for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}
void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
//	cudaError_t error;
//	for(int i=0; i<numGPUs; i++)
//	{
//		error=cudaSetDevice(i+gpuIndStart);
////		cerr<<"cpyGRGOSumGPUtoHostCUDA: switching to gpu #"<<i<<
////				": "<<cudaGetErrorString(error)<<endl;
//		error=cudaMemcpyAsync(grInputGOSumH[i], grInputGOSumGPU[i], cp->numGO*sizeof(ct_uint32_t),
//				cudaMemcpyDeviceToHost, sts[i][streamN]);
//#ifdef DEBUGOUT
////		cerr<<"cpyGRGOSumGPUtoHostCUDA: async copy for gpu #"<<i<<
////				": "<<cudaGetErrorString(error)<<endl;
//#endif
//	}

	cpyGRGOSumGPUtoHostCUDA(sts, streamN, grInputGOSumH);
}
void InNet::cpyGRBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{

	cpyGRBCSumGPUtoHostCUDA(sts, streamN, grInputBCSumH);
}
void InNet::cpyGRBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN, ct_uint32_t **grInputBCSumHost)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);

		error=cudaMemcpyAsync(grInputBCSumHost[i], grInputBCSumGPU[i], cp->numBC*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN, ct_uint32_t **grInputGOSumHost)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);

		error=cudaMemcpyAsync(grInputGOSumHost[i], grInputGOSumGPU[i], cp->numGO*sizeof(ct_uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void InNet::runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, unsigned long t)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
//		cerr<<"runUpdateGRHistoryCUDA: switching to gpu #"<<i<<
//				": "<<cudaGetErrorString(error)<<endl;
		if(t%ap->tsPerHistBinGR==0)
		{
			callUpdateGRHistKernel(sts[i][streamN], updateGRHistNumBlocks, updateGRHistNumGRPerB,
					apBufGRGPU[i], historyGRGPU[i], apBufGRHistMask);
#ifdef DEBUGOUT
//			error=cudaGetLastError();
//			cerr<<"runUpdateGRHistoryCUDA: kernel launch for gpu #"<<i<<
//					": "<<cudaGetErrorString(error)<<endl;
#endif
		}
	}
}
