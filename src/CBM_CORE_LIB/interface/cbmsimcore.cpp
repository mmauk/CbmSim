/*
 * cbmsimcore.cpp
 *
 *  Created on: Dec 15, 2011
 *      Author: consciousness
 */

#include "interface/cbmsimcore.h"

//#define NO_ASYNC
//#define DISP_CUDA_ERR

using namespace std;
CBMSimCore::CBMSimCore(CBMState *state, int *mzoneRSeed, int gpuIndStart, int numGPUP2)
{
	construct(state, mzoneRSeed, gpuIndStart, numGPUP2);

}

CBMSimCore::CBMSimCore(CBMState *state, int gpuIndStart, int numGPUP2)
{
	int *mzoneRSeed;
	CRandomSFMT0 *randGen;//(time(0));//(time(0));

	randGen=new CRandomSFMT0(time(0));
	mzoneRSeed=new int[state->getNumZones()];

	for(int i=0; i<state->getNumZones(); i++)
	{
		mzoneRSeed[i]=randGen->IRandom(0, INT_MAX);
	}

	construct(state, mzoneRSeed, gpuIndStart, numGPUP2);

	delete[] mzoneRSeed;
}

CBMSimCore::~CBMSimCore()
{
	delete inputNet;

//	for(int i=0; i<numZones; i++)
//	{
//		delete zones[i];
//	}
//	delete[] zones;

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		for(int j=0; j<8; j++)
		{
			cudaStreamDestroy(streams[i][j]);
		}
		delete[] streams[i];
	}

	delete[] streams;

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
	}
}

void CBMSimCore::writeToState()
{
	inputNet->writeToState();

	for(int i=0; i<numZones; i++)
	{
		zones[i]->writeToState();
	}
}

void CBMSimCore::writeToState(fstream& outfile)
{	
	writeToState();

	simState->writeState(outfile);
}

void CBMSimCore::initCUDA()
{
	cudaError_t error;

	int maxNumGPUs;

	error=cudaGetDeviceCount(&maxNumGPUs);

	cerr<<"CUDA max num devices: "<<maxNumGPUs<<", "<<cudaGetErrorString(error)<<endl;
	cerr<<"CUDA num devices: "<<numGPUs<<", starting at GPU# "<<gpuIndStart<<endl;

	streams = new cudaStream_t*[numGPUs];

	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		cerr<<"selecting device #"<<i<<": "<<cudaGetErrorString(error)<<endl;
		streams[i]=new cudaStream_t[8];
		cerr<<"resetting device #"<<i<<": "<<cudaGetErrorString(error)<<endl;
		cudaDeviceSynchronize();

		for(int j=0; j<8; j++)
		{
			//cout<<j<<endl;
			error=cudaStreamCreate(&streams[i][j]);
			cerr<<"initializing stream "<<j<<" for device "<<i<<
					": "<<cudaGetErrorString(error)<<endl;
		}
		cudaDeviceSynchronize();
		error=cudaGetLastError();
		cerr<<"CUDA dev "<<i<<": "<<cudaGetErrorString(error)<<endl;
	}
}

void CBMSimCore::initAuxVars()
{
	curTime=0;
}

void CBMSimCore::syncCUDA(string title)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
#ifdef DISP_CUDA_ERR
		cerr<<"sync point "<<title<<": switching to gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
		error=cudaDeviceSynchronize();
#ifdef DISP_CUDA_ERR
		cerr<<"sync point "<<title<<": sync for gpu #"<<i<<
				": "<<cudaGetErrorString(error)<<endl;
#endif
	}
}

void CBMSimCore::calcActivity(float goMin, int simNum, float GOGR, float GRGO, float MFGO, float gogoW, float spillFrac)
{
	cudaError_t error;
	syncCUDA("1");

	curTime++;

	inputNet->runGRActivitiesCUDA(streams, 0);
//	if(isGRStim){
//		inputNet->grStim(startGRStim, numGRStim);
//		isGRStim = false;
//	}


#ifdef NO_ASYNC
	syncCUDA("1a");
#endif

	for(int i=0; i<numZones; i++)
	{
		zones[i]->runPFPCSumCUDA(streams, i+1);
	}

#ifdef NO_ASYNC
	syncCUDA("1b");
#endif

	inputNet->runSumPFBCCUDA(streams, 2);


#ifdef NO_ASYNC
	syncCUDA("1c");
#endif

	inputNet->runSumPFSCCUDA(streams, 3);
	
#ifdef NO_ASYNC
	syncCUDA("1d");
#endif

	inputNet->runSumGRGOOutCUDA(streams, 4);
#ifdef NO_ASYNC
	syncCUDA("1e");
#endif
//					inputNet->runSumGRBCOutCUDA(streams, 1);
#ifdef NO_ASYNC
	syncCUDA("1f");
#endif

	// Only allow plasticty once HomeoTuning in GCL is complete
	if(curTime < 1000*5000){	
		for(int i=0; i<numZones; i++)
		{
			zones[i]->runPFPCPlastCUDA(streams, 1, curTime);
		}

	}
			#ifdef NO_ASYNC
				syncCUDA("1f");
			#endif
	inputNet->cpyDepAmpMFHosttoGPUCUDA(streams, 5);

#ifdef NO_ASYNC
	syncCUDA("1g");
#endif

	inputNet->cpyAPMFHosttoGPUCUDA(streams, 6);
#ifdef NO_ASYNC
	syncCUDA("1h");
#endif
	
	
	inputNet->cpyDepAmpGOGRHosttoGPUCUDA(streams, 2); 
	
#ifdef NO_ASYNC
	syncCUDA("1i");
#endif
	
	inputNet->cpyDynamicAmpGOGRHosttoGPUCUDA(streams, 3); 

#ifdef NO_ASYNC
	syncCUDA("1j");
#endif



	inputNet->cpyAPGOHosttoGPUCUDA(streams, 7);
#ifdef NO_ASYNC
	syncCUDA("1k");
#endif

//			inputNet->cpyDepAmpUBCHosttoGPUCUDA(streams, 1);
			
#ifdef NO_ASYNC
	syncCUDA("1l");
#endif

//			inputNet->cpyAPUBCHosttoGPUCUDA(streams, 2);


#ifdef NO_ASYNC
	syncCUDA("1m");
#endif





	for(int i=0; i<numZones; i++)
	{
		zones[i]->updateSCActivities(inputNet->exportAPSC());
		zones[i]->updatePFBCSum(inputNet->exportPFBCSum());
		zones[i]->calcPCActivities();
		zones[i]->calcBCActivities(inputNet->getGRInputBCSumHPointer());
		//////////////////zones[i]->calcBCActivities();
	}

	inputNet->calcSCActivities();
	
	syncCUDA("2");


//			inputNet->runUpdateUBCInGRCUDA(streams, 7);
#ifdef NO_ASYNC
	syncCUDA("2a");
#endif
	inputNet->runUpdateMFInGRCUDA(streams, 0);
#ifdef NO_ASYNC
	syncCUDA("2b");
#endif
	inputNet->runUpdateGOInGRCUDA(streams, 1, GOGR);
#ifdef NO_ASYNC
	syncCUDA("2c");
#endif


//	inputNet->runUpdateUBCInGRDepressionCUDA(streams, 2);

#ifdef NO_ASYNC
	syncCUDA("2d");
#endif
	

	inputNet->runUpdateMFInGRDepressionCUDA(streams, 2);



#ifdef NO_ASYNC
	syncCUDA("2e");
#endif




	inputNet->runUpdateGOInGRDepressionCUDA(streams, 3);

#ifdef NO_ASYNC
	syncCUDA("2f");
#endif

	inputNet->runUpdateGOInGRDynamicSpillCUDA(streams, 4);


	for(int i=0; i<numZones; i++)
	{
		zones[i]->runPFPCOutCUDA(streams, i+2);
		zones[i]->cpyPFPCSumCUDA(streams, i+2);
	}
#ifdef NO_ASYNC
	syncCUDA("2g");
#endif

	inputNet->runUpdatePFBCSCOutCUDA(streams, 4);
#ifdef NO_ASYNC
	syncCUDA("2h");
#endif

	inputNet->runUpdateGROutGOCUDA(streams, 7);
#ifdef NO_ASYNC
	syncCUDA("2i");
#endif
//			inputNet->runUpdateGROutBCCUDA(streams, 1);

#ifdef NO_ASYNC
	syncCUDA("2i");
#endif



		inputNet->cpyPFBCSumGPUtoHostCUDA(streams, 5);
#ifdef NO_ASYNC
	syncCUDA("2j");
#endif

		inputNet->cpyPFSCSumGPUtoHostCUDA(streams, 3);
#ifdef NO_ASYNC
	syncCUDA("2k");
#endif

	inputNet->cpyGRGOSumGPUtoHostCUDA(streams, 3);
#ifdef NO_ASYNC
	syncCUDA("2ia");
#endif
//			inputNet->cpyGRBCSumGPUtoHostCUDA(streams, 2);
#ifdef NO_ASYNC
	syncCUDA("2iz");
#endif

	inputNet->runUpdateGRHistoryCUDA(streams, 4, curTime);
#ifdef NO_ASYNC
	syncCUDA("2ib");
#endif

	inputNet->calcGOActivities(goMin, simNum, GRGO, MFGO, GOGR, gogoW);
#ifdef NO_ASYNC
	syncCUDA("2ic");
#endif
	
	//inputNet->updateMFtoUBCOut();
#ifdef NO_ASYNC
	syncCUDA("2id");
#endif
	//inputNet->updateGOtoUBCOut();
#ifdef NO_ASYNC
	syncCUDA("2ie");
#endif

	
	
	inputNet->updateMFtoGOOut();
#ifdef NO_ASYNC
	syncCUDA("2if");
#endif
	inputNet->updateGOtoGOOut();
#ifdef NO_ASYNC
	syncCUDA("2ig");
#endif

	inputNet->updateMFtoGROut();
#ifdef NO_ASYNC
	syncCUDA("2ih");
#endif

	inputNet->updateGOtoGROutParameters(GOGR, spillFrac);
#ifdef NO_ASYNC
	syncCUDA("2ii");
#endif


//	inputNet->calcUBCActivities();

#ifdef NO_ASYNC
	syncCUDA("2ij");
#endif

//	inputNet->updateUBCtoUBCOut();

#ifdef NO_ASYNC
	syncCUDA("2ik");
#endif


//	inputNet->updateUBCtoGOOut();


#ifdef NO_ASYNC
	syncCUDA("2il");
#endif

//	inputNet->updateUBCtoGROut();

#ifdef NO_ASYNC
	syncCUDA("2im");
#endif

	for(int i=0; i<numZones; i++)
	{
		zones[i]->calcIOActivities();
#ifdef NO_ASYNC
		syncCUDA("2in");
#endif

		zones[i]->calcNCActivities();
#ifdef NO_ASYNC
		syncCUDA("2io");
#endif

		zones[i]->updateMFNCOut();
#ifdef NO_ASYNC
		syncCUDA("2ip");
#endif

		zones[i]->updateBCPCOut();
#ifdef NO_ASYNC
		syncCUDA("2iq");
#endif
		zones[i]->updateSCPCOut();
#ifdef NO_ASYNC
		syncCUDA("2ir");
#endif


		zones[i]->updatePCOut();
#ifdef NO_ASYNC
		syncCUDA("2is");
#endif

		zones[i]->updateIOOut();
#ifdef NO_ASYNC
		syncCUDA("2it");
#endif
		zones[i]->updateNCOut();
#ifdef NO_ASYNC
		syncCUDA("2iu");
#endif
		
		// Only allow plasticity once HomeoTuning in GCL is complete.
/*		if(curTime > (1000000*10000)+(1000*5000)){
			zones[i]->updateMFNCSyn(inputNet->exportHistMF(), curTime);
			
			#ifdef NO_ASYNC
				syncCUDA("2iv");
			#endif
		}
	*/
	}


#ifdef NO_ASYNC
		syncCUDA("2iw");
#endif


	inputNet->resetMFHist(curTime);
#ifdef NO_ASYNC
		syncCUDA("2ix");
#endif
}

void CBMSimCore::updateMFInput(const ct_uint8_t *mfIn)
{
	inputNet->updateMFActivties(mfIn);

	for(int i=0; i<numZones; i++)
	{
		zones[i]->updateMFActivities(mfIn);
	}
}

void CBMSimCore::updateTrueMFs(bool *isTrueMF)
{

	for(int i=0; i<numZones; i++)
	{
			zones[i]->updateTrueMFs(isTrueMF);
	}
}

void CBMSimCore::updateGRStim(int startGRStim_, int numGRStim_)
{
	isGRStim = true;
	this->numGRStim = numGRStim_;
	this->startGRStim = startGRStim_;
}

void CBMSimCore::updateErrDrive(unsigned int zoneN, float errDriveRelative)
{
	zones[zoneN]->setErrDrive(errDriveRelative);
}

InNetInterface* CBMSimCore::getInputNet()
{
	return (InNetInterface *)inputNet;
}

MZoneInterface** CBMSimCore::getMZoneList()
{
	return (MZoneInterface **)zones;
}

void CBMSimCore::construct(CBMState *state, int *mzoneRSeed, int gpuIndStart, int numGPUP2)
{
	int maxNumGPUs;
	simState=state;

	numZones=simState->getNumZones();

	cudaGetDeviceCount(&maxNumGPUs);

	if(gpuIndStart<=0)
	{
		this->gpuIndStart=0;
	}
	else if(gpuIndStart>=maxNumGPUs)
	{
		this->gpuIndStart=maxNumGPUs-1;
	}
	else
	{
		this->gpuIndStart=gpuIndStart;
	}

	if(numGPUP2<0)
	{
		numGPUs=maxNumGPUs;
	}
	else
	{
		//numGPUs=(((unsigned int)1)<<numGPUP2);
		numGPUs=(unsigned int)numGPUP2;
	}

	if(this->gpuIndStart+numGPUs>maxNumGPUs)
	{
		numGPUs=1;
	}
	cout << numGPUs << endl;
	initCUDA();
	cout << "passed cuda" << endl;

	inputNet=new InNet(simState->getConParamsInternal(), simState->getActParamsInternal(),
			simState->getInnetConStateInternal(), simState->getInnetActStateInternal(),
			this->gpuIndStart, numGPUs);
//		new InNetAllGRMFGO(simState->getConParamsInternal(), simState->getActParamsInternal(),
//			simState->getInnetConStateInternal(), simState->getInnetActStateInternal(),
//			this->gpuIndStart, numGPUs);
	cout << "FUCKCKCK" << endl;
	zones=new MZone*[numZones];
	for(int i=0; i<numZones; i++)
	{
		zones[i]=new MZone(simState->getConParamsInternal(), simState->getActParamsInternal(),
				simState->getMZoneConStateInternal(i), simState->getMZoneActStateInternal(i),
				mzoneRSeed[i], inputNet->getApBufGRGPUPointer(),
				inputNet->getDelayBCPCSCMaskGPUPointer(), inputNet->getHistGRGPUPointer(),
				this->gpuIndStart, numGPUs);
	}
	cout << "Mzone construction complete" << endl;
	
	initAuxVars();
	cout << "AuxVars good" << endl;
}

