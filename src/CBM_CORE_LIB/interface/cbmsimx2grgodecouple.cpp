/*
 * cbmsimx2grgodecouple.cpp
 *
 *  Created on: Apr 30, 2013
 *      Author: consciousness
 */

#include "interface/cbmsimx2grgodecouple.h"

using namespace std;

CBMSimX2GRGODecouple::CBMSimX2GRGODecouple(CBMStateX2GRGODecouple *state)
{
	CRandomSFMT0 randGen(time(0));

	simState=state;

	numInnets=state->getNumInnets();
	numZones=state->getNumZones();

	initCUDA();

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]=new InNet(simState->getConParamsInternal(), simState->getActParamsInternal(i),
				simState->getInnetConStateInternal(i), simState->getInnetActStateInternal(i),
				0, numGPUs);
	}

	for(int i=0; i<numZones; i++)
	{
		zones[i]=new MZone(simState->getConParamsInternal(), simState->getActParamsInternal(i),
				simState->getMZoneConStateInternal(i), simState->getMZoneActStateInternal(i),
				randGen.IRandom(0, INT_MAX), inputNets[i]->getApBufGRGPUPointer(),
				inputNets[i]->getDelayBCPCSCMaskGPUPointer(), inputNets[i]->getHistGRGPUPointer(),
				0, numGPUs);
	}

	initAuxVars();
}

CBMSimX2GRGODecouple::~CBMSimX2GRGODecouple()
{
	for(int i=0; i<numInnets; i++)
	{
		delete inputNets[i];
	}

	for(int i=0; i<numZones; i++)
	{
		delete zones[i];
	}

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i);
		for(int j=0; j<8; j++)
		{
			cudaStreamDestroy(streams[i][j]);
		}
		delete[] streams[i];
	}

	delete[] streams;

	for(int i=0; i<numGPUs; i++)
	{
		cudaSetDevice(i);
		cudaDeviceReset();
	}
}

void CBMSimX2GRGODecouple::initCUDA()
{
	cudaError_t error;

	error=cudaGetDeviceCount(&numGPUs);
	cerr<<"CUDA num devices: "<<numGPUs<<", "<<cudaGetErrorString(error)<<endl;

	streams = new cudaStream_t*[numGPUs];

	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i);
		cerr<<"selecting device #"<<i<<": "<<cudaGetErrorString(error)<<endl;
		error=cudaDeviceReset();
		streams[i]=new cudaStream_t[8];
		cerr<<"resetting device #"<<i<<": "<<cudaGetErrorString(error)<<endl;
		cudaDeviceSynchronize();

		for(int j=0; j<8; j++)
		{
//			cout<<j<<endl;
			error=cudaStreamCreate(&streams[i][j]);
			cerr<<"initializing stream "<<j<<" for device "<<i<<
					": "<<cudaGetErrorString(error)<<endl;
		}
		cudaDeviceSynchronize();
		error=cudaGetLastError();
		cerr<<"CUDA dev "<<i<<": "<<cudaGetErrorString(error)<<endl;
	}
}

void CBMSimX2GRGODecouple::initAuxVars()
{
	curTime=0;
}

void CBMSimX2GRGODecouple::syncCUDA(string title)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i);
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

void CBMSimX2GRGODecouple::calcActivity()
{
	cudaError_t error;
	syncCUDA("1");

	curTime++;

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->runGRActivitiesCUDA(streams, 0);
	}
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

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->runSumPFBCCUDA(streams, 2);
	}
#ifdef NO_ASYNC
	syncCUDA("1c");
#endif

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->runSumPFSCCUDA(streams, 3);
	}
#ifdef NO_ASYNC
	syncCUDA("1d");
#endif

	inputNets[0]->runSumGRGOOutCUDA(streams, 4);
#ifdef NO_ASYNC
	syncCUDA("1e");
#endif

	for(int i=0; i<numZones; i++)
	{
		zones[i]->runPFPCPlastCUDA(streams, 1, curTime);
	}
#ifdef NO_ASYNC
	syncCUDA("1f");
#endif

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->cpyAPMFHosttoGPUCUDA(streams, 6);
	}
#ifdef NO_ASYNC
	syncCUDA("1g");
#endif

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->cpyAPGOHosttoGPUCUDA(streams, 7);
	}
#ifdef NO_ASYNC
	syncCUDA("1h");
#endif

	for(int i=0; i<numZones; i++)
	{
		zones[i]->updateSCActivities(inputNets[i]->exportAPSC());
		zones[i]->updatePFBCSum(inputNets[i]->exportPFBCSum());
		zones[i]->calcPCActivities();
		zones[i]->calcBCActivities(inputNets[i]->getGRInputBCSumHPointer());
		//zones[i]->calcBCActivities();
	}

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->calcSCActivities();
	}
	syncCUDA("2");

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->runUpdateMFInGRCUDA(streams, 0);
	}
#ifdef NO_ASYNC
	syncCUDA("2a");
#endif

	for(int i=0; i<numInnets; i++)
	{
		//inputNets[i]->runUpdateGOInGRCUDA(streams, 1);
	}
#ifdef NO_ASYNC
	syncCUDA("2b");
#endif

	for(int i=0; i<numZones; i++)
	{
		zones[i]->runPFPCOutCUDA(streams, i+2);
		zones[i]->cpyPFPCSumCUDA(streams, i+2);
	}
#ifdef NO_ASYNC
	syncCUDA("2c");
#endif

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->runUpdatePFBCSCOutCUDA(streams, 4);
	}
#ifdef NO_ASYNC
	syncCUDA("2d");
#endif

	inputNets[0]->runUpdateGROutGOCUDA(streams, 7);
#ifdef NO_ASYNC
	syncCUDA("2e");
#endif

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->cpyPFBCSumGPUtoHostCUDA(streams, 5);
	}
#ifdef NO_ASYNC
	syncCUDA("2f");
#endif

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->cpyPFSCSumGPUtoHostCUDA(streams, 3);
	}
#ifdef NO_ASYNC
	syncCUDA("2g");
#endif


	inputNets[0]->cpyGRGOSumGPUtoHostCUDA(streams, 3);
	inputNets[0]->cpyGRGOSumGPUtoHostCUDA(streams, 3, inputNets[1]->getGRInputGOSumHPointer());
#ifdef NO_ASYNC
	syncCUDA("2h");
#endif

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->runUpdateGRHistoryCUDA(streams, 4, curTime);
	}
#ifdef NO_ASYNC
	syncCUDA("2i");
#endif

	for(int i=0; i<numInnets; i++)
	{
		//inputNets[i]->calcGOActivities(goMinimum);
	}
#ifdef NO_ASYNC
	syncCUDA("2ia");
#endif

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->updateMFtoGOOut();
	}
#ifdef NO_ASYNC
	syncCUDA("2ib");
#endif

	inputNets[0]->updateGOtoGOOut();
#ifdef NO_ASYNC
	syncCUDA("2ic");
#endif
	for(int i=0; i<numZones; i++)
	{
		zones[i]->calcIOActivities();
#ifdef NO_ASYNC
		syncCUDA("2id");
#endif

		zones[i]->calcNCActivities();
#ifdef NO_ASYNC
		syncCUDA("2ie");
#endif

		zones[i]->updateMFNCOut();
#ifdef NO_ASYNC
		syncCUDA("2if");
#endif
		zones[i]->updateBCPCOut();
#ifdef NO_ASYNC
		syncCUDA("2ig");
#endif
		zones[i]->updateSCPCOut();
#ifdef NO_ASYNC
		syncCUDA("2ih");
#endif
		zones[i]->updatePCOut();
#ifdef NO_ASYNC
		syncCUDA("2ii");
#endif

		zones[i]->updateIOOut();
#ifdef NO_ASYNC
		syncCUDA("2ij");
#endif
		zones[i]->updateNCOut();
#ifdef NO_ASYNC
		syncCUDA("2ik");
#endif
		zones[i]->updateMFNCSyn(inputNets[i]->exportHistMF(), curTime);
#ifdef NO_ASYNC
		syncCUDA("2il");
#endif
	}

	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->resetMFHist(curTime);
	}
#ifdef NO_ASYNC
		syncCUDA("2im");
#endif
}

void CBMSimX2GRGODecouple::updateMFInput(const ct_uint8_t *mfIn)
{
	for(int i=0; i<numInnets; i++)
	{
		inputNets[i]->updateMFActivties(mfIn);
	}

	for(int i=0; i<numZones; i++)
	{
		zones[i]->updateMFActivities(mfIn);
	}
}

void CBMSimX2GRGODecouple::updateErrDrive(float errDriveRelative)
{
	for(int i=0; i<numZones; i++)
	{
		zones[i]->setErrDrive(errDriveRelative);
	}
}

InNetInterface** CBMSimX2GRGODecouple::getInputNetList()
{
	return (InNetInterface **)inputNets;
}

MZoneInterface** CBMSimX2GRGODecouple::getMZoneList()
{
	return (MZoneInterface **)zones;
}
