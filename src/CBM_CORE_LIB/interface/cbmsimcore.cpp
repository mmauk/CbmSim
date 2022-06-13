/*
 * cbmsimcore.cpp
 *
 *  Created on: Dec 15, 2011
 *      Author: consciousness
 */

#include "interface/cbmsimcore.h"

//#define NO_ASYNC
//#define DISP_CUDA_ERR

CBMSimCore::CBMSimCore() {}

CBMSimCore::CBMSimCore(ActivityParams &ap, CBMState *state,
	int gpuIndStart, int numGPUP2)
{
	CRandomSFMT0 randGen(time(0));
	int *mzoneRSeed = new int[state->getNumZones()];

	for (int i = 0; i < state->getNumZones(); i++)
	{
		mzoneRSeed[i] = randGen.IRandom(0, INT_MAX);
	}

	construct(ap, state, mzoneRSeed, gpuIndStart, numGPUP2);

	delete[] mzoneRSeed;
}

CBMSimCore::~CBMSimCore()
{
	delete inputNet;

	for (int i = 0; i < numGPUs; i++)
	{
		// How could gpuIndStart ever not be 0,
		// given we're looping from 0 to numGPUs?
		cudaSetDevice(i + gpuIndStart);

		for (int j = 0; j < 8; j++)
		{
			cudaStreamDestroy(streams[i][j]);
		}
		delete[] streams[i];
	}

	delete[] streams;

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
	}
}

void CBMSimCore::writeToState()
{
	inputNet->writeToState();

	for (int i = 0; i < numZones; i++)
	{
		zones[i]->writeToState();
	}
}

void CBMSimCore::writeState(ActivityParams &ap, std::fstream& outfile)
{	
	writeToState();
	simState->writeState(ap, outfile);
}

void CBMSimCore::initCUDAStreams()
{
	cudaError_t error;

	int maxNumGPUs;
	// TODO: use assert, try, and catch for these types of errors
	error = cudaGetDeviceCount(&maxNumGPUs);

	std::cerr << "CUDA max num devices: " << maxNumGPUs << ", "
		<< cudaGetErrorString(error) << std::endl;
	std::cerr << "CUDA num devices: " << numGPUs << ", starting at GPU# "
		<< gpuIndStart << std::endl;

	streams = new cudaStream_t*[numGPUs];

	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
		std::cerr << "selecting device #" << i << ": " << cudaGetErrorString(error) << std::endl;
		streams[i] = new cudaStream_t[8];
		std::cerr << "resetting device #" << i << ": " << cudaGetErrorString(error) << std::endl;
		cudaDeviceSynchronize();

		for (int j = 0; j < 8; j++)
		{
			error = cudaStreamCreate(&streams[i][j]);
			std::cerr << "initializing stream " << j << " for device " << i <<
					": "<<cudaGetErrorString(error) << std::endl;
		}
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		std::cerr << "CUDA dev " << i << ": " << cudaGetErrorString(error) << std::endl;
	}
}

void CBMSimCore::initAuxVars()
{
	curTime = 0;
}

void CBMSimCore::syncCUDA(std::string title)
{
	cudaError_t error;
	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
#ifdef DISP_CUDA_ERR
		std::cerr << "sync point " << title << ": switching to gpu #" << i <<
				": " << cudaGetErrorString(error) << std::endl;
#endif

		error = cudaDeviceSynchronize();
#ifdef DISP_CUDA_ERR
		std::cerr << "sync point " << title << ": sync for gpu #" << i <<
				": " << cudaGetErrorString(error) << std::endl;
#endif
	}
}

void CBMSimCore::calcActivity(float goMin, int simNum, float GOGR, float GRGO, float MFGO,
	float gogoW, float spillFrac)
{
	cudaError_t error;
	syncCUDA("1");

	curTime++;

	inputNet->runGRActivitiesCUDA(streams, 0);

#ifdef NO_ASYNC
	syncCUDA("1a");
#endif

	for (int i = 0; i < numZones; i++)
	{
		zones[i]->runPFPCSumCUDA(streams, i + 1);
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

#ifdef NO_ASYNC
	syncCUDA("1f");
#endif

	// Only allow plasticty once HomeoTuning in GCL is complete
	if (curTime < 1000 * 5000)
	{	
		for (int i = 0; i < numZones; i++)
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
			
#ifdef NO_ASYNC
	syncCUDA("1l");
#endif

#ifdef NO_ASYNC
	syncCUDA("1m");
#endif

	for (int i = 0; i < numZones; i++)
	{
		zones[i]->updateSCActivities(inputNet->exportAPSC());
		zones[i]->updatePFBCSum(inputNet->exportPFBCSum());
		zones[i]->calcPCActivities();
		zones[i]->calcBCActivities(inputNet->getGRInputBCSumHPointer());
	}

	inputNet->calcSCActivities();
	
	syncCUDA("2");

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

	for (int i = 0; i < numZones; i++)
	{
		zones[i]->runPFPCOutCUDA(streams, i + 2);
		zones[i]->cpyPFPCSumCUDA(streams, i + 2);
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
	
#ifdef NO_ASYNC
	syncCUDA("2id");
#endif

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

#ifdef NO_ASYNC
	syncCUDA("2ij");
#endif

#ifdef NO_ASYNC
	syncCUDA("2ik");
#endif

#ifdef NO_ASYNC
	syncCUDA("2il");
#endif

#ifdef NO_ASYNC
	syncCUDA("2im");
#endif

	for (int i = 0; i < numZones; i++)
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

	for (int i = 0; i < numZones; i++)
	{
		zones[i]->updateMFActivities(mfIn);
	}
}

void CBMSimCore::updateTrueMFs(bool *isTrueMF)
{

	for (int i = 0; i < numZones; i++)
	{
			zones[i]->updateTrueMFs(isTrueMF);
	}
}

void CBMSimCore::updateGRStim(int startGRStim_, int numGRStim_)
{
	isGRStim 		  = true;
	this->numGRStim   = numGRStim_;
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

void CBMSimCore::construct(ActivityParams &ap, CBMState *state,
	int *mzoneRSeed, int gpuIndStart, int numGPUP2)
{
	int maxNumGPUs;

	numZones = state->getNumZones();

	cudaGetDeviceCount(&maxNumGPUs);

	if (gpuIndStart <= 0)
	{
		this->gpuIndStart = 0;
	}
	else if (gpuIndStart >= maxNumGPUs)
	{
		this->gpuIndStart = maxNumGPUs - 1;
	}
	else
	{
		this->gpuIndStart = gpuIndStart;
	}

	if (numGPUP2 < 0)
	{
		numGPUs = maxNumGPUs;
	}
	else
	{
		numGPUs = (unsigned int)numGPUP2;
	}

	if (this->gpuIndStart + numGPUs > maxNumGPUs)
	{
		numGPUs = 1;
	}
	std::cout << " calculated (?) number of GPUs: " << numGPUs << std::endl;

	std::cout << "initializing cuda streams..." << std::endl;
	initCUDAStreams();
	std::cout << "finished initialzing cuda streams." << std::endl;

	// NOTE: inputNet has internal cp, no need to pass to constructor
	inputNet = new InNet(ap, state->getInnetConStateInternal(),
		state->getInnetActStateInternal(), this->gpuIndStart, numGPUs);

	zones = new MZone*[numZones];

	for (int i = 0; i < numZones; i++)
	{
		// same thing for zones as with innet	
		zones[i] = new MZone(ap, state->getMZoneConStateInternal(i),
			state->getMZoneActStateInternal(i), mzoneRSeed[i], inputNet->getApBufGRGPUPointer(),
			inputNet->getDelayBCPCSCMaskGPUPointer(), inputNet->getHistGRGPUPointer(),
			this->gpuIndStart, numGPUs);
	}
	std::cout << "Mzone construction complete" << std::endl;
	
	initAuxVars();
	std::cout << "AuxVars good" << std::endl;

	simState = state; // shallow copy
}

