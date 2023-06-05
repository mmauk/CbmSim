/*
 * cbmsimcore.cpp
 *
 *  Created on: Dec 15, 2011
 *      Author: consciousness
 */

#include "logger.h"
#include "cbmsimcore.h"

//#define NO_ASYNC
//#define DISP_CUDA_ERR

CBMSimCore::CBMSimCore() {}

CBMSimCore::CBMSimCore(CBMState *state,
	int gpuIndStart, int numGPUP2)
{
	CRandomSFMT0 randGen(time(0));
	int *mzoneRSeed = new int[state->getNumZones()];

	for (int i = 0; i < state->getNumZones(); i++)
	{
		mzoneRSeed[i] = randGen.IRandom(0, INT_MAX);
	}

	construct(state, mzoneRSeed, gpuIndStart, numGPUP2);

	delete[] mzoneRSeed;
}

CBMSimCore::~CBMSimCore()
{
	for (int i = 0; i < numZones; i++)
	{
		delete zones[i];
	}

	delete[] zones;
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
}

// for speed
void CBMSimCore::writeToState()
{
	inputNet->writeToState();

	for (int i = 0; i < numZones; i++)
	{
		zones[i]->writeToState();
	}
}

void CBMSimCore::writeState(std::fstream& outfile)
{
	writeToState();
	simState->writeState(outfile); // using internal cp
}

void CBMSimCore::initCUDAStreams()
{
	cudaError_t error;

	int maxNumGPUs;
	// TODO: use assert, try, and catch for these types of errors
	error = cudaGetDeviceCount(&maxNumGPUs);

	LOG_DEBUG("CUDA max num devices: %d", maxNumGPUs);
	LOG_DEBUG("%s", cudaGetErrorString(error));
	LOG_DEBUG("CUDA num devices: %d, starting at GPU %d", numGPUs, gpuIndStart);

	streams = new cudaStream_t*[numGPUs];

	for (int i = 0; i < numGPUs; i++)
	{
		error = cudaSetDevice(i + gpuIndStart);
		LOG_DEBUG("Selecting device %d", i);
		LOG_DEBUG("%s", cudaGetErrorString(error));
		streams[i] = new cudaStream_t[8];
		LOG_DEBUG("Resetting device %d", i);
		LOG_DEBUG("%s", cudaGetErrorString(error));
		cudaDeviceSynchronize();

		for (int j = 0; j < 8; j++)
		{
			error = cudaStreamCreate(&streams[i][j]);
			LOG_DEBUG("Initializing stream %d for device %d",j, i);
			LOG_DEBUG("%s", cudaGetErrorString(error));
		}
		cudaDeviceSynchronize();
		error = cudaGetLastError();
		LOG_DEBUG("Cuda device %d", i);
		LOG_DEBUG("%s", cudaGetErrorString(error));
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
		LOG_TRACE("sync point  %s, switching to gpu %d", title.c_str(), i);
		LOG_TRACE("%s", cudaGetErrorString(error));
#endif
		error = cudaDeviceSynchronize();
#ifdef DISP_CUDA_ERR
		LOG_TRACE("sync point  %s, switching to gpu %d", title.c_str(), i);
		LOG_TRACE("%s", cudaGetErrorString(error));
#endif
	}
}

void CBMSimCore::calcActivity(float spillFrac, enum plasticity pf_pc_plast, enum plasticity mf_nc_plast)
{
	syncCUDA("1");
	curTime++;

	inputNet->cpyAPMFHosttoGPUCUDA(streams, 6);
	inputNet->updateMFtoGROut();
	inputNet->cpyDepAmpMFHosttoGPUCUDA(streams, 5);

	inputNet->runUpdateMFInGRDepressionCUDA(streams, 2);
	inputNet->runUpdateMFInGRCUDA(streams, 0);

	inputNet->runGRActivitiesCUDA(streams, 0);

	inputNet->runUpdateGRHistoryCUDA(streams, 4, curTime);

	inputNet->runUpdateGROutGOCUDA(streams, 7);
	inputNet->runSumGRGOOutCUDA(streams, 4);
	inputNet->cpyGRGOSumGPUtoHostCUDA(streams, 3);

	inputNet->updateMFtoGOOut();
	inputNet->runGOActivitiesCUDA(streams, 5);

	inputNet->updateGOtoGOOut();
	inputNet->updateGOtoGROutParameters(spillFrac);

	inputNet->cpyDepAmpGOGRHosttoGPUCUDA(streams, 2); // NOTE: currently does nothing (08/11/2022)
	inputNet->cpyDynamicAmpGOGRHosttoGPUCUDA(streams, 3);
	inputNet->cpyAPGOHosttoGPUCUDA(streams, 7);

	// TODO: put in macro def for num_gpus so we don't run this line if running on one GPU
	//syncCUDA("2");
	inputNet->runUpdateGOInGRCUDA(streams, 1);
	inputNet->runUpdateGOInGRDepressionCUDA(streams, 3);
	inputNet->runUpdateGOInGRDynamicSpillCUDA(streams, 4);

	if (pf_pc_plast == GRADED)
	{
		for (int i = 0; i < numZones; i++)
		{
			zones[i]->runPFPCPlastCUDA(streams, 1, curTime);
		}
	}

	for (int i = 0; i < numZones; i++)
	{
		zones[i]->runPFPCOutCUDA(streams, i + 2);
		zones[i]->runPFPCSumCUDA(streams, 1);
		zones[i]->cpyPFPCSumCUDA(streams, i + 2);

		zones[i]->runUpdatePFBCSCOutCUDA(streams, i + 4); // adding i might break things in future
		zones[i]->cpyPFBCSumGPUtoHostCUDA(streams, 5);
		zones[i]->cpyPFSCSumGPUtoHostCUDA(streams, 3);
		zones[i]->runSumPFBCCUDA(streams, 2);
		zones[i]->runSumPFSCCUDA(streams, 3);

		zones[i]->calcSCActivities();
		zones[i]->calcBCActivities();
		zones[i]->updateBCPCOut();
		zones[i]->updateSCPCOut();

		zones[i]->calcPCActivities();
		zones[i]->updatePCOut();

		zones[i]->calcIOActivities();
		zones[i]->updateIOOut();

		// temp solution: by default mfnc plast is GRADED. no other
		// plasticity modes are given for these synapses
		if (mf_nc_plast != OFF )
		{
			zones[i]->updateMFNCSyn(inputNet->exportHistMF(), curTime);
		}
		zones[i]->updateMFNCOut();
		zones[i]->calcNCActivities();
		zones[i]->updateNCOut();
	}
	inputNet->resetMFHist(curTime);
}

void CBMSimCore::updateMFInput(const uint8_t *mfIn)
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

InNet* CBMSimCore::getInputNet()
{
	return (InNet *)inputNet;
}

MZone** CBMSimCore::getMZoneList()
{
	return (MZone **)zones;
}

void CBMSimCore::construct(CBMState *state,
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
	LOG_DEBUG("Calculated (?) number of GPUs: %d", numGPUs);

	LOG_DEBUG("Initializing cuda streams...");
	initCUDAStreams();
	LOG_DEBUG("Finished initialzing cuda streams.");

	// NOTE: inputNet has internal cp, no need to pass to constructor
	inputNet = new InNet(state->getInnetConStateInternal(),
		state->getInnetActStateInternal(), this->gpuIndStart, numGPUs);

	zones = new MZone*[numZones];

	for (int i = 0; i < numZones; i++)
	{
		// same thing for zones as with innet
		zones[i] = new MZone(state->getMZoneConStateInternal(i),
			state->getMZoneActStateInternal(i), mzoneRSeed[i], inputNet->getApBufGRGPUPointer(),
			inputNet->getHistGRGPUPointer(), this->gpuIndStart, numGPUs);
	}
	LOG_DEBUG("Mzone construction complete");
	initAuxVars();
	LOG_DEBUG("AuxVars good");

	simState = state; // shallow copy
}

