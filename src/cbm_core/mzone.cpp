/*
 * mzone.cpp
 *
 *   Created on: Jun 14, 2011
 *   Author: consciousness
 *
 */
#include <math.h>
#include <iostream>
#include <fstream>

#include "logger.h"
#include "connectivityparams.h"
#include "activityparams.h"
#include "dynamic2darray.h"
#include "sfmt.h"
#include "file_utility.h"
#include "mzone.h"

MZone::MZone() {}

MZone::MZone(cudaStream_t **stream, MZoneConnectivityState *cs, MZoneActivityState *as,
		int randSeed, uint32_t **apBufGRGPU, uint64_t **histGRGPU, int gpuIndStart, int numGPUs)
{
	randGen = new CRandomSFMT0(randSeed);

	// shallow copies. caller owns the data.
	this->cs = cs; 
	this->as = as; 

	// NOTE if we turn these guys into unique ptrs, we'll have to refactor
	// consider ownership: who should own these guys? maybe they should be global to both
	// innet and mzone (so within cbmsimcore) and fed in as const args to the respective
	// functions that call update kernels (06/16/2022)

	this->apBufGRGPU     = apBufGRGPU;
	this->histGRGPU      = histGRGPU;

	delayMaskGRGPU = new uint32_t*[numGPUs];

	pfSynWeightPCLinear       = new float[num_gr];
	pfPCSynWeightStatesLinear = new uint8_t[num_gr];
	pfPCPlastStepIO           = new float[num_io];

	this->numGPUs     = numGPUs;
	this->gpuIndStart = gpuIndStart;

	LOG_DEBUG("Initializing CUDA...");
	initCUDA(stream);
}

MZone::~MZone()
{
	LOG_DEBUG("Deleting mzone gpu arrays...");

	delete randGen;

	delete[] pfSynWeightPCLinear;
	delete[] pfPCSynWeightStatesLinear;
	delete[] pfPCPlastStepIO;

	// free cuda memory for rngs
	// FIXME: check for memory leaks
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaFree(mrg32k3aRNGs[i]);
		cudaFree(pfpcSynWRandNums[i]);
		cudaDeviceSynchronize();
	}
	delete[] mrg32k3aRNGs;
	delete[] pfpcSynWRandNums;

	//free cuda host memory
	cudaSetDevice(0 + gpuIndStart);
	cudaFreeHost(inputSumPFPCMZH);
	cudaDeviceSynchronize();

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		//free cuda device memory
		cudaFree(delayMaskGRGPU[i]);
		cudaFree(pfSynWeightPCGPU[i]);
		cudaFree(pfPCSynWeightStatesGPU[i]);
		cudaFree(inputPFPCGPU[i]);
		cudaFree(inputSumPFPCMZGPU[i]);
		cudaDeviceSynchronize();
	}

	delete[] delayMaskGRGPU;
	delete[] pfSynWeightPCGPU;
	delete[] pfPCSynWeightStatesGPU;
	delete[] inputPFPCGPU;
	delete[] inputPFPCGPUPitch;
	delete[] inputSumPFPCMZGPU;

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

	//bc
	cudaSetDevice(gpuIndStart);
	cudaFreeHost(inputSumPFBCH);

	cudaDeviceSynchronize();
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		
		cudaFree(inputPFBCGPU[i]);
		cudaFree(inputSumPFBCGPU[i]);

		cudaDeviceSynchronize();
	}
	
	delete[] inputPFBCGPU;
	delete[] inputPFBCGPUP;
	delete[] inputSumPFBCGPU;

	LOG_DEBUG("Finished deleting mzone gpu arrays.");
}

void MZone::initCUDA(cudaStream_t **stream)
{
	int maxNumGPUs;
	cudaGetDeviceCount(&maxNumGPUs);

	numGRPerGPU = num_gr / numGPUs;

	updatePFPCNumGRPerB = 512;
	updatePFPCNumBlocks = numGRPerGPU / updatePFPCNumGRPerB;

	updatePFPCSynWNumGRPerB = 512 * (num_p_pc_from_gr_to_pc > 512) +
			num_p_pc_from_gr_to_pc * (num_p_pc_from_gr_to_pc <= 512);
	updatePFPCSynWNumBlocks = num_p_pc_from_gr_to_pc / updatePFPCSynWNumGRPerB;

	updatePFBCSCNumGRPerB = 512;
	updatePFBCSCNumBlocks = numGRPerGPU / updatePFBCSCNumGRPerB;

	// set up rng
	LOG_DEBUG("Initializing curand state...");

	CRandomSFMT cudaRNGSeedGen(time(0));

	int32_t curandInitSeed = cudaRNGSeedGen.IRandom(0, INT_MAX);

	mrg32k3aRNGs = new curandStateMRG32k3a*[numGPUs];
	pfpcSynWRandNums = new float*[numGPUs];

	dim3 updatePFPCSynWGridDim(updatePFPCSynWNumBlocks);
	dim3 updatePFPCSynWBlockDim(updatePFPCSynWNumGRPerB);  

	for (uint8_t i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		// FIXME: instead of set seed, bring in a rand gen seed
		// FIXME: either need to access the threads for which
		//        mrg32k3aRNGs have been allocated by considering
		//        an offset in the plasticity kernel, or create
		//        enough rngs to cover that entire space
		cudaMalloc((void **)&mrg32k3aRNGs[i],
				updatePFPCSynWNumGRPerB * updatePFPCSynWNumBlocks * sizeof(curandStateMRG32k3a));
		callCurandSetupKernel<curandStateMRG32k3a, dim3, dim3>
		(stream[i][1], mrg32k3aRNGs[i], (curandInitSeed + (uint32_t)i) % UINT_MAX, updatePFPCSynWGridDim, updatePFPCSynWBlockDim);
		cudaMalloc((void **)&pfpcSynWRandNums[i], numGRPerGPU * sizeof(float));
		cudaDeviceSynchronize();
	}
	LOG_DEBUG("Finished initializing curand state.");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

	/* ======== not used ====== */
	updateGRBCOutNumGRPerR=512*(num_bc>512)+num_bc*(num_bc<=512);
	updateGRBCOutNumGRRows=numGRPerGPU/updateGRBCOutNumGRPerR;

	sumGRBCOutNumBCPerB=1024*(num_bc>1024)+num_bc*(num_bc<=1024);
	sumGRBCOutNumBlocks=num_bc/sumGRBCOutNumBCPerB;
	/* ======== not used ====== */

	cudaSetDevice(0 + gpuIndStart);
	//allocate host cuda memory
	cudaHostAlloc((void **)&inputSumPFPCMZH, num_pc * sizeof(float), cudaHostAllocPortable);

	cudaDeviceSynchronize();
	//initialize host cuda memory
	for (int i = 0; i < num_pc; i++)
	{
		inputSumPFPCMZH[i] = 0;
	}
	
	for (int i = 0; i < num_pc; i++)
	{
		for (int j = 0; j < num_p_pc_from_gr_to_pc; j++)
		{
			// TODO: get rid of pfSynWeightLinear and use our linearized version directly
			pfSynWeightPCLinear[i * num_p_pc_from_gr_to_pc + j] = as->pfSynWeightPC[i * num_p_pc_from_gr_to_pc + j];
			pfPCSynWeightStatesLinear[i * num_p_pc_from_gr_to_pc + j] = as->pfPCSynWeightStates[i * num_p_pc_from_gr_to_pc + j];
		}
	}

	pfSynWeightPCGPU = new float*[numGPUs];
	pfPCSynWeightStatesGPU = new uint8_t*[numGPUs];
	inputPFPCGPU = new float*[numGPUs];
	inputPFPCGPUPitch = new size_t[numGPUs];
	inputSumPFPCMZGPU = new float*[numGPUs];

	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd = i * numGRPerGPU;
		int cpySize     = numGRPerGPU;
		cudaSetDevice(i + gpuIndStart);

		// conduction delay variables
		cudaMalloc((void **)&delayMaskGRGPU[i], numGRPerGPU * sizeof(uint32_t));
		// TODO: put the delay mask info into mzoneconnectivitystate
		cudaMemcpy(delayMaskGRGPU[i], &(cs->pGRDelayMaskfromGRtoBSP[cpyStartInd]),
			cpySize * sizeof(uint32_t), cudaMemcpyHostToDevice);

		//allocate device cuda memory
		cudaMalloc((void **)&pfSynWeightPCGPU[i], numGRPerGPU * sizeof(float));
		cudaMalloc((void **)&pfPCSynWeightStatesGPU[i], numGRPerGPU * sizeof(float));
		cudaMallocPitch((void **)&inputPFPCGPU[i], (size_t *)&inputPFPCGPUPitch[i],
				num_p_pc_from_gr_to_pc * sizeof(float), num_pc / numGPUs);
		cudaMalloc((void **)&inputSumPFPCMZGPU[i], num_pc / numGPUs * sizeof(float));

		cudaDeviceSynchronize();
		//initialize device cuda memory
		cudaMemcpy(pfSynWeightPCGPU[i], &pfSynWeightPCLinear[cpyStartInd],
				numGRPerGPU*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(pfPCSynWeightStatesGPU[i], &pfPCSynWeightStatesLinear[cpyStartInd],
				numGRPerGPU*sizeof(uint8_t), cudaMemcpyHostToDevice);

		for (int j = 0; j < num_pc/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFPCGPU[i] + j * inputPFPCGPUPitch[i]),
					0, num_p_pc_from_gr_to_pc * sizeof(float));
		}
		cudaMemset(inputSumPFPCMZGPU[i], 0, num_pc / numGPUs * sizeof(float));

		cudaDeviceSynchronize();
	}
	initBCCUDA();
	LOG_DEBUG("Initialized BC CUDA");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

	initSCCUDA();
	LOG_DEBUG("Initialized SC CUDA");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));
	
	testReduction();
	LOG_DEBUG("Finished Test.");
}

void MZone::initBCCUDA()
{
	
	inputPFBCGPU    = new uint32_t*[numGPUs];
	inputPFBCGPUP   = new size_t[numGPUs];
	inputSumPFBCGPU = new uint32_t*[numGPUs];

	//allocate host memory
	LOG_DEBUG("Allocating BC cuda variables...");
	cudaSetDevice(gpuIndStart);
	cudaHostAlloc((void **)&inputSumPFBCH, num_bc*sizeof(uint32_t), cudaHostAllocPortable);
	cudaMemset(inputSumPFBCH, 0, num_bc * sizeof(uint32_t));

	cudaDeviceSynchronize();

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaMallocPitch((void **)&inputPFBCGPU[i], (size_t *)&inputPFBCGPUP[i],
			num_p_bc_from_gr_to_bc * sizeof(uint32_t), num_bc / numGPUs);
		cudaMalloc((void **)&inputSumPFBCGPU[i], num_bc / numGPUs * sizeof(uint32_t));
		cudaDeviceSynchronize();
	}		
	LOG_DEBUG("Finished BC variable cuda allocation");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

	// initialize BC vars
	LOG_DEBUG("Initializing BC cuda variables...");
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		for (int j = 0; j < num_bc / numGPUs; j++)
		{
			cudaMemset(((char *)inputPFBCGPU[i] + j * inputPFBCGPUP[i]), 0,
				num_p_bc_from_gr_to_bc * sizeof(uint32_t));
		}
		cudaMemset(inputSumPFBCGPU[i], 0, num_bc / numGPUs*sizeof(uint32_t));
		cudaDeviceSynchronize();
	}
	LOG_DEBUG("Finished initializing BC cuda variables...");
}

void MZone::initSCCUDA()
{
	inputPFSCGPU    = new uint32_t*[numGPUs];
	inputPFSCGPUP   = new size_t[numGPUs];
	inputSumPFSCGPU = new uint32_t*[numGPUs];

	//allocate host memory
	LOG_DEBUG("Allocating SC cuda variables...");
	cudaSetDevice(gpuIndStart);
	cudaHostAlloc((void **)&inputSumPFSCH, num_sc * sizeof(uint32_t), cudaHostAllocPortable);
	cudaMemset(inputSumPFSCH, 0, num_sc * sizeof(uint32_t));

	cudaDeviceSynchronize();

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMallocPitch((void **)&inputPFSCGPU[i], (size_t *)&inputPFSCGPUP[i],
				num_p_sc_from_gr_to_sc * sizeof(uint32_t), num_sc / numGPUs);
		cudaMalloc((void **)&inputSumPFSCGPU[i], num_sc / numGPUs * sizeof(uint32_t));

		cudaDeviceSynchronize();
	}
	LOG_DEBUG("Finished SC variable cuda allocation");
	LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

	// initialize SC vars
	LOG_DEBUG("Initializing SC cuda variables...");
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		for(int j =0; j < num_sc / numGPUs; j++)
		{
			cudaMemset(((char *)inputPFSCGPU[i] + j * inputPFSCGPUP[i]), 0,
					num_p_sc_from_gr_to_sc * sizeof(uint32_t));
		}
		cudaMemset(inputSumPFSCGPU[i], 0, num_sc / numGPUs * sizeof(uint32_t));

		cudaDeviceSynchronize();
	}
	LOG_DEBUG("Finished initializing SC cuda variables...");
}

void MZone::writeToState()
{
	// TODO: write everything to state...only doing weights and pfpc input sums :/
	cpyPFPCSynWCUDA();

	for (int i = 0; i < num_pc; i++)
	{
		as->inputSumPFPC[i] = inputSumPFPCMZH[i];
	}
}

void MZone::cpyPFPCSynWCUDA()
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemcpy((void *)&pfSynWeightPCLinear[i*numGRPerGPU], pfSynWeightPCGPU[i],
			numGRPerGPU * sizeof(float), cudaMemcpyDeviceToHost);
	}

	for (int i = 0; i < num_pc; i++)
	{
		for (int j = 0; j < num_p_pc_from_gr_to_pc; j++)
		{
			as->pfSynWeightPC[i * num_p_pc_from_gr_to_pc + j] = pfSynWeightPCLinear[i * num_p_pc_from_gr_to_pc + j];
		}
	}
}

void MZone::setErrDrive(float errDriveRelative)
{
	as->errDrive = errDriveRelative * maxExtIncVIO;
}

void MZone::updateMFActivities(const uint8_t *actMF)
{
	apMFInput = actMF;
}

void MZone::updateTrueMFs(bool *trueMF)
{
	isTrueMF = trueMF;
}

void MZone::calcPCActivities()
{
	for (int i = 0; i < num_pc; i++)
	{
		as->gPFPC[i] += inputSumPFPCMZH[i] * gIncGRtoPC;
		as->gPFPC[i] *= gDecGRtoPC;
		as->gBCPC[i] += as->inputBCPC[i] * gIncBCtoPC;
		as->gBCPC[i] *= gDecBCtoPC;
		as->gSCPC[i] += as->inputSCPC[i] * gIncSCtoPC;
		as->gSCPC[i] *= gDecSCtoPC;

		as->vPC[i] += gLeakPC * (eLeakPC - as->vPC[i])
					- as->gPFPC[i] * as->vPC[i]
					+ as->gBCPC[i] * (eBCtoPC - as->vPC[i])
					+ as->gSCPC[i] * (eSCtoPC - as->vPC[i]);
		as->threshPC[i] += threshDecPC * (threshRestPC - as->threshPC[i]);

		as->apPC[i]    = as->vPC[i] > as->threshPC[i];
		as->apBufPC[i] = (as->apBufPC[i] << 1) | (as->apPC[i] * 0x00000001);

		as->threshPC[i] = as->apPC[i] * threshMaxPC + (1-as->apPC[i]) * as->threshPC[i];
		as->pcPopAct   += as->apPC[i]; 
	}
}

void MZone::calcSCActivities()
{
	for (int i = 0; i < num_sc; i++)
	{
		as->gPFSC[i] += inputSumPFSCH[i] * gIncGRtoSC;
		as->gPFSC[i] *= gDecGRtoSC;

		as->vSC[i] += gLeakSC * (eLeakSC - as->vSC[i]) - as->gPFSC[i] * as->vSC[i];
		as->threshSC[i] += threshDecSC * (threshRestSC - as->threshSC[i]);

		as->apSC[i] = as->vSC[i] > as->threshSC[i];
		as->apBufSC[i] = (as->apBufSC[i] << 1) | (as->apSC[i] * 0x00000001);

		as->threshSC[i] = as->apSC[i] * threshMaxSC + (1-as->apSC[i]) * as->threshSC[i];
	}
}

void MZone::calcBCActivities()
{
	for (int i = 0; i < num_bc; i++)
	{
		as->gPFBC[i] += inputSumPFBCH[i] * gIncGRtoBC;
		as->gPFBC[i] *= gDecGRtoBC;
		as->gPCBC[i] += as->inputPCBC[i] * gIncPCtoBC;
		as->gPCBC[i] *= gDecPCtoBC;

		as->vBC[i] = as->vBC[i] +
				(gLeakBC * (eLeakBC - as->vBC[i])) -
				(as->gPFBC[i] * as->vBC[i]) +
				(as->gPCBC[i] * (ePCtoBC - as->vBC[i]));
		as->threshBC[i] += threshDecBC * (threshRestBC - as->threshBC[i]);

		as->apBC[i] = as->vBC[i] > as->threshBC[i];
		as->apBufBC[i] = (as->apBufBC[i] << 1) | (as->apBC[i] * 0x00000001);

		as->threshBC[i] = as->apBC[i] * threshMaxBC + (1-as->apBC[i]) * as->threshBC[i];
	}
}

void MZone::calcIOActivities()
{
	srand(clock());
	
	float r = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
	float gNoise = (r - 0.5) * 2.0;

	for (int i = 0; i < num_io; i++)
	{
		float gNCSum;
		gNCSum = 0;

		for (int j = 0; j < num_p_io_from_nc_to_io; j++)
		{
			as->gNCIO[i * num_p_io_from_nc_to_io + j] = as->gNCIO[i * num_p_io_from_nc_to_io + j]
			   * exp(-msPerTimeStep /
				(-gDecTSofNCtoIO * exp(-as->gNCIO[i * num_p_io_from_nc_to_io + j] / gDecTTofNCtoIO)
				 + gDecT0ofNCtoIO));
			as->gNCIO[i * num_p_io_from_nc_to_io + j] = as->gNCIO[i * num_p_io_from_nc_to_io + j]
			   + as->inputNCIO[i * num_p_io_from_nc_to_io + j]
			   * gIncNCtoIO * exp(-as->gNCIO[i * num_p_io_from_nc_to_io + j] / gIncTauNCtoIO);
			gNCSum += as->gNCIO[i * num_p_io_from_nc_to_io + j];

			as->inputNCIO[i * num_p_io_from_nc_to_io + j] = 0;
		}
		gNCSum = 1.5 * gNCSum / 3.1;

		as->vIO[i] += gLeakIO * (eLeakIO - as->vIO[i]) +
				gNCSum * (eNCtoIO - as->vIO[i]) + as->vCoupleIO[i] +
				as->errDrive + gNoise;
		as->threshIO[i] += threshDecIO * (threshRestIO - as->threshIO[i]);

		as->apIO[i] = as->vIO[i] > as->threshIO[i];
		as->apBufIO[i] = (as->apBufIO[i] << 1) | (as->apIO[i] * 0x00000001);

		as->threshIO[i] = as->apIO[i] * threshMaxIO + (1-as->apIO[i]) * as->threshIO[i];
	}
	as->errDrive = 0;
}

void MZone::calcNCActivities()
{
	float gDecay = exp(-1.0 / 20.0); 

	for (int i = 0; i < num_nc; i++)
	{
		float gMFNMDASum = 0;
		float gMFAMPASum = 0;
		float gPCNCSum   = 0;

		int inputPCNCSum = 0;
		int inputMFNCSum = 0;


		for (int j = 0; j < num_p_nc_from_mf_to_nc; j++)
		{
			inputMFNCSum += as->inputMFNC[i * num_p_nc_from_mf_to_nc + j]; /* dont use */

			as->gMFAMPANC[i * num_p_nc_from_mf_to_nc + j] = as->gMFAMPANC[i * num_p_nc_from_mf_to_nc + j]
			   * gDecay + (gAMPAIncMFtoNC * as->inputMFNC[i * num_p_nc_from_mf_to_nc + j]
				 * as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j]);
			gMFAMPASum += as->gMFAMPANC[i * num_p_nc_from_mf_to_nc + j];
		}

		gMFNMDASum *= msPerTimeStep / ((float)num_p_nc_from_mf_to_nc);
		gMFNMDASum *= -as->vNC[i] / 80.0; 

		gMFAMPASum *= msPerTimeStep / ((float)num_p_nc_from_mf_to_nc);

		for (int j = 0; j < num_p_nc_from_pc_to_nc; j++)
		{
			inputPCNCSum += as->inputPCNC[i * num_p_nc_from_pc_to_nc + j];

			as->gPCNC[i * num_p_nc_from_pc_to_nc + j] = as->gPCNC[i * num_p_nc_from_pc_to_nc + j] * gDecPCtoNC + 
				as->inputPCNC[i * num_p_nc_from_pc_to_nc + j] * gIncAvgPCtoNC
				* (1 - as->gPCNC[i * num_p_nc_from_pc_to_nc + j]);
			gPCNCSum += as->gPCNC[i * num_p_nc_from_pc_to_nc + j];

		}

		gPCNCSum *= msPerTimeStep / ((float)num_p_nc_from_pc_to_nc);

		as->vNC[i] += gLeakNC * (eLeakNC - as->vNC[i])
				   - (gMFNMDASum + gMFAMPASum) * as->vNC[i]
				   + gPCNCSum * (ePCtoNC - as->vNC[i]);
		as->threshNC[i] += threshDecNC * (threshRestNC - as->threshNC[i]);

		as->apNC[i] = as->vNC[i] > as->threshNC[i];
		as->apBufNC[i] = (as->apBufNC[i] << 1) | (as->apNC[i] * 0x00000001);

		as->threshNC[i] = as->apNC[i] * threshMaxNC + (1-as->apNC[i]) * as->threshNC[i];
	}
}

void MZone::updatePCOut()
{
	for (int i = 0; i < num_bc; i++) as->inputPCBC[i] = 0;

	for (int i = 0; i < num_pc; i++)
	{
		for (int j = 0; j < num_p_pc_from_pc_to_bc; j++)
		{
			as->inputPCBC[cs->pPCfromPCtoBC[i][j]] += as->apPC[i];
		}
	}

	for (int i = 0; i < num_nc; i++)
	{
		for (int j = 0; j < num_p_nc_from_pc_to_nc; j++)
		{
			as->inputPCNC[i * num_p_nc_from_pc_to_nc + j] = as->apPC[cs->pNCfromPCtoNC[i][j]];
		}
	}
}

void MZone::updateBCPCOut()
{
	for (int i = 0; i < num_pc; i++) as->inputBCPC[i] = 0;
	
	for (int i = 0; i < num_bc; i++)
	{
		if (as->apBC[i]) 
		{
			for (int j = 0; j < num_p_bc_from_bc_to_pc; j++)
			{
				/* if there is a bc spike, obtain the pcs that this connects with
				 * and increment the input that the pc gets 
				 */
				as->inputBCPC[cs->pBCfromBCtoPC[i][j]]++;
			}
		}
	}
}

void MZone::updateSCPCOut()
{
	for (int i = 0; i < num_pc; i++) as->inputSCPC[i] = 0;

	for (int i = 0; i < num_sc; i++)
	{
		if (as->apSC[i])
		{
			for (int j = 0; j < num_p_sc_from_sc_to_pc; j++)
			{
				as->inputSCPC[cs->pSCfromSCtoPC[i][j]]++;
			}
		}
	}
}

void MZone::updateIOOut()
{
	for (int i = 0; i < num_io; i++)
	{
		as->pfPCPlastTimerIO[i] = (1 - as->apIO[i]) * (as->pfPCPlastTimerIO[i] + 1) + as->apIO[i] * tsLTPEndAPIO;
		for (int j = 0; j < num_p_io_in_io_to_io; j++)
			as->vCoupleIO[i] = coupleRiRjRatioIO * (as->vIO[cs->pIOInIOIO[i][j]] - as->vIO[i]);
	}
}

void MZone::updateNCOut()
{
	for (int i = 0; i < num_nc; i++)
	{
		as->synIOPReleaseNC[i] *= exp(-msPerTimeStep / 
				(relPDecTSofNCtoIO * exp(-as->synIOPReleaseNC[i] / relPDecTTofNCtoIO) +
				 relPDecT0ofNCtoIO));
		as->synIOPReleaseNC[i] += as->apNC[i] * relPIncNCtoIO *
				exp(-as->synIOPReleaseNC[i] / relPIncTauNCtoIO);
	}

	for (int i = 0; i < num_io; i++)
	{
		for (int j = 0; j < num_p_io_from_nc_to_io; j++)
		{
			as->inputNCIO[i * num_p_io_from_nc_to_io + j] = (randGen->Random() < as->synIOPReleaseNC[cs->pIOfromNCtoIO[i][j]]);
		}
	}
}

void MZone::updateMFNCOut()
{
	for (int i = 0; i < num_nc; i++)
	{
		for (int j = 0; j < num_p_nc_from_mf_to_nc; j++)
		{
			as->inputMFNC[i * num_p_nc_from_mf_to_nc + j] = apMFInput[cs->pNCfromMFtoNC[i][j]];
		}
	}
}

void MZone::updateMFNCSyn(const uint8_t *histMF, uint32_t t)
{
	if (t % (uint32_t)tsPerPopHistBinPC == 0) return;

	as->histPCPopActSum -= (as->histPCPopAct[as->histPCPopActCurBinN]) + (as->pcPopAct);
	as->histPCPopAct[as->histPCPopActCurBinN] = as->pcPopAct;
	as->pcPopAct = 0;
	as->histPCPopActCurBinN++;
	as->histPCPopActCurBinN %= (uint32_t)numPopHistBinsPC;

	float avgAllAPPC = ((float)as->histPCPopActSum) / numPopHistBinsPC;

	bool doLTD = false;
	bool doLTP = false;
	if (avgAllAPPC >= synLTDPCPopActThreshMFtoNC && !as->noLTDMFNC)
	{
		doLTD = true;
		as->noLTDMFNC = true;
	}
	else if (avgAllAPPC < synLTDPCPopActThreshMFtoNC)
	{
		as->noLTDMFNC = false;
	}

	if (avgAllAPPC <= synLTPPCPopActThreshMFtoNC && !as->noLTPMFNC)
	{
		doLTP = true;
		as->noLTPMFNC = true;
	}
	else if (avgAllAPPC > synLTPPCPopActThreshMFtoNC)
	{
		as->noLTPMFNC = false;
	}

	for (int i = 0; i < num_nc; i++)
	{
		for(int j = 0; j < num_p_nc_from_mf_to_nc; j++)
		{
			float synWDelta = histMF[cs->pNCfromMFtoNC[i][j]] * (doLTD * synLTDStepSizeMFtoNC +
					doLTP * synLTPStepSizeMFtoNC);
			as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j] += synWDelta;
			as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j] *= as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j] > 0;
			as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j] *= as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j] <= 1; 
			as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j] += as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j] > 1;
			
			//Now uses isTrueMF to take collaterals into account
			as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j] *= isTrueMF[cs->pNCfromMFtoNC[i][j]];
		}
	}
}

void MZone::runPFPCOutCUDA(cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callUpdatePFPCOutKernel(sts[i][streamN], updatePFPCNumBlocks, updatePFPCNumGRPerB,
				apBufGRGPU[i], delayMaskGRGPU[i], pfSynWeightPCGPU[i], inputPFPCGPU[i],
				inputPFPCGPUPitch[i], num_p_pc_from_gr_to_pc_p2);
	}
}

void MZone::runPFPCSumCUDA(cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callSumKernel<float, true, false>(sts[i][streamN], inputPFPCGPU[i], inputPFPCGPUPitch[i],
				inputSumPFPCMZGPU[i], 1, num_pc / numGPUs, 1, num_p_pc_from_gr_to_pc);
	}
}

void MZone::cpyPFPCSumCUDA(cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemcpyAsync(&inputSumPFPCMZH[num_pc * i / numGPUs], inputSumPFPCMZGPU[i],
				num_pc / numGPUs * sizeof(float), cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void MZone::runPFPCBinaryPlastCUDA(cudaStream_t **sts, int streamN, uint32_t t)
{
	if (t % (uint32_t)tsPerHistBinGR == 0)
	{
		int curGROffset = 0;
		int curGPUInd = 0;
		int curIOInd = 0;

		float transition_prob[num_io] = {0.0};
		float weight_diff = binPlastWeightHigh - binPlastWeightLow;

		int numGRPerIO = num_gr / num_io;
		for (int i = 0; i < num_io; i++)
		{
			if (as->pfPCPlastTimerIO[i] < (tsLTDstartAPIO + (int)tsLTDDurationIO) &&
					as->pfPCPlastTimerIO[i] >= tsLTDstartAPIO)
			{
				pfPCPlastStepIO[i] = -weight_diff;
				transition_prob[i] = binPlastProbMin;
			}
			else if (as->pfPCPlastTimerIO[i] >= tsLTPstartAPIO ||
					as->pfPCPlastTimerIO[i] < tsLTPEndAPIO)
			{
				pfPCPlastStepIO[i] = weight_diff;
				transition_prob[i] = binPlastProbMax;
			}
		}
		cudaError_t error = cudaSetDevice(curGPUInd + gpuIndStart);
		for (int i = 0; i < num_gr; i += num_p_pc_from_gr_to_pc)
		{
			if (i >= (curGPUInd + 1) * numGRPerGPU)
			{
				curGPUInd++;
				curGROffset = 0;
				error = cudaSetDevice(curGPUInd+gpuIndStart);
			}
			if (i >= (curIOInd + 1) * numGRPerIO)
			{
				curIOInd++;
			}
			callCurandGenerateUniformKernel<curandStateMRG32k3a>(sts[curGPUInd][streamN + curIOInd],
				  mrg32k3aRNGs[curGPUInd], updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB,
				  pfpcSynWRandNums[curGPUInd], curGROffset);

			callPFPCBinaryPlastKernel<curandStateMRG32k3a>(sts[curGPUInd][streamN + curIOInd],
					updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB, pfSynWeightPCGPU[curGPUInd],
					histGRGPU[curGPUInd], grPCHistCheckBinIO, curGROffset, pfPCPlastStepIO[curIOInd],
					binPlastWeightLow, binPlastWeightHigh, transition_prob[curIOInd], pfpcSynWRandNums[curGPUInd]);

			curGROffset += num_p_pc_from_gr_to_pc;
		}
	}
}

void MZone::runPFPCAbbottCascadePlastCUDA(cudaStream_t **sts, int streamN, uint32_t t)
{
	cudaError_t error;
	if (t % (uint32_t)tsPerHistBinGR == 0)
	{
		int numGRPerIO = num_gr / num_io;
		for (int i = 0; i < num_io; i++)
		{
			if (as->pfPCPlastTimerIO[i] < (tsLTDstartAPIO + (int)tsLTDDurationIO) &&
				as->pfPCPlastTimerIO[i] >= tsLTDstartAPIO)
			{
				int curGROffset = 0;
				int curGPUInd = 0;
				int curIOInd = 0;

				error = cudaSetDevice(curGPUInd + gpuIndStart);
				for (int i = 0; i < num_gr; i += num_p_pc_from_gr_to_pc)
				{
					if (i >= (curGPUInd + 1) * numGRPerGPU)
					{
						curGPUInd++;
						curGROffset = 0;
						error = cudaSetDevice(curGPUInd+gpuIndStart);
					}
					if (i >= (curIOInd + 1) * numGRPerIO)
					{
						curIOInd++;
					}
					callPFPCAbbottCascadeLTDPlastKernel<curandStateMRG32k3a>(sts[curGPUInd][streamN + curIOInd],
							updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB, pfSynWeightPCGPU[curGPUInd],
							pfPCSynWeightStatesGPU[curGPUInd], histGRGPU[curGPUInd], grPCHistCheckBinIO, curGROffset,
							cascPlastWeightLow, cascPlastProbMin, pfpcSynWRandNums[curGPUInd]);

					curGROffset += num_p_pc_from_gr_to_pc;
				}
			}
			else if (as->pfPCPlastTimerIO[i] >= tsLTPstartAPIO ||
					as->pfPCPlastTimerIO[i] < tsLTPEndAPIO)
			{
				int curGROffset = 0;
				int curGPUInd = 0;
				int curIOInd = 0;

				error = cudaSetDevice(curGPUInd + gpuIndStart);
				for (int i = 0; i < num_gr; i += num_p_pc_from_gr_to_pc)
				{
					if (i >= (curGPUInd + 1) * numGRPerGPU)
					{
						curGPUInd++;
						curGROffset = 0;
						error = cudaSetDevice(curGPUInd+gpuIndStart);
					}
					if (i >= (curIOInd + 1) * numGRPerIO)
					{
						curIOInd++;
					}
					callPFPCAbbottCascadeLTPPlastKernel<curandStateMRG32k3a>(sts[curGPUInd][streamN + curIOInd],
							updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB, pfSynWeightPCGPU[curGPUInd],
							pfPCSynWeightStatesGPU[curGPUInd], histGRGPU[curGPUInd], grPCHistCheckBinIO, curGROffset,
							cascPlastWeightHigh, cascPlastProbMax, pfpcSynWRandNums[curGPUInd]);

					curGROffset += num_p_pc_from_gr_to_pc;
				}
			}
		}
	}
}

void MZone::runPFPCMaukCascadePlastCUDA(cudaStream_t **sts, int streamN, uint32_t t)
{
	cudaError_t error;
	if (t % (uint32_t)tsPerHistBinGR == 0)
	{
		int numGRPerIO = num_gr / num_io;
		for (int i = 0; i < num_io; i++)
		{
			if (as->pfPCPlastTimerIO[i] < (tsLTDstartAPIO + (int)tsLTDDurationIO) &&
				as->pfPCPlastTimerIO[i] >= tsLTDstartAPIO)
			{
				int curGROffset = 0;
				int curGPUInd = 0;
				int curIOInd = 0;

				error = cudaSetDevice(curGPUInd + gpuIndStart);
				for (int i = 0; i < num_gr; i += num_p_pc_from_gr_to_pc)
				{
					if (i >= (curGPUInd + 1) * numGRPerGPU)
					{
						curGPUInd++;
						curGROffset = 0;
						error = cudaSetDevice(curGPUInd+gpuIndStart);
					}
					if (i >= (curIOInd + 1) * numGRPerIO)
					{
						curIOInd++;
					}
					callPFPCMaukCascadeLTDPlastKernel<curandStateMRG32k3a>(sts[curGPUInd][streamN + curIOInd],
							updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB, pfSynWeightPCGPU[curGPUInd],
							pfPCSynWeightStatesGPU[curGPUInd], histGRGPU[curGPUInd], grPCHistCheckBinIO, curGROffset,
							cascPlastWeightLow, cascPlastProbMin, pfpcSynWRandNums[curGPUInd]);

					curGROffset += num_p_pc_from_gr_to_pc;
				}
			}
			else if (as->pfPCPlastTimerIO[i] >= tsLTPstartAPIO ||
					as->pfPCPlastTimerIO[i] < tsLTPEndAPIO)
			{
				int curGROffset = 0;
				int curGPUInd = 0;
				int curIOInd = 0;

				error = cudaSetDevice(curGPUInd + gpuIndStart);
				for (int i = 0; i < num_gr; i += num_p_pc_from_gr_to_pc)
				{
					if (i >= (curGPUInd + 1) * numGRPerGPU)
					{
						curGPUInd++;
						curGROffset = 0;
						error = cudaSetDevice(curGPUInd+gpuIndStart);
					}
					if (i >= (curIOInd + 1) * numGRPerIO)
					{
						curIOInd++;
					}
					callPFPCMaukCascadeLTPPlastKernel<curandStateMRG32k3a>(sts[curGPUInd][streamN + curIOInd],
							updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB, pfSynWeightPCGPU[curGPUInd],
							pfPCSynWeightStatesGPU[curGPUInd], histGRGPU[curGPUInd], grPCHistCheckBinIO, curGROffset,
							cascPlastWeightHigh, cascPlastProbMax, pfpcSynWRandNums[curGPUInd]);

					curGROffset += num_p_pc_from_gr_to_pc;
				}
			}
		}
	}

}


void MZone::runPFPCGradedPlastCUDA(cudaStream_t **sts, int streamN, uint32_t t)
{
	cudaError_t error;
	if (t % (uint32_t)tsPerHistBinGR == 0)
	{
		int curGROffset = 0;
		int curGPUInd = 0;
		int curIOInd = 0;

		int numGRPerIO = num_gr / num_io;

		for (int i = 0; i < num_io; i++)
		{
			if (as->pfPCPlastTimerIO[i] < (tsLTDstartAPIO + (int)tsLTDDurationIO) &&
					as->pfPCPlastTimerIO[i] >= tsLTDstartAPIO)
			{
				pfPCPlastStepIO[i] = synLTDStepSizeGRtoPC;
			}
			else if (as->pfPCPlastTimerIO[i] >= tsLTPstartAPIO ||
					as->pfPCPlastTimerIO[i] < tsLTPEndAPIO)
			{
				pfPCPlastStepIO[i] = synLTPStepSizeGRtoPC;
			}
			else // this branch is never reached: the two above are mutually exclusive and span all possibilities
			{
				pfPCPlastStepIO[i] = 0;
			}
		}

		error = cudaSetDevice(curGPUInd + gpuIndStart);
		for (int i = 0; i < num_gr; i += num_p_pc_from_gr_to_pc)
		{
			if (i >= (curGPUInd + 1) * numGRPerGPU)
			{
				curGPUInd++;
				curGROffset = 0;
				error = cudaSetDevice(curGPUInd+gpuIndStart);
			}
			if (i >= (curIOInd + 1) * numGRPerIO)
			{
				curIOInd++;
			}
			callPFPCGradedPlastKernel(sts[curGPUInd][streamN + curIOInd],
					updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB, pfSynWeightPCGPU[curGPUInd],
					histGRGPU[curGPUInd], grPCHistCheckBinIO, curGROffset, pfPCPlastStepIO[curIOInd]);

			curGROffset += num_p_pc_from_gr_to_pc;
		}
	}
}

void MZone::runSumPFSCCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callSumKernel<uint32_t, true, false>
		(sts[i][streamN], inputPFSCGPU[i], inputPFSCGPUP[i],
				inputSumPFSCGPU[i], 1, num_sc/numGPUs, 1, num_p_sc_from_gr_to_sc);
	}
}

void MZone::cpyPFSCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(&inputSumPFSCH[num_sc * i / numGPUs], inputSumPFSCGPU[i],
				num_sc / numGPUs * sizeof(uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void MZone::runUpdatePFBCSCOutCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callUpdatePFBCSCOutKernel(sts[i][streamN], updatePFBCSCNumBlocks, updatePFBCSCNumGRPerB,
				apBufGRGPU[i], delayMaskGRGPU[i],
				inputPFBCGPU[i], inputPFBCGPUP[i], num_p_bc_from_gr_to_bc_p2, 
				inputPFSCGPU[i], inputPFSCGPUP[i], num_p_sc_from_gr_to_sc_p2); 
	}
}

void MZone::runSumPFBCCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		callSumKernel<uint32_t, true, false>
		(sts[i][streamN], inputPFBCGPU[i], inputPFBCGPUP[i],
				inputSumPFBCGPU[i], 1, num_bc/numGPUs, 1, num_p_bc_from_gr_to_bc);
	}
}

void MZone::cpyPFBCSumGPUtoHostCUDA(cudaStream_t **sts, int streamN)
{
	cudaError_t error;
	for(int i=0; i<numGPUs; i++)
	{
		error=cudaSetDevice(i+gpuIndStart);
		error=cudaMemcpyAsync(&inputSumPFBCH[num_bc * i / numGPUs], inputSumPFBCGPU[i],
				num_bc / numGPUs * sizeof(uint32_t),
				cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

const float* MZone::exportPFPCWeights()
{
	cpyPFPCSynWCUDA();
	return (const float *)pfSynWeightPCLinear; 
}

const float* MZone::exportMFDCNWeights()
{
	return (const float *)as->mfSynWeightNC.get(); 
}

void MZone::load_pfpc_weights_from_file(std::fstream &in_file_buf)
{
	rawBytesRW((char *)pfSynWeightPCLinear, num_gr * sizeof(float), true, in_file_buf);
	
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		int cpyStartInd = i * numGRPerGPU;
		cudaMemcpy(pfSynWeightPCGPU[i], &pfSynWeightPCLinear[cpyStartInd],
				numGRPerGPU * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void MZone::load_mfdcn_weights_from_file(std::fstream &in_file_buf)
{
	rawBytesRW((char *)as->mfSynWeightNC.get(), num_nc * num_p_nc_from_mf_to_nc * sizeof(float), true, in_file_buf);
}

// Why not write one export function which takes in the thing you want to export?
const uint8_t* MZone::exportAPNC()
{
	return (const uint8_t *)as->apNC.get();
}

const uint8_t* MZone::exportAPSC()
{
	return (const uint8_t *)as->apSC.get();
}

const uint8_t* MZone::exportAPBC()
{
	return (const uint8_t *)as->apBC.get();
}

const uint8_t* MZone::exportAPPC()
{
	return (const uint8_t *)as->apPC.get();
}

const uint8_t* MZone::exportAPIO()
{
	return (const uint8_t *)as->apIO.get();
}

const float* MZone::exportgBCPC()
{
	return (const float *)as->gBCPC.get();
}

const float* MZone::exportgPFPC()
{
	return (const float *)as->gPFPC.get();
}

const float* MZone::exportVmBC()
{
	return (const float *)as->vBC.get();
}

const float* MZone::exportVmPC()
{
	return (const float *)as->vPC.get();
}

const float* MZone::exportVmNC()
{
	return (const float *)as->vNC.get();
}

const float* MZone::exportVmIO()
{
	return (const float *)as->vIO.get();
}

const unsigned int* MZone::exportAPBufBC()
{
	return (const unsigned int *)as->apBufBC.get();
}

const uint32_t* MZone::exportAPBufPC()
{
	return (const uint32_t *)as->apBufPC.get();
}

const uint8_t* MZone::exportAPBufIO()
{
	return (const uint8_t *)as->apBufIO.get();
}

const uint32_t* MZone::exportAPBufNC()
{
	return (const uint32_t *)as->apBufNC.get();
}

void MZone::testReduction()
{
	cudaError_t error;
	cudaStream_t *sts = new cudaStream_t[numGPUs];

	float hostTestData[num_gr] = {0.0};
	float hostPCSum[num_pc] = {0.0};
	float hostBCSum[num_bc] = {0.0};
	float hostSCSum[num_sc] = {0.0};

	float gpuToHostPCSum[num_pc] = {0.0};
	float gpuToHostBCSum[num_bc] = {0.0};
	float gpuToHostSCSum[num_sc] = {0.0};

	// leaving these dynamic for now as i do not understand cuda oof
	float **gpuPCTestData = new float*[numGPUs];
	float **gpuBCTestData = new float*[numGPUs];
	float **gpuSCTestData = new float*[numGPUs];

	size_t *gpuPCP = new size_t[numGPUs];
	size_t *gpuBCP = new size_t[numGPUs];
	size_t *gpuSCP = new size_t[numGPUs];

	float **gpuPCSum = new float*[numGPUs];
	float **gpuBCSum = new float*[numGPUs];
	float **gpuSCSum = new float*[numGPUs];

	for (int i = 0; i < num_gr; i++)
	{
		hostTestData[i] = randGen->Random();
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaStreamCreate(&sts[i]);

		cudaMallocPitch(&gpuPCTestData[i], &gpuPCP[i],
				num_p_pc_from_gr_to_pc * sizeof(float), num_pc / numGPUs);
		cudaMallocPitch(&gpuBCTestData[i], &gpuBCP[i],
				num_p_bc_from_gr_to_bc * sizeof(float), num_bc / numGPUs);
		cudaMallocPitch(&gpuSCTestData[i], &gpuSCP[i],
				num_p_sc_from_gr_to_sc*sizeof(float), num_sc / numGPUs);

		cudaMalloc(&gpuPCSum[i], num_pc / numGPUs * sizeof(float));
		cudaMalloc(&gpuBCSum[i], num_bc / numGPUs * sizeof(float));
		cudaMalloc(&gpuSCSum[i], num_sc / numGPUs * sizeof(float));

		LOG_DEBUG("Allocating memory for gpu %d", i);
		LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

		cudaDeviceSynchronize();
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		for (int j = 0; j < num_pc / numGPUs; j++)
		{
		   cudaMemcpy(((char *)gpuPCTestData[i] + j * gpuPCP[i]),
				 &hostTestData[i * numGRPerGPU + j * num_p_pc_from_gr_to_pc],
				 num_p_pc_from_gr_to_pc * sizeof(float), cudaMemcpyHostToDevice);
		}

		for (int j = 0; j < num_bc / numGPUs; j++)
		{
		   cudaMemcpy(((char *)gpuBCTestData[i] + j * gpuBCP[i]),
				 &hostTestData[i * numGRPerGPU + j * num_p_bc_from_gr_to_bc],
				 num_p_bc_from_gr_to_bc * sizeof(float), cudaMemcpyHostToDevice);
		}

		for (int j = 0; j < num_sc / numGPUs; j++)
		{
		   cudaMemcpy(((char *)gpuSCTestData[i] + j * gpuSCP[i]),
				 &hostTestData[i * numGRPerGPU + j * num_p_sc_from_gr_to_sc],
				 num_p_sc_from_gr_to_sc * sizeof(float), cudaMemcpyHostToDevice);
		}

		LOG_DEBUG("Copying memory for gpu %d", i);
		LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

		cudaDeviceSynchronize();
	}

	for (int i = 0; i < num_pc; i++)
	{
		hostPCSum[i] = 0;

		for (int j = 0; j < num_p_pc_from_gr_to_pc; j++)
		{
			hostPCSum[i] += hostTestData[i * num_p_pc_from_gr_to_pc + j];
		}
	}

	for (int i = 0; i < num_bc; i++)
	{
		hostBCSum[i] = 0;

		for (int j = 0; j < num_p_bc_from_gr_to_bc; j++)
		{
			hostBCSum[i] += hostTestData[i * num_p_bc_from_gr_to_bc + j];
		}
	}

	for (int i = 0; i < num_sc; i++)
	{
		hostSCSum[i] = 0;

		for (int j = 0; j < num_p_sc_from_gr_to_sc; j++)
		{
			hostSCSum[i] += hostTestData[i * num_p_sc_from_gr_to_sc + j];
		}
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callSumKernel<float, true, false>(sts[i], gpuPCTestData[i], gpuPCP[i],
				gpuPCSum[i], 1, num_pc / numGPUs, 1, num_p_pc_from_gr_to_pc);

		callSumKernel<float, true, false>(sts[i], gpuBCTestData[i], gpuBCP[i],
				gpuBCSum[i], 1, num_bc / numGPUs, 1, num_p_bc_from_gr_to_bc);

		callSumKernel<float, true, false>(sts[i], gpuSCTestData[i], gpuSCP[i],
				gpuSCSum[i], 1, num_sc / numGPUs, 1, num_p_sc_from_gr_to_sc);

		cudaDeviceSynchronize();

		LOG_DEBUG("Calling sum kernels for gpu %d", i);
		LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));
	}

	for (int i = 0; i<numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaMemcpy(&gpuToHostPCSum[i * num_pc / numGPUs], gpuPCSum[i],
			num_pc / numGPUs * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gpuToHostBCSum[i * num_bc / numGPUs], gpuBCSum[i],
			num_bc / numGPUs * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gpuToHostSCSum[i * num_sc / numGPUs], gpuSCSum[i],
			num_sc / numGPUs * sizeof(float), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
	}

	LOG_DEBUG("NumPC per GPU: %d", num_pc / numGPUs);
	LOG_DEBUG("NumBC per GPU: %d", num_bc / numGPUs);
	LOG_DEBUG("NumSC per GPU: %d", num_sc / numGPUs);

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i+gpuIndStart);
		cudaFree(gpuPCTestData[i]);
		cudaFree(gpuBCTestData[i]);
		cudaFree(gpuSCTestData[i]);
		cudaFree(gpuPCSum[i]);
		cudaFree(gpuBCSum[i]);
		cudaFree(gpuSCSum[i]);
		
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaStreamDestroy(sts[i]);
	}
	delete[] sts;

	delete[] gpuPCTestData;
	delete[] gpuBCTestData;
	delete[] gpuSCTestData;

	delete[] gpuPCP;
	delete[] gpuBCP;
	delete[] gpuSCP;

	delete[] gpuPCSum;
	delete[] gpuSCSum;
	delete[] gpuBCSum;
}

