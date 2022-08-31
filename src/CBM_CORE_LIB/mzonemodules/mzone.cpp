/*
 * mzone.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: consciousness
 */

#include "mzonemodules/mzone.h"

MZone::MZone() {}

MZone::MZone(MZoneConnectivityState *cs, MZoneActivityState *as, int randSeed, ct_uint32_t **actBufGRGPU,
		ct_uint32_t **delayMaskGRGPU, ct_uint64_t **histGRGPU, int gpuIndStart, int numGPUs)
{
	randGen = new CRandomSFMT0(randSeed);

	// as with innet, shallow-copying all (for now). Thus, important
	// that only the caller deletes these items AND that this object
	// goes out of scope before the caller
	this->cs = cs; 
	this->as = as; 

	// NOTE if we turn these guys into unique ptrs, we'll have to refactor
	// consider ownership: who should own these guys? maybe they should be global to both
	// innet and mzone (so within cbmsimcore) and fed in as const args to the respective
	// functions that call update kernels (06/16/2022)
	apBufGRGPU           = actBufGRGPU;
	delayBCPCSCMaskGRGPU = delayMaskGRGPU;
	historyGRGPU         = histGRGPU;

	pfSynWeightPCLinear = new float[num_gr];
	pfPCPlastStepIO     = new float[num_io];

	tempGRPCLTDStep = synLTDStepSizeGRtoPC;
	tempGRPCLTPStep = synLTPStepSizeGRtoPC;

	this->numGPUs     = numGPUs;
	this->gpuIndStart = gpuIndStart;

	std::cout << "Initializing CUDA..." << std::endl;
	initCUDA();
}

MZone::~MZone()
{
	std::cout << "[INFO] Deleting mzone gpu arrays..." << std::endl;
	//clean up allocated memory
	delete randGen;

	delete[] pfSynWeightPCLinear;
	delete[] pfPCPlastStepIO;

//	//free cuda host memory
	cudaSetDevice(0 + gpuIndStart);
	cudaFreeHost(inputSumPFPCMZH);
	cudaDeviceSynchronize();

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		//free cuda device memory
		cudaFree(pfSynWeightPCGPU[i]);
		cudaFree(inputPFPCGPU[i]);
		cudaFree(inputSumPFPCMZGPU[i]);
		cudaDeviceSynchronize();
	}

	delete[] pfSynWeightPCGPU;
	delete[] inputPFPCGPU;
	delete[] inputPFPCGPUPitch;
	delete[] inputSumPFPCMZGPU;
	std::cout << "[INFO] Finished deleting mzone gpu arrays." << std::endl;
}

void MZone::initCUDA()
{
	int maxNumGPUs;
	cudaGetDeviceCount(&maxNumGPUs);

	numGRPerGPU = num_gr / numGPUs;

	updatePFPCNumGRPerB = 512;
	updatePFPCNumBlocks = numGRPerGPU / updatePFPCNumGRPerB;

	updatePFPCSynWNumGRPerB = 512 * (num_p_pc_from_gr_to_pc > 512) +
			num_p_pc_from_gr_to_pc * (num_p_pc_from_gr_to_pc <= 512);
	updatePFPCSynWNumBlocks = num_p_pc_from_gr_to_pc / updatePFPCSynWNumGRPerB;

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
		}
	}

	pfSynWeightPCGPU = new float*[numGPUs];
	inputPFPCGPU = new float*[numGPUs];
	inputPFPCGPUPitch = new size_t[numGPUs];
	inputSumPFPCMZGPU = new float*[numGPUs];

	for (int i = 0; i < numGPUs; i++)
	{
		int cpyStartInd;

		cpyStartInd = i * numGRPerGPU;

		cudaSetDevice(i + gpuIndStart);
		//allocate device cuda memory
		cudaMalloc((void **)&pfSynWeightPCGPU[i], numGRPerGPU * sizeof(float));
		cudaMallocPitch((void **)&inputPFPCGPU[i], (size_t *)&inputPFPCGPUPitch[i],
				num_p_pc_from_gr_to_pc * sizeof(float), num_pc / numGPUs);
		cudaMalloc((void **)&inputSumPFPCMZGPU[i], num_pc / numGPUs * sizeof(float));

		cudaDeviceSynchronize();
		//initialize device cuda memory
		cudaMemcpy(pfSynWeightPCGPU[i], &pfSynWeightPCLinear[cpyStartInd],
				numGRPerGPU*sizeof(float), cudaMemcpyHostToDevice);

		for (int j = 0; j < num_pc/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFPCGPU[i] + j * inputPFPCGPUPitch[i]),
					0, num_p_pc_from_gr_to_pc * sizeof(float));
		}
		cudaMemset(inputSumPFPCMZGPU[i], 0, num_pc / numGPUs * sizeof(float));

		cudaDeviceSynchronize();
	}
	
	testReduction();
	std::cout << "Finished Test." << std::endl;
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

void MZone::updateMFActivities(const ct_uint8_t *actMF)
{
	apMFInput = actMF;
}

void MZone::updateTrueMFs(bool *trueMF)
{
	isTrueMF = trueMF;
}

void MZone::updateSCActivities(const ct_uint8_t *actSC)
{
	apSCInput = actSC;
}

void MZone::updatePFBCSum(const ct_uint32_t *pfBCSum)
{
	sumPFBCInput = pfBCSum;
}

void MZone::calcPCActivities()
{
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_pc; i++)
		{
			float gSCPCSum;

			as->gPFPC[i] = as->gPFPC[i] + inputSumPFPCMZH[i] * gIncGRtoPC;
			as->gPFPC[i] = as->gPFPC[i] * gDecGRtoPC;
			as->gBCPC[i] = as->gBCPC[i] + as->inputBCPC[i] * gIncBCtoPC;
			as->gBCPC[i] = as->gBCPC[i] * gDecBCtoPC;

			gSCPCSum = 0;

			for (int j = 0; j < num_p_pc_from_sc_to_pc; j++)
			{
				as->gSCPC[i * num_p_pc_from_sc_to_pc + j] = as->gSCPC[i * num_p_pc_from_sc_to_pc + j]
				   + gIncSCtoPC * (1 - as->gSCPC[i * num_p_pc_from_sc_to_pc + j])
				   * as->inputSCPC[i * num_p_pc_from_sc_to_pc + j];
				as->gSCPC[i * num_p_pc_from_sc_to_pc + j] = as->gSCPC[i * num_p_pc_from_sc_to_pc + j]
				   * gDecSCtoPC;
				gSCPCSum += as->gSCPC[i * num_p_pc_from_sc_to_pc + j];
			}

			as->vPC[i] = as->vPC[i] +
					(gLeakPC * (eLeakPC - as->vPC[i])) -
					(as->gPFPC[i] * as->vPC[i]) +
					(as->gBCPC[i] * (eBCtoPC - as->vPC[i])) +
					(gSCPCSum * (eSCtoPC - as->vPC[i]));

			as->threshPC[i] = as->threshPC[i] + (threshDecPC * (threshRestPC - as->threshPC[i]));

			as->apPC[i] = as->vPC[i] > as->threshPC[i];
			as->apBufPC[i] = (as->apBufPC[i] << 1) | (as->apPC[i] * 0x00000001);

			as->threshPC[i] = as->apPC[i] * threshMaxPC + (!as->apPC[i]) * as->threshPC[i];
			as->pcPopAct = as->pcPopAct + as->apPC[i];
		}
	}
}

void MZone::calcBCActivities(ct_uint32_t **pfInput)
{
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_bc; i++)
		{
			int totalPFInput = 0;

			for (int j = 0; j < numGPUs; j++)
			{
				totalPFInput += pfInput[j][i];
			}
			
			as->gPFBC[i] = as->gPFBC[i] + sumPFBCInput[i] * gIncGRtoBC;
			as->gPFBC[i] = as->gPFBC[i] * gDecGRtoBC;
			as->gPCBC[i] = as->gPCBC[i] + as->inputPCBC[i] * gIncPCtoBC;
			as->gPCBC[i] = as->gPCBC[i] * gDecPCtoBC;

			as->vBC[i] = as->vBC[i] +
					(gLeakBC * (eLeakBC - as->vBC[i])) -
					(as->gPFBC[i] * as->vBC[i]) +
					(as->gPCBC[i] * (ePCtoBC - as->vBC[i]));

			as->threshBC[i] = as->threshBC[i] + threshDecBC * (threshRestBC - as->threshBC[i]);
			as->apBC[i] = as->vBC[i] > as->threshBC[i];
			as->apBufBC[i] = (as->apBufBC[i] << 1) | (as->apBC[i] * 0x00000001);

			as->threshBC[i] = as->apBC[i] * threshMaxBC + (!as->apBC[i]) * (as->threshBC[i]);
		}
	}
}

void MZone::calcIOActivities()
{
#pragma omp parallel for
	clock_t t;
	t = clock();
	srand(t);
	
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

		as->vIO[i] = as->vIO[i] + gLeakIO * (eLeakIO - as->vIO[i]) +
				gNCSum * (eNCtoIO - as->vIO[i]) + as->vCoupleIO[i] +
				as->errDrive + gNoise;

		as->apIO[i] = as->vIO[i] > as->threshIO[i];
		as->apBufIO[i] = (as->apBufIO[i] << 1) |(as->apIO[i] * 0x00000001);

		as->threshIO[i] = threshMaxIO * as->apIO[i] +
				(!as->apIO[i]) * (as->threshIO[i] + threshDecIO * (threshRestIO - as->threshIO[i]));
	}
	as->errDrive = 0;
}

void MZone::calcNCActivities()
{
//TODO: make function calcActivities which takes in the parameters which vary dep
//		on the type of activity that is being calculated. Else this is redundant  
float gDecay = exp(-1.0 / 20.0); 

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < num_nc; i++)
		{
			float gMFNMDASum;
			float gMFAMPASum;
			float gPCNCSum;

			int inputPCNCSum;
			int inputMFNCSum;

			gMFNMDASum   = 0;
			gMFAMPASum   = 0;
			inputMFNCSum = 0;

			for (int j = 0; j < num_p_nc_from_mf_to_nc; j++)
			{
				inputMFNCSum += as->inputMFNC[i * num_p_nc_from_mf_to_nc + j];

				as->gMFAMPANC[i * num_p_nc_from_mf_to_nc + j] = as->gMFAMPANC[i * num_p_nc_from_mf_to_nc + j]
				   * gDecay + (gAMPAIncMFtoNC * as->inputMFNC[i * num_p_nc_from_mf_to_nc + j]
					 * as->mfSynWeightNC[i * num_p_nc_from_mf_to_nc + j]);
				gMFAMPASum += as->gMFAMPANC[i * num_p_nc_from_mf_to_nc + j];
			}

			gMFNMDASum = gMFNMDASum * msPerTimeStep / ((float)num_p_nc_from_mf_to_nc);
			gMFAMPASum = gMFAMPASum * msPerTimeStep / ((float)num_p_nc_from_mf_to_nc);
			gMFNMDASum = gMFNMDASum * -as->vNC[i] / 80.0f;

			gPCNCSum = 0;
			inputPCNCSum = 0;

			for (int j = 0; j < num_p_nc_from_pc_to_nc; j++)
			{
				inputPCNCSum += as->inputPCNC[i * num_p_nc_from_pc_to_nc + j];

				as->gPCNC[i * num_p_nc_from_pc_to_nc + j] = as->gPCNC[i * num_p_nc_from_pc_to_nc + j] * gDecPCtoNC + 
					as->inputPCNC[i * num_p_nc_from_pc_to_nc + j] * gIncAvgPCtoNC
					* (1 - as->gPCNC[i * num_p_nc_from_pc_to_nc + j]);
				gPCNCSum += as->gPCNC[i * num_p_nc_from_pc_to_nc + j];

			}
			gPCNCSum = gPCNCSum * msPerTimeStep / ((float)num_p_nc_from_pc_to_nc);
			
			as->vNC[i] = as->vNC[i] + gLeakNC * (eLeakNC - as->vNC[i]) -
					(gMFNMDASum + gMFAMPASum) * as->vNC[i] + gPCNCSum * (ePCtoNC - as->vNC[i]);
			
			as->threshNC[i] = as->threshNC[i] + threshDecNC * (threshRestNC - as->threshNC[i]);
			as->apNC[i] = as->vNC[i] > as->threshNC[i];
			as->apBufNC[i] = (as->apBufNC[i] << 1) | (as->apNC[i] * 0x00000001);

			as->threshNC[i] = as->apNC[i] * threshMaxNC + (!as->apNC[i]) * as->threshNC[i];
		}
	}
}

void MZone::updatePCOut()
{
#ifdef DEBUGOUT
	std::cout << "resetting inputPCBC " << num_bc << std::endl;
#endif
	for (int i = 0; i < num_bc; i++)
	{
		as->inputPCBC[i] = 0;
	}
#ifdef DEBUGOUT
	std::cout << "updating pc to bc " << std::endl;
#endif
	for (int i = 0; i < num_pc; i++)
	{
		for (int j = 0; j < num_p_pc_from_pc_to_bc; j++)
		{
#ifdef DEBUGOUT
			std::cout << "i: " << i << " j: " << j << std::endl;
#endif
			as->inputPCBC[cs->pPCfromPCtoBC[i][j]] += as->apPC[i];
		}
	}
#ifdef DEBUGOUT
	std::cout << "updating pc to nc " << std::endl;
#endif
	for (int i = 0; i < num_nc; i++)
	{
		for (int j = 0; j < num_p_nc_from_pc_to_nc; j++)
		{
#ifdef DEBUGOUT
			std::cout << "i: " << i << " j: " << j <<
				"cs->pNCfromPCtoNC[i][j]: " << cs->pNCfromPCtoNC[i][j] << std::endl;
#endif
			as->inputPCNC[i * num_p_nc_from_pc_to_nc + j] = as->apPC[cs->pNCfromPCtoNC[i][j]];
		}
	}
#ifdef DEBUGOUT
	std::cout << "finished " << std::endl;
#endif
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
				as->inputBCPC[cs->pBCfromBCtoPC[i][j]]++;
			}
		}
	}
}

void MZone::updateSCPCOut()
{
#pragma omp parallel for
	for (int i = 0; i < num_pc; i++)
	{
		for (int j = 0; j < num_p_pc_from_sc_to_pc; j++)
		{
			as->inputSCPC[i * num_p_pc_from_sc_to_pc + j] = apSCInput[cs->pPCfromSCtoPC[i][j]];
		}
	}
}

void MZone::updateIOOut()
{
	for (int i = 0; i < num_io; i++)
	{
		as->pfPCPlastTimerIO[i] = (!as->apIO[i]) * (as->pfPCPlastTimerIO[i] +1 ) + as->apIO[i] * tsLTPEndAPIO;
		as->vCoupleIO[i] = 0;
		for (int j = 0; j < num_p_io_in_io_to_io; j++)
		{
			as->vCoupleIO[i] += coupleRiRjRatioIO * (as->vIO[cs->pIOInIOIO[i][j]] - as->vIO[i]);
		}
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
/*
 * NOTE: it is okay that we are passing the unique ptr by value, since we only read from it, not write
 */
//void MZone::updateMFNCSyn(const std::unique_ptr<ct_uint8_t[]> histMF, unsigned long t)
//{
//	
//	bool reset;
//	float avgAllAPPC;
//	bool doLTD;
//	bool doLTP;
//
//#ifdef DEBUGOUT
//	float sumSynW;
//#endif
//	if(t % ap->tsPerPopHistBinPC == 0) return;
//
//	//histMFInput = histMF;
//
//	as->histPCPopActSum = (as->histPCPopActSum) - (as->histPCPopAct[as->histPCPopActCurBinN]) + (as->pcPopAct);
//	as->histPCPopAct[as->histPCPopActCurBinN] = as->pcPopAct;
//	as->pcPopAct = 0;
//	as->histPCPopActCurBinN++;
//	as->histPCPopActCurBinN %= ap->numPopHistBinsPC;
//
//	avgAllAPPC = ((float)as->histPCPopActSum) / ap->numPopHistBinsPC;
//
//#ifdef DEBUGOUT
//	std::cout << "avgAllAPPC: " << avgAllAPPC << std::endl;
//#endif
//
//	doLTD = false;
//	doLTP = false;
//	if (avgAllAPPC >= ap->synLTDPCPopActThreshMFtoNC && !as->noLTDMFNC)
//	{
//		doLTD = true;
//		as->noLTDMFNC = true;
//	}
//	else if (avgAllAPPC < ap->synLTDPCPopActThreshMFtoNC)
//	{
//		as->noLTDMFNC = false;
//	}
//
//	if (avgAllAPPC <= ap->synLTPPCPopActThreshMFtoNC && !as->noLTPMFNC)
//	{
//		doLTP = true;
//		as->noLTPMFNC = true;
//	}
//	else if (avgAllAPPC > ap->synLTPPCPopActThreshMFtoNC)
//	{
//		as->noLTPMFNC = false;
//	}
//
//#ifdef DEBUGOUT
//	sumSynW = 0;
//#endif
//	for (int i = 0; i < NUM_NC; i++)
//	{
//		for(int j = 0; j < num_p_nc_from_mf_to_nc; j++)
//		{
//			float synWDelta;
//			synWDelta = histMF[cs->pNCfromMFtoNC[i][j]] * (doLTD * ap->synLTDStepSizeMFtoNC +
//					doLTP * ap->synLTPStepSizeMFtoNC);
//			as->mfSynWeightNC[i][j] += synWDelta;
//			as->mfSynWeightNC[i][j] *= as->mfSynWeightNC[i][j] > 0;
//			as->mfSynWeightNC[i][j] *= as->mfSynWeightNC[i][j] <= 1; 
//			as->mfSynWeightNC[i][j] += as->mfSynWeightNC[i][j] > 1;
//			
//			//Now uses isTrueMF to take collaterals into account
//			as->mfSynWeightNC[i][j] *= isTrueMF[cs->pNCfromMFtoNC[i][j]];
//#ifdef DEBUGOUT
//			sumSynW += as->mfSynWeightNC[i][j];
//#endif
//		}
//	}
//#ifdef DEBUGOUT
//	std::cout << sumSynW / NUM_MF << std::endl;
//#endif
//}

void MZone::runPFPCOutCUDA(cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callUpdatePFPCOutKernel(sts[i][streamN], updatePFPCNumBlocks, updatePFPCNumGRPerB,
				apBufGRGPU[i], delayBCPCSCMaskGRGPU[i], pfSynWeightPCGPU[i], inputPFPCGPU[i],
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

void MZone::runPFPCPlastCUDA(cudaStream_t **sts, int streamN, unsigned long t)
{
	cudaError_t error;
	if (t % (unsigned long)tsPerHistBinGR == 0)
	{
		int curGROffset;
		int curGPUInd;
		int curIOInd;

		int numGRPerIO;

		curGROffset = 0;
		curGPUInd   = 0;
		curIOInd    = 0;

		numGRPerIO = num_gr / num_io;

		for (int i = 0; i < num_io; i++)
		{
			if (as->pfPCPlastTimerIO[i] < (tsLTDstartAPIO + (int)tsLTDDurationIO) &&
					as->pfPCPlastTimerIO[i] >= tsLTDstartAPIO)
			{
				pfPCPlastStepIO[i] = tempGRPCLTDStep;
			}
			else if (as->pfPCPlastTimerIO[i] >= tsLTPstartAPIO ||
					as->pfPCPlastTimerIO[i] < tsLTPEndAPIO)
			{
				pfPCPlastStepIO[i] = tempGRPCLTPStep;
			}
			else
			{
				pfPCPlastStepIO[i] = 0;
			}
		}

#ifdef DEBUGOUT
		std::cout << "pfPCPlastStepiO[0]: " << pfPCPlastStepIO[0] << " as->pfPCPlastTimerIO[0: ]" <<
			as->pfPCPlastTimerIO[0] << std::endl;
#endif
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
			callUpdatePFPCPlasticityIOKernel(sts[curGPUInd][streamN + curIOInd],
					updatePFPCSynWNumBlocks, updatePFPCSynWNumGRPerB, pfSynWeightPCGPU[curGPUInd],
					historyGRGPU[curGPUInd], grPCHistCheckBinIO, curGROffset, pfPCPlastStepIO[curIOInd]);

			curGROffset += num_p_pc_from_gr_to_pc;
		}
	}
}

void MZone::setGRPCPlastSteps(float ltdStep, float ltpStep)
{
	tempGRPCLTDStep = ltdStep;
	tempGRPCLTPStep = ltpStep;
}

void MZone::resetGRPCPlastSteps()
{
	tempGRPCLTDStep = synLTDStepSizeGRtoPC;
	tempGRPCLTPStep = synLTPStepSizeGRtoPC;
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

// Why not write one export function which takes in the thing you want to export?
const ct_uint8_t* MZone::exportAPNC()
{
	return (const ct_uint8_t *)as->apNC.get();
}

const ct_uint8_t* MZone::exportAPBC()
{
	return (const ct_uint8_t *)as->apBC.get();
}

const ct_uint8_t* MZone::exportAPPC()
{
	return (const ct_uint8_t *)as->apPC.get();
}

const ct_uint8_t* MZone::exportAPIO()
{
	return (const ct_uint8_t *)as->apIO.get();
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

const ct_uint32_t* MZone::exportAPBufPC()
{
	return (const ct_uint32_t *)as->apBufPC.get();
}

const ct_uint8_t* MZone::exportAPBufIO()
{
	return (const ct_uint8_t *)as->apBufIO.get();
}

const ct_uint32_t* MZone::exportAPBufNC()
{
	return (const ct_uint32_t *)as->apBufNC.get();
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

		error = cudaGetLastError();
		std::cout << "allocating memory for gpu " << i << " " << cudaGetErrorString(error) << std::endl;

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

		error = cudaGetLastError();
		std::cout << "copying memory for gpu " << i << " " << cudaGetErrorString(error) << std::endl;

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

		error = cudaGetLastError();
		std::cout << "calling sum kernels for gpu " << i << " " << cudaGetErrorString(error) << std::endl;
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

	std::cout << "NumPC per GPU: " << num_pc / numGPUs << std::endl <<
		"NumBC per GPU: " << num_bc / numGPUs << 
		"NUMSC per GPU: " << num_sc / numGPUs << std::endl;

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

