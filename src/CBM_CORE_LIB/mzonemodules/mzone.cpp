/*
 * mzone.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: consciousness
 */

#include "mzonemodules/mzone.h"

MZone::MZone(ConnectivityParams &cp, ActivityParams *ap,
		MZoneConnectivityState *conState, MZoneActivityState *actState,
		int randSeed, ct_uint32_t **actBufGRGPU,
		ct_uint32_t **delayMaskGRGPU, ct_uint64_t **histGRGPU,
		int gpuIndStart, int numGPUs)
{
	std::cout << "MZone constructor entered" << std::endl;
	
	randGen = new CRandomSFMT0(randSeed);

	//cp = conParams;
	//ap = actParams;
	cs = conState;
	as = actState;

	apBufGRGPU 			 = actBufGRGPU;
	delayBCPCSCMaskGRGPU = delayMaskGRGPU;
	historyGRGPU 		 = histGRGPU;

	pfSynWeightPCLinear = new float[cp.NUM_GR];
	pfPCPlastStepIO = new float[cp.NUM_IO];

	tempGRPCLTDStep = ap->synLTDStepSizeGRtoPC;
	tempGRPCLTPStep = ap->synLTPStepSizeGRtoPC;

	this->numGPUs 	  = numGPUs;
	this->gpuIndStart = gpuIndStart;

	std::cout << "Initializing CUDA..." << std::endl;
	initCUDA(cp);
}

MZone::~MZone()
{
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
}

void MZone::initCUDA(ConnectivityParams &cp)
{
	int maxNumGPUs;
	cudaGetDeviceCount(&maxNumGPUs);

	numGRPerGPU = cp.NUM_GR / numGPUs;

	updatePFPCNumGRPerB = 512;
	updatePFPCNumBlocks = numGRPerGPU / updatePFPCNumGRPerB;

	updatePFPCSynWNumGRPerB = 512 * (cp.NUM_P_PC_FROM_GR_TO_PC > 512) +
			cp.NUM_P_PC_FROM_GR_TO_PC * (cp.NUM_P_PC_FROM_GR_TO_PC <= 512);
	updatePFPCSynWNumBlocks = cp.NUM_P_PC_FROM_GR_TO_PC / updatePFPCSynWNumGRPerB;

	cudaSetDevice(0 + gpuIndStart);
	//allocate host cuda memory
	cudaHostAlloc((void **)&inputSumPFPCMZH, cp.NUM_PC * sizeof(float), cudaHostAllocPortable);

	cudaDeviceSynchronize();
	//initialize host cuda memory
	for (int i = 0; i < cp.NUM_PC; i++)
	{
		inputSumPFPCMZH[i] = 0;
	}

	for (int i = 0; i < cp.NUM_PC; i++)
	{
		for (int j = 0; j < cp.NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			pfSynWeightPCLinear[i * cp.NUM_P_PC_FROM_GR_TO_PC + j] = as->pfSynWeightPC[i][j];
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
				cp.NUM_P_PC_FROM_GR_TO_PC * sizeof(float), cp.NUM_PC / numGPUs);
		cudaMalloc((void **)&inputSumPFPCMZGPU[i], cp.NUM_PC / numGPUs * sizeof(float));

		cudaDeviceSynchronize();
		//initialize device cuda memory
		cudaMemcpy(pfSynWeightPCGPU[i], &pfSynWeightPCLinear[cpyStartInd],
				numGRPerGPU*sizeof(float), cudaMemcpyHostToDevice);

		for (int j = 0; j < cp.NUM_PC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFPCGPU[i] + j * inputPFPCGPUPitch[i]),
					0, cp.NUM_P_PC_FROM_GR_TO_PC * sizeof(float));
		}
		cudaMemset(inputSumPFPCMZGPU[i], 0, cp.NUM_PC / numGPUs * sizeof(float));

		cudaDeviceSynchronize();
	}
	
	testReduction();
	std::cout << "Finished Test." << std::endl;
}

void MZone::writeToState(ConnectivityParams &cp)
{
	cpyPFPCSynWCUDA(cp);

	for (int i = 0; i < cp.NUM_PC; i++)
	{
		as->inputSumPFPC[i] = inputSumPFPCMZH[i];
	}
}

void MZone::cpyPFPCSynWCUDA(ConnectivityParams &cp)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemcpy((void *)&pfSynWeightPCLinear[i*numGRPerGPU], pfSynWeightPCGPU[i],
				numGRPerGPU * sizeof(float), cudaMemcpyDeviceToHost);
	}

	for (int i = 0; i < cp.NUM_PC; i++)
	{
		for (int j = 0; j < cp.NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			as->pfSynWeightPC[i][j] = pfSynWeightPCLinear[i * cp.NUM_P_PC_FROM_GR_TO_PC + j];
		}
	}
}

void MZone::setErrDrive(ActivityParams *ap, float errDriveRelative)
{
	as->errDrive = errDriveRelative * ap->maxExtIncVIO;
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

void MZone::calcPCActivities(ConnectivityParams &cp, ActivityParams *ap)
{
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < cp.NUM_PC; i++)
		{
			float gSCPCSum;

			as->gPFPC[i] = as->gPFPC[i] + inputSumPFPCMZH[i] * ap->gIncGRtoPC;
			as->gPFPC[i] = as->gPFPC[i] * ap->gDecGRtoPC;
			as->gBCPC[i] = as->gBCPC[i] + as->inputBCPC[i] * ap->gIncBCtoPC;
			as->gBCPC[i] = as->gBCPC[i] * ap->gDecBCtoPC;

			gSCPCSum = 0;

			for (int j = 0; j < cp.NUM_P_PC_FROM_SC_TO_PC; j++)
			{
				as->gSCPC[i][j] = as->gSCPC[i][j] + ap->gIncSCtoPC *(1 - as->gSCPC[i][j]) * as->inputSCPC[i][j];
				as->gSCPC[i][j] = as->gSCPC[i][j] * ap->gDecSCtoPC;//GSCDECAYPC;
				gSCPCSum += as->gSCPC[i][j];
			}

			as->vPC[i] = as->vPC[i] +
					(ap->gLeakPC * (ap->eLeakPC - as->vPC[i])) -
					(as->gPFPC[i] * as->vPC[i]) +
					(as->gBCPC[i] * (ap->eBCtoPC-as->vPC[i])) +
					(gSCPCSum * (ap->eSCtoPC - as->vPC[i]));	

			as->threshPC[i] = as->threshPC[i] + (ap->threshDecPC * (ap->threshRestPC - as->threshPC[i]));

			as->apPC[i] = as->vPC[i] > as->threshPC[i];
			as->apBufPC[i] = (as->apBufPC[i] << 1) | (as->apPC[i] * 0x00000001);

			as->threshPC[i] = as->apPC[i] * ap->threshMaxPC + (!as->apPC[i]) * as->threshPC[i];
			as->pcPopAct = as->pcPopAct + as->apPC[i];
		}
	}		
}

void MZone::calcBCActivities(ConnectivityParams &cp, ActivityParams *ap, ct_uint32_t **pfInput)
{
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < cp.NUM_BC; i++)
		{
			int totalPFInput = 0;

			for (int j = 0; j < numGPUs; j++)
			{
				totalPFInput += pfInput[j][i];	
			}
			
			as->gPFBC[i] = as->gPFBC[i] + (sumPFBCInput[i] * ap->gIncGRtoBC);
			as->gPFBC[i] = as->gPFBC[i] * ap->gDecGRtoBC;
			as->gPCBC[i] = as->gPCBC[i] + (as->inputPCBC[i] * ap->gIncPCtoBC);
			as->gPCBC[i] = as->gPCBC[i] * ap->gDecPCtoBC;

			as->vBC[i] = as->vBC[i] +
					(ap->gLeakBC * (ap->eLeakBC - as->vBC[i])) -
					(as->gPFBC[i] * as->vBC[i]) +
					(as->gPCBC[i] * (ap->ePCtoBC - as->vBC[i]));

			as->threshBC[i] = as->threshBC[i] + ap->threshDecBC * (ap->threshRestBC - as->threshBC[i]);
			as->apBC[i] = as->vBC[i] > as->threshBC[i];
			as->apBufBC[i] = (as->apBufBC[i] << 1) | (as->apBC[i] * 0x00000001);

			as->threshBC[i] = as->apBC[i] * ap->threshMaxBC + (!as->apBC[i]) * (as->threshBC[i]);
		}
	}
}

void MZone::calcIOActivities(ConnectivityParams &cp, ActivityParams *ap)
{
#pragma omp parallel for
	clock_t t;
	t = clock();
	srand(t);
	
	float r = static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
	float gNoise = (r - 0.5) * 2.0;

	for (int i = 0; i < cp.NUM_IO; i++)
	{
		float gNCSum;
		gNCSum = 0;

		for (int j = 0; j < cp.NUM_P_IO_FROM_NC_TO_IO; j++)
		{
			as->gNCIO[i][j] = as->gNCIO[i][j] * exp(-ap->msPerTimeStep /
				(-ap->gDecTSofNCtoIO * exp(-as->gNCIO[i][j] / ap->gDecTTofNCtoIO) + ap->gDecT0ofNCtoIO));
			as->gNCIO[i][j] = as->gNCIO[i][j] + as->inputNCIO[i][j]
				* ap->gIncNCtoIO * exp(-as->gNCIO[i][j] / ap->gIncTauNCtoIO);
			gNCSum += as->gNCIO[i][j];

			as->inputNCIO[i][j] = 0;
		}

		gNCSum = 1.5 * gNCSum / 3.1;

		as->vIO[i] = as->vIO[i] + ap->gLeakIO*(ap->eLeakIO - as->vIO[i]) +
				gNCSum * (ap->eNCtoIO - as->vIO[i]) + as->vCoupleIO[i] +
				as->errDrive + gNoise;

		as->apIO[i] = as->vIO[i] > as->threshIO[i];
		as->apBufIO[i] = (as->apBufIO[i] << 1) |(as->apIO[i] * 0x00000001);

		as->threshIO[i] = ap->threshMaxIO * as->apIO[i] +
				(!as->apIO[i]) * (as->threshIO[i] + ap->threshDecIO * (ap->threshRestIO - as->threshIO[i]));
	}
	as->errDrive = 0;
}

void MZone::calcNCActivities(ConnectivityParams &cp, ActivityParams *ap)
{
//TODO: make function calcActivities which takes in the parameters which vary dep
//		on the type of activity that is being calculated. Else this is redundant  
float gDecay = exp(-1.0 / 20.0); 

#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < cp.NUM_NC; i++)
		{
			float gMFNMDASum;
			float gMFAMPASum;
			float gPCNCSum;

			int inputPCNCSum;
			int inputMFNCSum;

			gMFNMDASum   = 0;
			gMFAMPASum   = 0;
			inputMFNCSum = 0;

			for (int j = 0; j < cp.NUM_P_NC_FROM_MF_TO_NC; j++)
			{
				inputMFNCSum += as->inputMFNC[i][j];

				as->gMFAMPANC[i][j] = as->gMFAMPANC[i][j] * gDecay + 
					(ap->gAMPAIncMFtoNC * as->inputMFNC[i][j] * as->mfSynWeightNC[i][j]);
				gMFAMPASum += as->gMFAMPANC[i][j];
			}

			gMFNMDASum = gMFNMDASum * ap->msPerTimeStep / ((float)cp.NUM_P_NC_FROM_MF_TO_NC);
			gMFAMPASum = gMFAMPASum * ap->msPerTimeStep / ((float)cp.NUM_P_NC_FROM_MF_TO_NC);
			gMFNMDASum = gMFNMDASum * -as->vNC[i] / 80.0f;

			gPCNCSum = 0;
			inputPCNCSum = 0;

			for (int j = 0; j < cp.NUM_P_NC_FROM_PC_TO_NC; j++)
			{
				inputPCNCSum += as->inputPCNC[i][j];

				as->gPCNC[i][j] = as->gPCNC[i][j] * ap->gDecPCtoNC + 
					as->inputPCNC[i][j] * ap->gIncAvgPCtoNC*(1 - as->gPCNC[i][j]);
				gPCNCSum += as->gPCNC[i][j];

			}
			gPCNCSum = gPCNCSum * ap->msPerTimeStep / ((float)cp.NUM_P_NC_FROM_PC_TO_NC);
			
			as->vNC[i] = as->vNC[i] + ap->gLeakNC * (ap->eLeakNC - as->vNC[i]) -
					(gMFNMDASum + gMFAMPASum) * as->vNC[i] + gPCNCSum * (ap->ePCtoNC - as->vNC[i]);
			
			as->threshNC[i] = as->threshNC[i] + ap->threshDecNC * (ap->threshRestNC - as->threshNC[i]);
			as->apNC[i] = as->vNC[i] > as->threshNC[i];
			as->apBufNC[i] = (as->apBufNC[i] << 1) | (as->apNC[i] * 0x00000001);

			as->threshNC[i] = as->apNC[i] * ap->threshMaxNC + (!as->apNC[i]) * as->threshNC[i];
		}
	}
}

void MZone::updatePCOut(ConnectivityParams &cp)
{
#ifdef DEBUGOUT
	std::cout << "resetting inputPCBC " << cp.NUM_BC << std::endl;
#endif
	for (int i = 0; i < cp.NUM_BC; i++)
	{
		as->inputPCBC[i] = 0;
	}
#ifdef DEBUGOUT
	std::cout << "updating pc to bc " << std::endl;
#endif
	for (int i = 0; i < cp.NUM_PC; i++)
	{
		for (int j = 0; j < cp.NUM_P_PC_FROM_PC_TO_BC; j++)
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
	for (int i = 0; i < cp.NUM_NC; i++)
	{
		for (int j = 0; j < cp.NUM_P_NC_FROM_PC_TO_NC; j++)
		{
#ifdef DEBUGOUT
			std::cout << "i: " << i << " j: " << j <<
				"cs->pNCfromPCtoNC[i][j]: " << cs->pNCfromPCtoNC[i][j] << std::endl;
#endif
			as->inputPCNC[i][j] = as->apPC[cs->pNCfromPCtoNC[i][j]];
		}
	}
#ifdef DEBUGOUT
	std::cout << "finished " << std::endl;
#endif
}

void MZone::updateBCPCOut(ConnectivityParams &cp)
{
	for (int i = 0; i < cp.NUM_PC; i++) as->inputBCPC[i] = 0;
	
	for (int i = 0; i < cp.NUM_BC; i++)
	{
		if (as->apBC[i])
		{
			for (int j = 0; j < cp.NUM_P_BC_FROM_BC_TO_PC; j++)
			{
				as->inputBCPC[cs->pBCfromBCtoPC[i][j]]++;
			}
		}
	}
}

void MZone::updateSCPCOut(ConnectivityParams &cp)
{
#pragma omp parallel for
	for (int i = 0; i < cp.NUM_PC; i++)
	{
		for (int j = 0; j < cp.NUM_P_PC_FROM_SC_TO_PC; j++)
		{
			as->inputSCPC[i][j] = apSCInput[cs->pPCfromSCtoPC[i][j]];
		}
	}
}

void MZone::updateIOOut(ConnectivityParams &cp, ActivityParams *ap)
{
	for (int i = 0; i < cp.NUM_IO; i++)
	{
		as->pfPCPlastTimerIO[i] = (!as->apIO[i]) * (as->pfPCPlastTimerIO[i] +1 ) + as->apIO[i]*ap->tsLTPEndAPIO;
		as->vCoupleIO[i] = 0;
		for (int j = 0; j < cp.NUM_P_IO_IN_IO_TO_IO; j++)
		{
			as->vCoupleIO[i] += ap->coupleRiRjRatioIO * (as->vIO[cs->pIOInIOIO[i][j]] - as->vIO[i]);
		}
	}
}

void MZone::updateNCOut(ConnectivityParams &cp, ActivityParams *ap)
{
	for (int i = 0; i < cp.NUM_NC; i++)
	{
		as->synIOPReleaseNC[i] *= exp(-ap->msPerTimeStep / 
				(ap->relPDecTSofNCtoIO * exp(-as->synIOPReleaseNC[i] / ap->relPDecTTofNCtoIO) +
				 ap->relPDecT0ofNCtoIO));
		as->synIOPReleaseNC[i] += as->apNC[i] * ap->relPIncNCtoIO *
				exp(-as->synIOPReleaseNC[i] / ap->relPIncTauNCtoIO);
	}

	for (int i = 0; i < cp.NUM_IO; i++)
	{
		for (int j = 0; j < cp.NUM_P_IO_FROM_NC_TO_IO; j++)
		{
			as->inputNCIO[i][j] = (randGen->Random() < as->synIOPReleaseNC[cs->pIOfromNCtoIO[i][j]]);
		}
	}
}

void MZone::updateMFNCOut(ConnectivityParams &cp)
{
	for (int i = 0; i < cp.NUM_NC; i++)
	{
		for (int j = 0; j < cp.NUM_P_NC_FROM_MF_TO_NC; j++)
		{
			as->inputMFNC[i][j] = apMFInput[cs->pNCfromMFtoNC[i][j]];
		}
	}
}

void MZone::updateMFNCSyn(ConnectivityParams &cp, ActivityParams *ap, const ct_uint8_t *histMF, unsigned long t)
{
	
	bool reset;
	float avgAllAPPC;
	bool doLTD;
	bool doLTP;

#ifdef DEBUGOUT
	float sumSynW;
#endif
	if(t % ap->tsPerPopHistBinPC == 0) return;

	histMFInput = histMF;

	as->histPCPopActSum = (as->histPCPopActSum) - (as->histPCPopAct[as->histPCPopActCurBinN]) + (as->pcPopAct);
	as->histPCPopAct[as->histPCPopActCurBinN] = as->pcPopAct;
	as->pcPopAct = 0;
	as->histPCPopActCurBinN++;
	as->histPCPopActCurBinN %= ap->numPopHistBinsPC;

	avgAllAPPC = ((float)as->histPCPopActSum) / ap->numPopHistBinsPC;

#ifdef DEBUGOUT
	std::cout << "avgAllAPPC: " << avgAllAPPC << std::endl;
#endif

	doLTD = false;
	doLTP = false;
	if (avgAllAPPC >= ap->synLTDPCPopActThreshMFtoNC && !as->noLTDMFNC)
	{
		doLTD = true;
		as->noLTDMFNC = true;
	}
	else if (avgAllAPPC < ap->synLTDPCPopActThreshMFtoNC)
	{
		as->noLTDMFNC = false;
	}

	if (avgAllAPPC <= ap->synLTPPCPopActThreshMFtoNC && !as->noLTPMFNC)
	{
		doLTP = true;
		as->noLTPMFNC = true;
	}
	else if (avgAllAPPC > ap->synLTPPCPopActThreshMFtoNC)
	{
		as->noLTPMFNC = false;
	}

#ifdef DEBUGOUT
	sumSynW = 0;
#endif
	for (int i = 0; i < cp.NUM_NC; i++)
	{
		for(int j = 0; j < cp.NUM_P_NC_FROM_MF_TO_NC; j++)
		{
			float synWDelta;
			synWDelta = histMFInput[cs->pNCfromMFtoNC[i][j]] * (doLTD * ap->synLTDStepSizeMFtoNC +
					doLTP * ap->synLTPStepSizeMFtoNC);
			as->mfSynWeightNC[i][j] += synWDelta;
			as->mfSynWeightNC[i][j] *= as->mfSynWeightNC[i][j] > 0;
			as->mfSynWeightNC[i][j] *= as->mfSynWeightNC[i][j] <= 1; 
			as->mfSynWeightNC[i][j]	+= as->mfSynWeightNC[i][j] > 1;
			
			//Now uses isTrueMF to take collaterals into account
			as->mfSynWeightNC[i][j] *= isTrueMF[cs->pNCfromMFtoNC[i][j]];
#ifdef DEBUGOUT
			sumSynW += as->mfSynWeightNC[i][j];
#endif
		}
	}
#ifdef DEBUGOUT
	std::cout << sumSynW / cp.NUM_MF << std::endl;
#endif
}

void MZone::runPFPCOutCUDA(ConnectivityParams &cp, cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callUpdatePFPCOutKernel(sts[i][streamN], updatePFPCNumBlocks, updatePFPCNumGRPerB,
				apBufGRGPU[i], delayBCPCSCMaskGRGPU[i], pfSynWeightPCGPU[i], inputPFPCGPU[i],
				inputPFPCGPUPitch[i], cp.NUM_P_PC_FROM_GR_TO_PC_P2);
	}
}

void MZone::runPFPCSumCUDA(ConnectivityParams &cp, cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callSumKernel<float, true, false>(sts[i][streamN], inputPFPCGPU[i], inputPFPCGPUPitch[i],
				inputSumPFPCMZGPU[i], 1, cp.NUM_PC / numGPUs, 1, cp.NUM_P_PC_FROM_GR_TO_PC);
	}
}

void MZone::cpyPFPCSumCUDA(ConnectivityParams &cp, cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemcpyAsync(&inputSumPFPCMZH[cp.NUM_PC * i / numGPUs], inputSumPFPCMZGPU[i],
				cp.NUM_PC / numGPUs * sizeof(float), cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void MZone::runPFPCPlastCUDA(ConnectivityParams &cp, ActivityParams *ap, cudaStream_t **sts, int streamN, unsigned long t)
{
	cudaError_t error;
	if (t % ap->tsPerHistBinGR == 0)
	{
		int curGROffset;
		int curGPUInd;
		int curIOInd;

		int numGRPerIO;

		curGROffset = 0;
		curGPUInd   = 0;
		curIOInd    = 0;

		numGRPerIO = cp.NUM_GR / cp.NUM_IO;

		for (int i = 0; i < cp.NUM_IO; i++)
		{
			if (as->pfPCPlastTimerIO[i] < (ap->tsLTDStartAPIO + ((int)(ap->tsLTDDurationIO))) &&
					as->pfPCPlastTimerIO[i] >= ap->tsLTDStartAPIO)
			{
				pfPCPlastStepIO[i] = tempGRPCLTDStep;
			}
			else if (as->pfPCPlastTimerIO[i] >= ap->tsLTPStartAPIO ||
					as->pfPCPlastTimerIO[i] < ap->tsLTPEndAPIO)
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
		for (int i = 0; i < cp.NUM_GR; i += cp.NUM_P_PC_FROM_GR_TO_PC)
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
					historyGRGPU[curGPUInd], ap->grPCHistCheckBinIO, curGROffset, pfPCPlastStepIO[curIOInd]);

			curGROffset += cp.NUMP_PC_FROM_GR_TO_PC;
		}
	}
}

void MZone::setGRPCPlastSteps(float ltdStep, float ltpStep)
{
	tempGRPCLTDStep = ltdStep;
	tempGRPCLTPStep = ltpStep;
}

void MZone::resetGRPCPlastSteps(ActivityParams *ap)
{
	tempGRPCLTDStep = ap->synLTDStepSizeGRtoPC;
	tempGRPCLTPStep = ap->synLTPStepSizeGRtoPC;
}

const float* MZone::exportPFPCWeights(ConnectivityParams &cp)
{
	cpyPFPCSynWCUDA(cp);
	return (const float *)pfSynWeightPCLinear; 
}

// Why not write one export function which takes in the weight you want to export?
const ct_uint8_t* MZone::exportAPNC()
{
	return (const ct_uint8_t *)as->apNC;
}

const ct_uint8_t* MZone::exportAPBC()
{
	return (const ct_uint8_t *)as->apBC;
}

const ct_uint8_t* MZone::exportAPPC()
{
	return (const ct_uint8_t *)as->apPC;
}

const ct_uint8_t* MZone::exportAPIO()
{
	return (const ct_uint8_t *)as->apIO;
}

const float* MZone::exportgBCPC()
{
	return (const float *)as->gBCPC;
}

const float* MZone::exportgPFPC()
{
	return (const float *)as->gPFPC;
}

const float* MZone::exportVmBC()
{
	return (const float *)as->vBC;
}

const float* MZone::exportVmPC()
{
	return (const float *)as->vPC;
}

const float* MZone::exportVmNC()
{
	return (const float *)as->vNC;
}

const float* MZone::exportVmIO()
{
	return (const float *)as->vIO;
}

const unsigned int* MZone::exportAPBufBC()
{
	return (const unsigned int *)as->apBufBC;
}

const ct_uint32_t* MZone::exportAPBufPC()
{
	return (const ct_uint32_t *)as->apBufPC;
}

const ct_uint32_t* MZone::exportAPBufIO()
{
	return (const ct_uint32_t *)as->apBufIO;
}

const ct_uint32_t* MZone::exportAPBufNC()
{
	return (const ct_uint32_t *)as->apBufNC;
}

void MZone::testReduction(ConnectivityParams &cp)
{
	cudaError_t error;
	cudaStream_t *sts = new cudaStream_t[numGPUs];

	float hostTestData[cp.NUM_GR]();
	float hostPCSum[cp.NUM_PC]();
	float hostBCSum[cp.NUM_BC]();
	float hostSCSum[cp.NUM_SC]();

	float gpuToHostPCSum[cp.NUM_PC]();
	float gpuToHostBCSum[cp.NUM_BC]();
	float gpuToHostSCSum[cp.NUM_SC]();

	// leaving these dynamic for now as i do not understand cuda oof
	float **gpuPCTestData = new float*[numGPUs];
	float **gpuBCTestData = new float*[numGPUs];
	float **gpuSCTestData = new float*[numGPUs];

	size_t *gpuPCP = new siz_t[numGPUs];
	size_t *gpuBCP = new siz_t[numGPUs];
	size_t *gpuSCP = new siz_t[numGPUs];

	float **gpuPCSum = new float*[numGPUs];
	float **gpuBCSum = new float*[numGPUs];
	float **gpuSCSum = new float*[numGPUs];

	for (int i = 0; i < cp.NUM_GR; i++)
	{
		hostTestData[i] = randGen->Random();
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaStreamCreate(&sts[i]);

		cudaMallocPitch(&gpuPCTestData[i], &gpuPCP[i],
				cp.NUM_P_PC_FROM_GR_TO_PC * sizeof(float), cp.NUM_PC / numGPUs);
		cudaMallocPitch(&gpuBCTestData[i], &gpuBCP[i],
				cp.NUM_P_BC_FROM_GR_TO_BC * sizeof(float), cp.NUM_BC / numGPUs);
		cudaMallocPitch(&gpuSCTestData[i], &gpuSCP[i],
				cp.NUM_P_SC_FROM_GR_TO_SC*sizeof(float), cp.num_SC / numGPUs);

		cudaMalloc(&gpuPCSum[i], cp.NUM_PC / numGPUs * sizeof(float));
		cudaMalloc(&gpuBCSum[i], cp.NUM_BC / numGPUs * sizeof(float));
		cudaMalloc(&gpuSCSum[i], cp.NUM_SC / numGPUs * sizeof(float));

		error = cudaGetLastError();
		std::cout << "allocating memory for gpu " << i << " " << cudaGetErrorString(error) << std::endl;

		cudaDeviceSynchronize();
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		for (int j = 0; j < cp.NUM_PC / numGPUs; j++)
		{
			cudaMemcpy(((char *)gpuPCTestData[i] + j * gpuPCP[i]),
					&hostTestData[i * numGRPerGPU + j * cp.NUM_P_PC_FROM_GR_TO_PC],
					cp.NUM_P_PC_FROM_GR_TO_PC * sizeof(float), cudaMemcpyHostToDevice);
		}

		for (int j = 0; j < cp.NUM_BC / numGPUs; j++)
		{
			cudaMemcpy(((char *)gpuBCTestData[i] + j * gpuBCP[i]),
					&hostTestData[i * numGRPerGPU + j * cp.NUM_P_BC_FROM_GR_TO_BC],
					cp.NUM_P_BC_FROM_GR_TO_BC * sizeof(float), cudaMemcpyHostToDevice);
		}

		for (int j = 0; j < cp.NUM_SC / numGPUs; j++)
		{
			cudaMemcpy(((char *)gpuSCTestData[i] + j * gpuSCP[i]),
					&hostTestData[i * numGRPerGPU + j * cp.NUM_P_SC_FROM_GR_TO_SC],
					cp.NUM_P_SC_FROM_GR_TO_SC * sizeof(float), cudaMemcpyHostToDevice);
		}

		error = cudaGetLastError();
		std::cout << "copying memory for gpu " << i << " " << cudaGetErrorString(error) << std::endl;

		cudaDeviceSynchronize();
	}

	for (int i = 0; i < cp.NUM_PC; i++)
	{
		hostPCSum[i] = 0;

		for (int j = 0; j < cp.NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			hostPCSum[i] += hostTestData[i * cp.NUM_P_PC_FROM_GR_TO_PC + j];
		}
	}

	for (int i = 0; i < cp.NUM_BC; i++)
	{
		hostBCSum[i] = 0;

		for (int j = 0; j < cp.NUM_P_BC_FROM_GR_TO_BC; j++)
		{
			hostBCSum[i] += hostTestData[i * cp.NUM_P_BC_FROM_GR_TO_BC + j];
		}
	}

	for (int i = 0; i < cp.NUM_SC; i++)
	{
		hostSCSum[i] = 0;

		for (int j = 0; j < cp.NUM_P_SC_FROM_GR_TO_SC; j++)
		{
			hostSCSum[i] += hostTestData[i * cp.NUM_P_SC_FROM_GR_TO_SC + j];
		}
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callSumKernel<float, true, false>(sts[i], gpuPCTestData[i], gpuPCP[i],
				gpuPCSum[i], 1, cp.NUM_PC / numGPUs, 1, cp.NUM_P_PC_FROM_GR_TO_PC);

		callSumKernel<float, true, false>(sts[i], gpuBCTestData[i], gpuBCP[i],
				gpuBCSum[i], 1, cp.NUM_BC / numGPUs, 1, cp.NUM_P_BC_FROM_GR_TO_BC);

		callSumKernel<float, true, false>(sts[i], gpuSCTestData[i], gpuSCP[i],
				gpuSCSum[i], 1, cp.NUM_SC / numGPUs, 1, cp.NUM_P_SC_FROM_GR_TO_SC);

		cudaDeviceSynchronize();

		error = cudaGetLastError();
		std::cout << "calling sum kernels for gpu " << i << " " << cudaGetErrorString(error) << std::endl;
	}

	for (int i = 0; i<numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaMemcpy(&gpuToHostPCSum[i * cp.NUM_PC / numGPUs], gpuPCSum[i],
				32 * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gpuToHostBCSum[i * cp.NUM_BC / numGPUs], gpuBCSum[i],
				cp.NUM_BC / numGPUs * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gpuToHostSCSum[i * cp.NUM_SC / numGPUs], gpuSCSum[i],
				cp.NUM_SC / numGPUs * sizeof(float), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
	}

	std::cout << "NumPC per GPU: " << cp.NUM_PC / numGPUs << std::endl <<
		"NumBC per GPU: " << cp.NUM_BC / numGPUs << 
		"NUMSC per GPU: " << cp.NUM_SC / numGPUs << std::endl;

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

