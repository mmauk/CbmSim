/*
 * mzone.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: consciousness
 */

#include "mzonemodules/mzone.h"

MZone::MZone() {}

MZone::MZone(ActivityParams *ap, MZoneConnectivityState *cs,
		MZoneActivityState *as, int randSeed, ct_uint32_t **actBufGRGPU,
		ct_uint32_t **delayMaskGRGPU, ct_uint64_t **histGRGPU, int gpuIndStart, int numGPUs)
{
	// TODO: make this a non dynamic object wtf bro
	randGen = new CRandomSFMT0(randSeed);

	this->ap = *ap; /* deep copy on what input ap points to, for now */
	this->cs = cs; /* shallow copy (boo) */
	this->as = as; /* also shallow copy */

	// NOTE if we turn these guys into unique ptrs, we'll have to refactor
	// consider ownership: who should own these guys? maybe they should be global to both
	// innet and mzone (so within cbmsimcore) and fed in as const args to the respective
	// functions that call update kernels (06/16/2022)
	apBufGRGPU           = actBufGRGPU;
	delayBCPCSCMaskGRGPU = delayMaskGRGPU;
	historyGRGPU         = histGRGPU;

	pfSynWeightPCLinear = new float[NUM_GR];
	pfPCPlastStepIO = new float[NUM_IO];

	tempGRPCLTDStep = ap->synLTDStepSizeGRtoPC;
	tempGRPCLTPStep = ap->synLTPStepSizeGRtoPC;

	this->numGPUs     = numGPUs;
	this->gpuIndStart = gpuIndStart;

	std::cout << "Initializing CUDA..." << std::endl;
	initCUDA();
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

void MZone::initCUDA()
{
	int maxNumGPUs;
	cudaGetDeviceCount(&maxNumGPUs);

	numGRPerGPU = NUM_GR / numGPUs;

	updatePFPCNumGRPerB = 512;
	updatePFPCNumBlocks = numGRPerGPU / updatePFPCNumGRPerB;

	updatePFPCSynWNumGRPerB = 512 * (NUM_P_PC_FROM_GR_TO_PC > 512) +
			NUM_P_PC_FROM_GR_TO_PC * (NUM_P_PC_FROM_GR_TO_PC <= 512);
	updatePFPCSynWNumBlocks = NUM_P_PC_FROM_GR_TO_PC / updatePFPCSynWNumGRPerB;

	cudaSetDevice(0 + gpuIndStart);
	//allocate host cuda memory
	cudaHostAlloc((void **)&inputSumPFPCMZH, NUM_PC * sizeof(float), cudaHostAllocPortable);

	cudaDeviceSynchronize();
	//initialize host cuda memory
	for (int i = 0; i < NUM_PC; i++)
	{
		inputSumPFPCMZH[i] = 0;
	}

	for (int i = 0; i < NUM_PC; i++)
	{
		for (int j = 0; j < NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			// TODO: get rid of pfSynWeightLinear and use our linearized version directly
			pfSynWeightPCLinear[i * NUM_P_PC_FROM_GR_TO_PC + j] = as->pfSynWeightPC[i * NUM_P_PC_FROM_GR_TO_PC + j];
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
				NUM_P_PC_FROM_GR_TO_PC * sizeof(float), NUM_PC / numGPUs);
		cudaMalloc((void **)&inputSumPFPCMZGPU[i], NUM_PC / numGPUs * sizeof(float));

		cudaDeviceSynchronize();
		//initialize device cuda memory
		cudaMemcpy(pfSynWeightPCGPU[i], &pfSynWeightPCLinear[cpyStartInd],
				numGRPerGPU*sizeof(float), cudaMemcpyHostToDevice);

		for (int j = 0; j < NUM_PC/numGPUs; j++)
		{
			cudaMemset(((char *)inputPFPCGPU[i] + j * inputPFPCGPUPitch[i]),
					0, NUM_P_PC_FROM_GR_TO_PC * sizeof(float));
		}
		cudaMemset(inputSumPFPCMZGPU[i], 0, NUM_PC / numGPUs * sizeof(float));

		cudaDeviceSynchronize();
	}
	
	testReduction();
	std::cout << "Finished Test." << std::endl;
}

void MZone::writeToState()
{
	cpyPFPCSynWCUDA();

	for (int i = 0; i < NUM_PC; i++)
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

	for (int i = 0; i < NUM_PC; i++)
	{
		for (int j = 0; j < NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			as->pfSynWeightPC[i * NUM_P_PC_FROM_GR_TO_PC + j] = pfSynWeightPCLinear[i * NUM_P_PC_FROM_GR_TO_PC + j];
		}
	}
}

void MZone::setErrDrive(float errDriveRelative)
{
	as->errDrive = errDriveRelative * ap.maxExtIncVIO;
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
		for (int i = 0; i < NUM_PC; i++)
		{
			float gSCPCSum;

			as->gPFPC[i] = as->gPFPC[i] + inputSumPFPCMZH[i] * ap.gIncGRtoPC;
			as->gPFPC[i] = as->gPFPC[i] * ap.gDecGRtoPC;
			as->gBCPC[i] = as->gBCPC[i] + as->inputBCPC[i] * ap.gIncBCtoPC;
			as->gBCPC[i] = as->gBCPC[i] * ap.gDecBCtoPC;

			gSCPCSum = 0;

			for (int j = 0; j < NUM_P_PC_FROM_SC_TO_PC; j++)
			{
				as->gSCPC[i * NUM_P_PC_FROM_SC_TO_PC + j] = as->gSCPC[i * NUM_P_PC_FROM_SC_TO_PC + j]
				   + ap.gIncSCtoPC * (1 - as->gSCPC[i * NUM_P_PC_FROM_SC_TO_PC + j])
				   * as->inputSCPC[i * NUM_P_PC_FROM_SC_TO_PC + j];
				as->gSCPC[i * NUM_P_PC_FROM_SC_TO_PC + j] = as->gSCPC[i * NUM_P_PC_FROM_SC_TO_PC + j]
				   * ap.gDecSCtoPC; //GSCDECAYPC;
				gSCPCSum += as->gSCPC[i * NUM_P_PC_FROM_SC_TO_PC + j];
			}

			as->vPC[i] = as->vPC[i] +
					(ap.gLeakPC * (ap.eLeakPC - as->vPC[i])) -
					(as->gPFPC[i] * as->vPC[i]) +
					(as->gBCPC[i] * (ap.eBCtoPC-as->vPC[i])) +
					(gSCPCSum * (ap.eSCtoPC - as->vPC[i]));	

			as->threshPC[i] = as->threshPC[i] + (ap.threshDecPC * (ap.threshRestPC - as->threshPC[i]));

			as->apPC[i] = as->vPC[i] > as->threshPC[i];
			as->apBufPC[i] = (as->apBufPC[i] << 1) | (as->apPC[i] * 0x00000001);

			as->threshPC[i] = as->apPC[i] * ap.threshMaxPC + (!as->apPC[i]) * as->threshPC[i];
			as->pcPopAct = as->pcPopAct + as->apPC[i];
		}
	}		
}

void MZone::calcBCActivities(ct_uint32_t **pfInput)
{
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < NUM_BC; i++)
		{
			int totalPFInput = 0;

			for (int j = 0; j < numGPUs; j++)
			{
				totalPFInput += pfInput[j][i];	
			}
			
			as->gPFBC[i] = as->gPFBC[i] + (sumPFBCInput[i] * ap.gIncGRtoBC);
			as->gPFBC[i] = as->gPFBC[i] * ap.gDecGRtoBC;
			as->gPCBC[i] = as->gPCBC[i] + (as->inputPCBC[i] * ap.gIncPCtoBC);
			as->gPCBC[i] = as->gPCBC[i] * ap.gDecPCtoBC;

			as->vBC[i] = as->vBC[i] +
					(ap.gLeakBC * (ap.eLeakBC - as->vBC[i])) -
					(as->gPFBC[i] * as->vBC[i]) +
					(as->gPCBC[i] * (ap.ePCtoBC - as->vBC[i]));

			as->threshBC[i] = as->threshBC[i] + ap.threshDecBC * (ap.threshRestBC - as->threshBC[i]);
			as->apBC[i] = as->vBC[i] > as->threshBC[i];
			as->apBufBC[i] = (as->apBufBC[i] << 1) | (as->apBC[i] * 0x00000001);

			as->threshBC[i] = as->apBC[i] * ap.threshMaxBC + (!as->apBC[i]) * (as->threshBC[i]);
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

	for (int i = 0; i < NUM_IO; i++)
	{
		float gNCSum;
		gNCSum = 0;

		for (int j = 0; j < NUM_P_IO_FROM_NC_TO_IO; j++)
		{
			as->gNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j] = as->gNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j]
			   * exp(-ap.msPerTimeStep /
				(-ap.gDecTSofNCtoIO * exp(-as->gNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j] / ap.gDecTTofNCtoIO)
				 + ap.gDecT0ofNCtoIO));
			as->gNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j] = as->gNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j]
			   + as->inputNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j]
			   * ap.gIncNCtoIO * exp(-as->gNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j] / ap.gIncTauNCtoIO);
			gNCSum += as->gNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j];

			as->inputNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j] = 0;
		}

		gNCSum = 1.5 * gNCSum / 3.1;

		as->vIO[i] = as->vIO[i] + ap.gLeakIO*(ap.eLeakIO - as->vIO[i]) +
				gNCSum * (ap.eNCtoIO - as->vIO[i]) + as->vCoupleIO[i] +
				as->errDrive + gNoise;

		as->apIO[i] = as->vIO[i] > as->threshIO[i];
		as->apBufIO[i] = (as->apBufIO[i] << 1) |(as->apIO[i] * 0x00000001);

		as->threshIO[i] = ap.threshMaxIO * as->apIO[i] +
				(!as->apIO[i]) * (as->threshIO[i] + ap.threshDecIO * (ap.threshRestIO - as->threshIO[i]));
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
		for (int i = 0; i < NUM_NC; i++)
		{
			float gMFNMDASum;
			float gMFAMPASum;
			float gPCNCSum;

			int inputPCNCSum;
			int inputMFNCSum;

			gMFNMDASum   = 0;
			gMFAMPASum   = 0;
			inputMFNCSum = 0;

			for (int j = 0; j < NUM_P_NC_FROM_MF_TO_NC; j++)
			{
				inputMFNCSum += as->inputMFNC[i * NUM_P_NC_FROM_MF_TO_NC + j];

				as->gMFAMPANC[i * NUM_P_NC_FROM_MF_TO_NC + j] = as->gMFAMPANC[i * NUM_P_NC_FROM_MF_TO_NC + j]
				   * gDecay + (ap.gAMPAIncMFtoNC * as->inputMFNC[i * NUM_P_NC_FROM_MF_TO_NC + j]
					 * as->mfSynWeightNC[i * NUM_P_NC_FROM_MF_TO_NC + j]);
				gMFAMPASum += as->gMFAMPANC[i * NUM_P_NC_FROM_MF_TO_NC + j];
			}

			gMFNMDASum = gMFNMDASum * ap.msPerTimeStep / ((float)NUM_P_NC_FROM_MF_TO_NC);
			gMFAMPASum = gMFAMPASum * ap.msPerTimeStep / ((float)NUM_P_NC_FROM_MF_TO_NC);
			gMFNMDASum = gMFNMDASum * -as->vNC[i] / 80.0f;

			gPCNCSum = 0;
			inputPCNCSum = 0;

			for (int j = 0; j < NUM_P_NC_FROM_PC_TO_NC; j++)
			{
				inputPCNCSum += as->inputPCNC[i * NUM_P_NC_FROM_PC_TO_NC + j];

				as->gPCNC[i * NUM_P_NC_FROM_PC_TO_NC + j] = as->gPCNC[i * NUM_P_NC_FROM_PC_TO_NC + j] * ap.gDecPCtoNC + 
					as->inputPCNC[i * NUM_P_NC_FROM_PC_TO_NC + j] * ap.gIncAvgPCtoNC
					* (1 - as->gPCNC[i * NUM_P_NC_FROM_PC_TO_NC + j]);
				gPCNCSum += as->gPCNC[i * NUM_P_NC_FROM_PC_TO_NC + j];

			}
			gPCNCSum = gPCNCSum * ap.msPerTimeStep / ((float)NUM_P_NC_FROM_PC_TO_NC);
			
			as->vNC[i] = as->vNC[i] + ap.gLeakNC * (ap.eLeakNC - as->vNC[i]) -
					(gMFNMDASum + gMFAMPASum) * as->vNC[i] + gPCNCSum * (ap.ePCtoNC - as->vNC[i]);
			
			as->threshNC[i] = as->threshNC[i] + ap.threshDecNC * (ap.threshRestNC - as->threshNC[i]);
			as->apNC[i] = as->vNC[i] > as->threshNC[i];
			as->apBufNC[i] = (as->apBufNC[i] << 1) | (as->apNC[i] * 0x00000001);

			as->threshNC[i] = as->apNC[i] * ap.threshMaxNC + (!as->apNC[i]) * as->threshNC[i];
		}
	}
}

void MZone::updatePCOut()
{
#ifdef DEBUGOUT
	std::cout << "resetting inputPCBC " << NUM_BC << std::endl;
#endif
	for (int i = 0; i < NUM_BC; i++)
	{
		as->inputPCBC[i] = 0;
	}
#ifdef DEBUGOUT
	std::cout << "updating pc to bc " << std::endl;
#endif
	for (int i = 0; i < NUM_PC; i++)
	{
		for (int j = 0; j < NUM_P_PC_FROM_PC_TO_BC; j++)
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
	for (int i = 0; i < NUM_NC; i++)
	{
		for (int j = 0; j < NUM_P_NC_FROM_PC_TO_NC; j++)
		{
#ifdef DEBUGOUT
			std::cout << "i: " << i << " j: " << j <<
				"cs->pNCfromPCtoNC[i][j]: " << cs->pNCfromPCtoNC[i][j] << std::endl;
#endif
			as->inputPCNC[i * NUM_P_NC_FROM_PC_TO_NC + j] = as->apPC[cs->pNCfromPCtoNC[i][j]];
		}
	}
#ifdef DEBUGOUT
	std::cout << "finished " << std::endl;
#endif
}

void MZone::updateBCPCOut()
{
	for (int i = 0; i < NUM_PC; i++) as->inputBCPC[i] = 0;
	
	for (int i = 0; i < NUM_BC; i++)
	{
		if (as->apBC[i])
		{
			for (int j = 0; j < NUM_P_BC_FROM_BC_TO_PC; j++)
			{
				as->inputBCPC[cs->pBCfromBCtoPC[i][j]]++;
			}
		}
	}
}

void MZone::updateSCPCOut()
{
#pragma omp parallel for
	for (int i = 0; i < NUM_PC; i++)
	{
		for (int j = 0; j < NUM_P_PC_FROM_SC_TO_PC; j++)
		{
			as->inputSCPC[i * NUM_P_PC_FROM_SC_TO_PC + j] = apSCInput[cs->pPCfromSCtoPC[i][j]];
		}
	}
}

void MZone::updateIOOut()
{
	for (int i = 0; i < NUM_IO; i++)
	{
		as->pfPCPlastTimerIO[i] = (!as->apIO[i]) * (as->pfPCPlastTimerIO[i] +1 ) + as->apIO[i]*ap.tsLTPEndAPIO;
		as->vCoupleIO[i] = 0;
		for (int j = 0; j < NUM_P_IO_IN_IO_TO_IO; j++)
		{
			as->vCoupleIO[i] += ap.coupleRiRjRatioIO * (as->vIO[cs->pIOInIOIO[i][j]] - as->vIO[i]);
		}
	}
}

void MZone::updateNCOut()
{
	for (int i = 0; i < NUM_NC; i++)
	{
		as->synIOPReleaseNC[i] *= exp(-ap.msPerTimeStep / 
				(ap.relPDecTSofNCtoIO * exp(-as->synIOPReleaseNC[i] / ap.relPDecTTofNCtoIO) +
				 ap.relPDecT0ofNCtoIO));
		as->synIOPReleaseNC[i] += as->apNC[i] * ap.relPIncNCtoIO *
				exp(-as->synIOPReleaseNC[i] / ap.relPIncTauNCtoIO);
	}

	for (int i = 0; i < NUM_IO; i++)
	{
		for (int j = 0; j < NUM_P_IO_FROM_NC_TO_IO; j++)
		{
			as->inputNCIO[i * NUM_P_IO_FROM_NC_TO_IO + j] = (randGen->Random() < as->synIOPReleaseNC[cs->pIOfromNCtoIO[i][j]]);
		}
	}
}

void MZone::updateMFNCOut()
{
	for (int i = 0; i < NUM_NC; i++)
	{
		for (int j = 0; j < NUM_P_NC_FROM_MF_TO_NC; j++)
		{
			as->inputMFNC[i * NUM_P_NC_FROM_MF_TO_NC + j] = apMFInput[cs->pNCfromMFtoNC[i][j]];
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
//	if(t % ap.tsPerPopHistBinPC == 0) return;
//
//	//histMFInput = histMF;
//
//	as->histPCPopActSum = (as->histPCPopActSum) - (as->histPCPopAct[as->histPCPopActCurBinN]) + (as->pcPopAct);
//	as->histPCPopAct[as->histPCPopActCurBinN] = as->pcPopAct;
//	as->pcPopAct = 0;
//	as->histPCPopActCurBinN++;
//	as->histPCPopActCurBinN %= ap.numPopHistBinsPC;
//
//	avgAllAPPC = ((float)as->histPCPopActSum) / ap.numPopHistBinsPC;
//
//#ifdef DEBUGOUT
//	std::cout << "avgAllAPPC: " << avgAllAPPC << std::endl;
//#endif
//
//	doLTD = false;
//	doLTP = false;
//	if (avgAllAPPC >= ap.synLTDPCPopActThreshMFtoNC && !as->noLTDMFNC)
//	{
//		doLTD = true;
//		as->noLTDMFNC = true;
//	}
//	else if (avgAllAPPC < ap.synLTDPCPopActThreshMFtoNC)
//	{
//		as->noLTDMFNC = false;
//	}
//
//	if (avgAllAPPC <= ap.synLTPPCPopActThreshMFtoNC && !as->noLTPMFNC)
//	{
//		doLTP = true;
//		as->noLTPMFNC = true;
//	}
//	else if (avgAllAPPC > ap.synLTPPCPopActThreshMFtoNC)
//	{
//		as->noLTPMFNC = false;
//	}
//
//#ifdef DEBUGOUT
//	sumSynW = 0;
//#endif
//	for (int i = 0; i < NUM_NC; i++)
//	{
//		for(int j = 0; j < NUM_P_NC_FROM_MF_TO_NC; j++)
//		{
//			float synWDelta;
//			synWDelta = histMF[cs->pNCfromMFtoNC[i][j]] * (doLTD * ap.synLTDStepSizeMFtoNC +
//					doLTP * ap.synLTPStepSizeMFtoNC);
//			as->mfSynWeightNC[i][j] += synWDelta;
//			as->mfSynWeightNC[i][j] *= as->mfSynWeightNC[i][j] > 0;
//			as->mfSynWeightNC[i][j] *= as->mfSynWeightNC[i][j] <= 1; 
//			as->mfSynWeightNC[i][j]	+= as->mfSynWeightNC[i][j] > 1;
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
				inputPFPCGPUPitch[i], NUM_P_PC_FROM_GR_TO_PC_P2);
	}
}

void MZone::runPFPCSumCUDA(cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callSumKernel<float, true, false>(sts[i][streamN], inputPFPCGPU[i], inputPFPCGPUPitch[i],
				inputSumPFPCMZGPU[i], 1, NUM_PC / numGPUs, 1, NUM_P_PC_FROM_GR_TO_PC);
	}
}

void MZone::cpyPFPCSumCUDA(cudaStream_t **sts, int streamN)
{
	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		cudaMemcpyAsync(&inputSumPFPCMZH[NUM_PC * i / numGPUs], inputSumPFPCMZGPU[i],
				NUM_PC / numGPUs * sizeof(float), cudaMemcpyDeviceToHost, sts[i][streamN]);
	}
}

void MZone::runPFPCPlastCUDA(cudaStream_t **sts, int streamN, unsigned long t)
{
	cudaError_t error;
	if (t % ap.tsPerHistBinGR == 0)
	{
		int curGROffset;
		int curGPUInd;
		int curIOInd;

		int numGRPerIO;

		curGROffset = 0;
		curGPUInd   = 0;
		curIOInd    = 0;

		numGRPerIO = NUM_GR / NUM_IO;

		for (int i = 0; i < NUM_IO; i++)
		{
			if (as->pfPCPlastTimerIO[i] < (ap.tsLTDStartAPIO + ((int)(ap.tsLTDDurationIO))) &&
					as->pfPCPlastTimerIO[i] >= ap.tsLTDStartAPIO)
			{
				pfPCPlastStepIO[i] = tempGRPCLTDStep;
			}
			else if (as->pfPCPlastTimerIO[i] >= ap.tsLTPStartAPIO ||
					as->pfPCPlastTimerIO[i] < ap.tsLTPEndAPIO)
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
		for (int i = 0; i < NUM_GR; i += NUM_P_PC_FROM_GR_TO_PC)
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
					historyGRGPU[curGPUInd], ap.grPCHistCheckBinIO, curGROffset, pfPCPlastStepIO[curIOInd]);

			curGROffset += NUM_P_PC_FROM_GR_TO_PC;
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
	tempGRPCLTDStep = ap.synLTDStepSizeGRtoPC;
	tempGRPCLTPStep = ap.synLTPStepSizeGRtoPC;
}

const float* MZone::exportPFPCWeights()
{
	cpyPFPCSynWCUDA();
	return (const float *)pfSynWeightPCLinear; 
}

// Why not write one export function which takes in the weight you want to export?
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

const ct_uint32_t* MZone::exportAPBufIO()
{
	return (const ct_uint32_t *)as->apBufIO.get();
}

const ct_uint32_t* MZone::exportAPBufNC()
{
	return (const ct_uint32_t *)as->apBufNC.get();
}

void MZone::testReduction()
{
	cudaError_t error;
	cudaStream_t *sts = new cudaStream_t[numGPUs];

	float hostTestData[NUM_GR] = {0.0};
	float hostPCSum[NUM_PC] = {0.0};
	float hostBCSum[NUM_BC] = {0.0};
	float hostSCSum[NUM_SC] = {0.0};

	float gpuToHostPCSum[NUM_PC] = {0.0};
	float gpuToHostBCSum[NUM_BC] = {0.0};
	float gpuToHostSCSum[NUM_SC] = {0.0};

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

	for (int i = 0; i < NUM_GR; i++)
	{
		hostTestData[i] = randGen->Random();
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaStreamCreate(&sts[i]);

		cudaMallocPitch(&gpuPCTestData[i], &gpuPCP[i],
				NUM_P_PC_FROM_GR_TO_PC * sizeof(float), NUM_PC / numGPUs);
		cudaMallocPitch(&gpuBCTestData[i], &gpuBCP[i],
				NUM_P_BC_FROM_GR_TO_BC * sizeof(float), NUM_BC / numGPUs);
		cudaMallocPitch(&gpuSCTestData[i], &gpuSCP[i],
				NUM_P_SC_FROM_GR_TO_SC*sizeof(float), NUM_SC / numGPUs);

		cudaMalloc(&gpuPCSum[i], NUM_PC / numGPUs * sizeof(float));
		cudaMalloc(&gpuBCSum[i], NUM_BC / numGPUs * sizeof(float));
		cudaMalloc(&gpuSCSum[i], NUM_SC / numGPUs * sizeof(float));

		error = cudaGetLastError();
		std::cout << "allocating memory for gpu " << i << " " << cudaGetErrorString(error) << std::endl;

		cudaDeviceSynchronize();
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		for (int j = 0; j < NUM_PC / numGPUs; j++)
		{
		   cudaMemcpy(((char *)gpuPCTestData[i] + j * gpuPCP[i]),
				 &hostTestData[i * numGRPerGPU + j * NUM_P_PC_FROM_GR_TO_PC],
				 NUM_P_PC_FROM_GR_TO_PC * sizeof(float), cudaMemcpyHostToDevice);
		}

		for (int j = 0; j < NUM_BC / numGPUs; j++)
		{
		   cudaMemcpy(((char *)gpuBCTestData[i] + j * gpuBCP[i]),
				 &hostTestData[i * numGRPerGPU + j * NUM_P_BC_FROM_GR_TO_BC],
				 NUM_P_BC_FROM_GR_TO_BC * sizeof(float), cudaMemcpyHostToDevice);
		}

		for (int j = 0; j < NUM_SC / numGPUs; j++)
		{
		   cudaMemcpy(((char *)gpuSCTestData[i] + j * gpuSCP[i]),
				 &hostTestData[i * numGRPerGPU + j * NUM_P_SC_FROM_GR_TO_SC],
				 NUM_P_SC_FROM_GR_TO_SC * sizeof(float), cudaMemcpyHostToDevice);
		}

		error = cudaGetLastError();
		std::cout << "copying memory for gpu " << i << " " << cudaGetErrorString(error) << std::endl;

		cudaDeviceSynchronize();
	}

	for (int i = 0; i < NUM_PC; i++)
	{
		hostPCSum[i] = 0;

		for (int j = 0; j < NUM_P_PC_FROM_GR_TO_PC; j++)
		{
			hostPCSum[i] += hostTestData[i * NUM_P_PC_FROM_GR_TO_PC + j];
		}
	}

	for (int i = 0; i < NUM_BC; i++)
	{
		hostBCSum[i] = 0;

		for (int j = 0; j < NUM_P_BC_FROM_GR_TO_BC; j++)
		{
			hostBCSum[i] += hostTestData[i * NUM_P_BC_FROM_GR_TO_BC + j];
		}
	}

	for (int i = 0; i < NUM_SC; i++)
	{
		hostSCSum[i] = 0;

		for (int j = 0; j < NUM_P_SC_FROM_GR_TO_SC; j++)
		{
			hostSCSum[i] += hostTestData[i * NUM_P_SC_FROM_GR_TO_SC + j];
		}
	}

	for (int i = 0; i < numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);
		callSumKernel<float, true, false>(sts[i], gpuPCTestData[i], gpuPCP[i],
				gpuPCSum[i], 1, NUM_PC / numGPUs, 1, NUM_P_PC_FROM_GR_TO_PC);

		callSumKernel<float, true, false>(sts[i], gpuBCTestData[i], gpuBCP[i],
				gpuBCSum[i], 1, NUM_BC / numGPUs, 1, NUM_P_BC_FROM_GR_TO_BC);

		callSumKernel<float, true, false>(sts[i], gpuSCTestData[i], gpuSCP[i],
				gpuSCSum[i], 1, NUM_SC / numGPUs, 1, NUM_P_SC_FROM_GR_TO_SC);

		cudaDeviceSynchronize();

		error = cudaGetLastError();
		std::cout << "calling sum kernels for gpu " << i << " " << cudaGetErrorString(error) << std::endl;
	}

	for (int i = 0; i<numGPUs; i++)
	{
		cudaSetDevice(i + gpuIndStart);

		cudaMemcpy(&gpuToHostPCSum[i * NUM_PC / numGPUs], gpuPCSum[i],
			NUM_PC / numGPUs * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gpuToHostBCSum[i * NUM_BC / numGPUs], gpuBCSum[i],
			NUM_BC / numGPUs * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(&gpuToHostSCSum[i * NUM_SC / numGPUs], gpuSCSum[i],
			NUM_SC / numGPUs * sizeof(float), cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
	}

	std::cout << "NumPC per GPU: " << NUM_PC / numGPUs << std::endl <<
		"NumBC per GPU: " << NUM_BC / numGPUs << 
		"NUMSC per GPU: " << NUM_SC / numGPUs << std::endl;

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

