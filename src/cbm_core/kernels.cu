 /*
 * kernels.cu
 *
 *  Created on: Jun 6, 2011
 *      Author: consciousness
 */

#include <curand_kernel.h>
#include "kernels.h"

extern __shared__ uint32_t sharedIOBufGR[];
extern __shared__ float  sharedIOBufGRfloat[];

__global__ void testKernel(float *a, float *b, float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	c[i] = a[i] + b[i];
}

//**-----------------GR Kernels------------------**

__global__ void calcActivityGRGPU(float *vm, float *gKCa, float *gLeak, float *gNMDA,
	float *gNMDAInc, float *thresh, uint32_t *apBuf, uint8_t *apOutGR, uint32_t *apGR,
	int *apMFtoGR, float *gESum, float *gISum, float eLeak, float eGOIn, float gAMPAInc, 
	float threshBase, float threshMax, float threshDecay)
{
	float tempThresh;
	unsigned int tempAP;

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	float tempGKCa = gKCa[i];
	float tempV = vm[i];

	gLeak[i] = 0.0000001021370733 * tempV * tempV * tempV * tempV
	   		 + 0.00001636462 * tempV * tempV * tempV
			 + 0.00113971219 * tempV * tempV
			 + 0.038772 * tempV
			 + 0.6234929;
	
	
	gNMDAInc[i] = 0.00000011969 * tempV * tempV * tempV
	   			+ 0.000089369 * tempV * tempV
				+ 0.0151 * tempV
				+ 0.7713;

	gNMDA[i] = gNMDAInc[i] * gAMPAInc * apMFtoGR[i] + gNMDA[i] * 0.9672;


	tempV += gLeak[i] * (eLeak - tempV)
		   - gESum[i] * tempV 
		   - gNMDA[i] * tempV
		   + gISum[i] * (eGOIn - tempV); 

	if (tempV > threshMax) tempV = threshMax;

	tempThresh = thresh[i] + (threshBase - thresh[i]) * threshDecay;
	tempAP 	   = tempV > tempThresh;
	thresh[i]  = tempAP * threshMax + (!tempAP) * tempThresh;

	tempGKCa = tempGKCa * 0.9999f; 
	gKCa[i] = tempAP * (tempGKCa + 0.000f) + (!tempAP) * tempGKCa;

	apBuf[i]   = (apBuf[i] << 1) | tempAP;
	apOutGR[i] = tempAP;
	apGR[i]    = tempAP;
	vm[i]      = tempV;
}

__global__ void updateGRGOOutGPU(uint32_t *apBuf,
		uint32_t *goOut, size_t goOutPitch,
		uint32_t *delay, size_t delayPitch,
		uint32_t *con, size_t conPitch,
		int32_t *numSyn, int nWrites)
{
	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int *conRow;
	unsigned int *delayRow;
	unsigned int *goRow=(unsigned int *)((char *)goOut+blockIdx.x*goOutPitch);

	int tempNS=numSyn[index];
	unsigned int tempOut;

	for(int i=0; i<nWrites; i++)
	{
		sharedIOBufGR[tid+i*blockDim.x]=0;
	}

	__syncthreads();
	for(int i=0; i<tempNS; i++)
	{
		conRow=(uint32_t *)((char *)con+i*conPitch);
		delayRow=(uint32_t *)((char *)delay+i*delayPitch);

		tempOut=(apBuf[index]&delayRow[index])>0;

		if(tempOut>0)
		{
			atomicAdd(&sharedIOBufGR[conRow[index]], 1);
		}
	}
	__syncthreads();
	for(int i=0; i<nWrites; i++)
	{
		goRow[tid+i*blockDim.x]=sharedIOBufGR[tid+i*blockDim.x];
	}
}

__global__ void updateGRBCOutGPU(uint32_t *apBuf,
		uint32_t *bcOut, size_t bcOutPitch,
		uint32_t *delay, size_t delayPitch,
		uint32_t *con, size_t conPitch,
		int32_t *numSyn, int nWrites)
{
	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int *conRow;
	unsigned int *delayRow;
	unsigned int *bcRow=(unsigned int *)((char *)bcOut+blockIdx.x*bcOutPitch);

	int tempNS=numSyn[index];
	unsigned int tempOut;

	for(int i=0; i<nWrites; i++)
	{
		sharedIOBufGR[tid+i*blockDim.x]=0;
	}

	__syncthreads();
	for(int i=0; i<tempNS; i++)
	{
		conRow=(uint32_t *)((char *)con+i*conPitch);
		delayRow=(uint32_t *)((char *)delay+i*delayPitch);

		tempOut=(apBuf[index]&delayRow[index])>0;

		if(tempOut>0)
		{
			atomicAdd(&sharedIOBufGR[conRow[index]], 1);
		}
	}
	__syncthreads();
	for(int i=0; i<nWrites; i++)
	{
		bcRow[tid+i*blockDim.x]=sharedIOBufGR[tid+i*blockDim.x];
	}
}

__global__ void sumGRGOOutGPU(unsigned int nRows, uint32_t *goOut, size_t goOutPitch, uint32_t *goOutSum)
{
	unsigned int *goOutRow;
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempSum;

	tempSum=0;
	for(int i=0; i<nRows; i++)
	{
		goOutRow=(unsigned int *)((char *)goOut+i*goOutPitch);

		tempSum+=goOutRow[index];
	}

	goOutSum[index]=tempSum;
	//goOutSum[index]=1;
}

__global__ void sumGRBCOutGPU(unsigned int nRows, uint32_t *bcOut, size_t bcOutPitch, uint32_t *bcOutSum)
{
	unsigned int *bcOutRow;
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempSum;

	tempSum=0;
	for(int i=0; i<nRows; i++)
	{
		bcOutRow=(unsigned int *)((char *)bcOut+i*bcOutPitch);

		tempSum+=bcOutRow[index];
	}

	bcOutSum[index]=tempSum;
	//bcOutSum[index]=1;
}

__global__ void updateGRInOPGPU(unsigned int inNLoads, uint32_t *apIn, float *dynamicSpillAmp,
		float *g, size_t gPitch, uint32_t *conFromIn, size_t conFromInPitch, int32_t *numInPerGR,
		float *gSum, float *gDirect, float *gSpillover,  float gDecayD, float gIncD, float gDecayS,
		float gIncFracS)
{
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int *conRow;

	int tempNSyn = numInPerGR[index];

	int tempApInSum = 0;

	for (int i = 0; i < inNLoads; i++)
	{
		sharedIOBufGR[tid + i * blockDim.x] = apIn[tid + i * blockDim.x];
	}
	__syncthreads();

	for (int i = 0; i < tempNSyn; i++)
	{
		conRow = (unsigned int *)((char *)conFromIn + i * conFromInPitch);
		tempApInSum += sharedIOBufGR[conRow[index]];
	}

	gDirect[index] = gDirect[index] * gDecayD + gIncD * tempApInSum;
	gSpillover[index] = gSpillover[index] * 0.99 + dynamicSpillAmp[index] * tempApInSum;

	gSum[index] = gDirect[index] + gSpillover[index]; 
}

__global__ void updateGOGRDepressionInOPGPU(unsigned int inNLoads, float *depAmp, uint32_t *conFromIn,
	  size_t conFromInPitch, int32_t *numInPerGR, float *depAmpGOGRGPU)
{
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int *conRow;

	int tempNSyn = numInPerGR[index];

	float tempDepAmpSum = 0;
	for (int i = 0; i < inNLoads; i++)
	{
		sharedIOBufGRfloat[tid + i * blockDim.x] = depAmp[tid + i * blockDim.x];
	}
	__syncthreads();

	for (int i = 0; i < tempNSyn; i++)
	{
		conRow = (unsigned int *)((char *) conFromIn + i * conFromInPitch);
		tempDepAmpSum += sharedIOBufGRfloat[conRow[index]];
	}
	depAmpGOGRGPU[index] = tempDepAmpSum / 3;
}

__global__ void updateMFGRDepressionInOPGPU(unsigned int inNLoads, float *depAmp, uint32_t *conFromIn,
	  size_t conFromInPitch, int32_t *numInPerGR, int *numMFperGR, float *depAmpMFGRGPU)
{
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + tid;

	unsigned int *conRow;

	int tempNSyn = numInPerGR[index];

	float tempDepAmpSum = 0;
	for (int i = 0; i < inNLoads; i++)
	{
		sharedIOBufGRfloat[tid + i * blockDim.x] = depAmp[tid + i * blockDim.x];
	}
	__syncthreads();

	for (int i = 0; i < tempNSyn; i++)
	{
		conRow = (unsigned int *)((char *)conFromIn + i * conFromInPitch);
		tempDepAmpSum += sharedIOBufGRfloat[conRow[index]];
	}
	depAmpMFGRGPU[index] = tempDepAmpSum / numMFperGR[index];
}

__global__ void updateGOGRDynamicSpillInOPGPU(unsigned int inNLoads, float *dynamicAmp,
		uint32_t *conFromIn, size_t conFromInPitch,
		int32_t *numInPerGR, float *dynamicAmpGOGRGPU)
{

	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	unsigned int *conRow;

	int tempNSyn=numInPerGR[index];

	float tempDynamicAmpSum=0;
	for(int i=0; i<inNLoads; i++)
	{
		sharedIOBufGRfloat[tid+i*blockDim.x]=dynamicAmp[tid+i*blockDim.x];
	}
	__syncthreads();
	

	for(int i=0; i<tempNSyn; i++)
	{
		conRow=(unsigned int *)((char *)conFromIn+i*conFromInPitch);
		tempDynamicAmpSum+=sharedIOBufGRfloat[conRow[index]];	
	}

	dynamicAmpGOGRGPU[index] = tempDynamicAmpSum/3;
}

__global__ void updateUBCGRInOPGPU(unsigned int inNLoads, uint32_t *apIn, float *depAmp,
		float *g, size_t gPitch,
		uint32_t *conFromIn, size_t conFromInPitch,
		int32_t *numInPerGR, int *apUBCtoGRp, float *gSum, float *gDirect, float *gSpillover, 
		float gDecayD, float gIncD, float gDecayS, float gIncFracS)
{
	int tid=threadIdx.x;
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	unsigned int *conRow;

	int tempNSyn=numInPerGR[index];

	int tempApInSum=0;
	for(int i=0; i<inNLoads; i++)
	{
		sharedIOBufGR[tid+i*blockDim.x]=apIn[tid+i*blockDim.x];
	}
	
	__syncthreads();
	

	for(int i=0; i<tempNSyn; i++)
	{
		conRow=(unsigned int *)((char *)conFromIn+i*conFromInPitch);
		tempApInSum+=sharedIOBufGR[conRow[index]];	
	}
	/* what gives rise to nans in depAmp? => re-think how modifying to eliminate branch */
	/* (also we don't work with ubcs currently 08/05/2022) */
	if( isnan(depAmp[index]) )
	{
		gDirect[index] = 0;
		gSpillover[index] = 0;

		apUBCtoGRp[index] = 0;
	}
	else{
		
		gDirect[index] = gDirect[index]*gDecayD + gIncD*(tempApInSum)*depAmp[index];
		gSpillover[index] = gSpillover[index]*gDecayS + gIncD*gIncFracS*(tempApInSum)*depAmp[index];

		apUBCtoGRp[index] = tempApInSum;
	}
	gSum[index] = gDirect[index] + gSpillover[index];
}

__global__ void updateMFGRInOPGPU(unsigned int inNLoads, uint32_t *apIn, float*depAmp,
		float *g, size_t gPitch, uint32_t *conFromIn, size_t conFromInPitch,
		int32_t *numInPerGR, int *apMFtoGR, float *gSum, float *gDirect, float *gSpillover, 
		float gDecayD, float gIncD, float gDecayS, float gIncFracS)
{
	int tid = threadIdx.x;
	int index = blockIdx.x*blockDim.x+threadIdx.x;

	unsigned int *conRow;

	int tempNSyn = numInPerGR[index];

	int tempApInSum = 0;
	for (int i = 0; i < inNLoads; i++)
	{
		sharedIOBufGR[tid + i * blockDim.x] = apIn[tid + i * blockDim.x];
	}
	__syncthreads();

	for (int i = 0; i < tempNSyn; i++)
	{
		conRow = (unsigned int *)((char *)conFromIn + i *conFromInPitch);
		tempApInSum += sharedIOBufGR[conRow[index]];	
	}

	gDirect[index] = gDirect[index] * gDecayD + gIncD * tempApInSum * depAmp[index];
	gSpillover[index] = gSpillover[index] * gDecayS + gIncD * gIncFracS * tempApInSum * depAmp[index];

	gSum[index] = gDirect[index] + gSpillover[index];
	apMFtoGR[index] = tempApInSum;
}

__global__ void updateGRHistory(uint32_t *apBuf, uint64_t *apHist, uint32_t bufTestMask)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	uint64_t tempHist=apHist[i]<<1;
	apHist[i]=tempHist|((apBuf[i]&bufTestMask)>0)*0x00000001;
}

__global__ void updatePFBCSCOutGPU(uint32_t *apBuf, uint32_t *delay,
		uint32_t *pfBC, size_t pfBCPitch, unsigned int numPFInPerBC, unsigned int numPFInPerBCP2,
		uint32_t *pfSC, size_t pfSCPitch, unsigned int numPFInPerSC, unsigned int numPFInPerSCP2)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	uint32_t tempOut;
	unsigned int *pfBCRow=(uint32_t *)((char *)pfBC+(index>>numPFInPerBCP2)*pfBCPitch);
	unsigned int *pfSCRow=(uint32_t *)((char *)pfSC+(index>>numPFInPerSCP2)*pfSCPitch);

	tempOut=(apBuf[index]&delay[index])>0;

	pfBCRow[index&(numPFInPerBC-1)]=tempOut;
	pfSCRow[index&(numPFInPerSC-1)]=tempOut;
}

__global__ void updatePFPCOutGPU(uint32_t *apBuf, uint32_t *delay,
		float *synWeight, float *pfPC, size_t pfPCPitch, unsigned int numPFInPerPC, unsigned int numPFInPerPCP2)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int tempOut;
	float *pfPCRow=(float *)((char *)pfPC+(index>>numPFInPerPCP2)*pfPCPitch);

	tempOut=(apBuf[index]&delay[index])>0;

	pfPCRow[index&(numPFInPerPC-1)]=synWeight[index]*tempOut;
}

//**---------------end GR Kernels-------------------**


//**---------------IO kernels-----------------**

template <typename randState>
__global__ void updatePFPCBinarySynWeightKernel(float *synWPFPC, uint64_t *historyGR, uint64_t plastCheckMask,
		unsigned int offset, float plastStep, float synWLow, float synWHigh, float trans_prob, float *randoms)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
	if (randoms[i] < trans_prob)
	{
		synWPFPC[i] += ((historyGR[i] & plastCheckMask) > 0) * plastStep;

		synWPFPC[i] = (synWPFPC[i] > synWLow) * synWPFPC[i] + (synWPFPC[i] <=synWLow) * synWLow;
		synWPFPC[i] = (synWPFPC[i] > synWHigh) * synWHigh + (synWPFPC[i] <= synWHigh) * synWPFPC[i];
	}
}

template <typename randState>
__global__ void updatePFPCAbbottCascadeLTDPlastKernel(float *synWPFPC, uint8_t *synStatesPFPC, uint64_t *historyGPU,
		uint64_t plastCheckMask, unsigned int offset, float synWLow, float trans_prob_base, float *randoms)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
	if ((historyGPU[i] & plastCheckMask) > 0)
	{
		switch (synStatesPFPC[i])
		{
			case 1:
				if (randoms[i] < trans_prob_base / 8.0)
				{
					synStatesPFPC[i] = 0;
				}
				break;
			case 2:
				if (randoms[i] < trans_prob_base / 4.0)
				{
					synStatesPFPC[i] = 1;
				}
				break;
			case 3:
				if (randoms[i] < trans_prob_base / 2.0)
				{
					synStatesPFPC[i] = 2;
				}
				break;
			case 4:
				if (randoms[i] < trans_prob_base)
				{
					synStatesPFPC[i] = 3;
					synWPFPC[i] = synWLow;
				}
				break;
			case 5:
				if (randoms[i] < trans_prob_base / 2.0)
				{
					synStatesPFPC[i] = 3;
					synWPFPC[i] = synWLow;
				}
				break;
			case 6:
				if (randoms[i] < trans_prob_base / 4.0)
				{
					synStatesPFPC[i] = 3;
					synWPFPC[i] = synWLow;
				}
				break;
			case 7:
				if (randoms[i] < trans_prob_base / 8.0)
				{
					synStatesPFPC[i] = 3;
					synWPFPC[i] = synWLow;
				}
				break;
		}
	}
}

template <typename randState>
__global__ void updatePFPCAbbottCascadeLTPPlastKernel(float *synWPFPC, uint8_t *synStatesPFPC, uint64_t *historyGPU,
	  uint64_t plastCheckMask, unsigned int offset, float synWHigh, float trans_prob_base, float *randoms)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
	if ((historyGPU[i] & plastCheckMask) > 0)
	{
		switch (synStatesPFPC[i])
		{
			case 0:
				if (randoms[i] < trans_prob_base / 8.0)
				{
					synStatesPFPC[i] = 4;
					synWPFPC[i] = synWHigh;
				}
				break;
			case 1:
				if (randoms[i] < trans_prob_base / 4.0)
				{
					synStatesPFPC[i] = 4;
					synWPFPC[i] = synWHigh;
				}
				break;
			case 2:
				if (randoms[i] < trans_prob_base / 2.0)
				{
					synStatesPFPC[i] = 4;
					synWPFPC[i] = synWHigh;
				}
				break;
			case 3:
				if (randoms[i] < trans_prob_base)
				{
					synStatesPFPC[i] = 4;
					synWPFPC[i] = synWHigh;
				}
				break;
			case 4:
				if (randoms[i] < trans_prob_base / 2.0)
				{
					synStatesPFPC[i] = 5;
				}
				break;
			case 5:
				if (randoms[i] < trans_prob_base / 4.0)
				{
					synStatesPFPC[i] = 6;
				}
				break;
			case 6:
				if (randoms[i] < trans_prob_base / 8.0)
				{
					synStatesPFPC[i] = 7;
				}
				break;
		}
	}
}

template <typename randState>
__global__ void updatePFPCMaukCascadeLTDPlastKernel(float *synWPFPC, uint8_t *synStatesPFPC, uint64_t *historyGPU,
		uint64_t plastCheckMask, unsigned int offset, float synWLow, float trans_prob_base, float *randoms)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
	if ((historyGPU[i] & plastCheckMask) > 0)
	{
		switch (synStatesPFPC[i])
		{
			case 1:
				if (randoms[i] < trans_prob_base / 8.0)
				{
					synStatesPFPC[i] = 0;
				}
				break;
			case 2:
				if (randoms[i] < trans_prob_base / 4.0)
				{
					synStatesPFPC[i] = 1;
				}
				break;
			case 3:
				if (randoms[i] < trans_prob_base / 2.0)
				{
					synStatesPFPC[i] = 2;
				}
				break;
			case 4:
				if (randoms[i] < trans_prob_base)
				{
					synStatesPFPC[i] = 3;
					synWPFPC[i] = synWLow;
				}
				break;
			case 5:
				if (randoms[i] < trans_prob_base / 2.0)
				{
					synStatesPFPC[i] = 4;
				}
				break;
			case 6:
				if (randoms[i] < trans_prob_base / 4.0)
				{
					synStatesPFPC[i] = 5;
				}
				break;
			case 7:
				if (randoms[i] < trans_prob_base / 8.0)
				{
					synStatesPFPC[i] = 6;
				}
				break;
		}
	}
}

template <typename randState>
__global__ void updatePFPCMaukCascadeLTPPlastKernel(float *synWPFPC, uint8_t *synStatesPFPC, uint64_t *historyGPU,
	  uint64_t plastCheckMask, unsigned int offset, float synWHigh, float trans_prob_base, float *randoms)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
	if ((historyGPU[i] & plastCheckMask) > 0)
	{
		switch (synStatesPFPC[i])
		{
			case 0:
				if (randoms[i] < trans_prob_base / 8.0)
				{
					synStatesPFPC[i] = 1;
				}
				break;
			case 1:
				if (randoms[i] < trans_prob_base / 4.0)
				{
					synStatesPFPC[i] = 2;
				}
				break;
			case 2:
				if (randoms[i] < trans_prob_base / 2.0)
				{
					synStatesPFPC[i] = 3;
				}
				break;
			case 3:
				if (randoms[i] < trans_prob_base)
				{
					synStatesPFPC[i] = 4;
					synWPFPC[i] = synWHigh;
				}
				break;
			case 4:
				if (randoms[i] < trans_prob_base / 2.0)
				{
					synStatesPFPC[i] = 5;
				}
				break;
			case 5:
				if (randoms[i] < trans_prob_base / 4.0)
				{
					synStatesPFPC[i] = 6;
				}
				break;
			case 6:
				if (randoms[i] < trans_prob_base / 8.0)
				{
					synStatesPFPC[i] = 7;
				}
				break;
		}
	}
}

__global__ void updatePFPCGradedSynWeightKernel(float *synWPFPC, uint64_t *historyGR, uint64_t plastCheckMask,
		unsigned int offset, float plastStep)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x+offset;
	synWPFPC[i]=synWPFPC[i]+((historyGR[i]&plastCheckMask)>0)*plastStep;

	synWPFPC[i]=(synWPFPC[i]>0)*synWPFPC[i];
	synWPFPC[i]=(synWPFPC[i]>1)+(synWPFPC[i]<=1)*synWPFPC[i];
}

__global__ void updatePFPCSTPKernel(uint32_t use_cs, uint32_t use_us, float grEligBase, float grEligMax,
	float grEligExpScale, float grEligDecay, float grStpDecay, float grStpInc, float *grEligGPU, float *pfpcSTPsGPU,
	uint32_t *apBufGPU, uint32_t *delayMaskGPU)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t apGR = (apBufGPU[index] & delayMaskGPU[index]) > 0;

	grEligGPU[index] = apGR * grEligGPU[index] * grEligExpScale
		+ (1 - apGR) * (grEligGPU[index] - (grEligGPU[index] - grEligBase) * grEligDecay);

	// ensure the eligibility is greater than grEligBase
	grEligGPU[index] = (grEligGPU[index] < grEligBase) * grEligBase + (grEligGPU[index] >= grEligBase) * grEligGPU[index];

	// rule for inducing stp
	if (grEligGPU[index] > grEligMax)
	{
		pfpcSTPsGPU[index] += grStpInc;
		grEligGPU[index] = grEligBase;
	}

	// special stp decay rule: only do so during background trials
	if (use_cs == 0 && use_us == 0)
	{
		pfpcSTPsGPU[index] *= grStpDecay;
	}
}

//**---------------end IO kernels-------------**


//**---------------common kernels-------------**

template <typename Type, unsigned int blockSize, unsigned int sDataSize, bool inMultiPitch, bool outMultiPitch>
__global__ void sumInputsNew(Type *input, unsigned int inputPitch,
		Type *output, unsigned int outputPitch, unsigned int rowLength)
{
		__shared__ Type sData[sDataSize];

	int tid=threadIdx.x;
	int index=blockIdx.x*(blockSize*2)+tid;
	int gridSize=blockSize*2*gridDim.x;
	Type *inputRow;

	Type tempSum=0;

	if(inMultiPitch)
	{
		inputRow=(Type *)((char *)input+blockIdx.y*inputPitch);
	}
	else
	{
		inputRow=input+blockIdx.y;
	}

	while(index<rowLength)
	{
		tempSum+=inputRow[index]+inputRow[index+blockSize];
		index+=gridSize;
	}
	sData[tid]=tempSum;
	__syncthreads();

	if(blockSize>=512)
	{
		if(tid<256)
			sData[tid]+=sData[tid+256];
		__syncthreads();
	}

	if(blockSize>=256)
	{
		if(tid<128)
			sData[tid]+=sData[tid+128];
		__syncthreads();
	}

	if(blockSize>=128)
	{
		if(tid<64)
			sData[tid]+=sData[tid+64];
		__syncthreads();
	}

	if(tid<32)
	{
		volatile Type* sMem = sData;
		if(blockSize>=64)
			sMem[tid]+=sMem[tid+32];
		if(blockSize>=32)
			sMem[tid]+=sMem[tid+16];
		if(blockSize>=16)
			sMem[tid]+=sMem[tid+8];
		if(blockSize>=8)
			sMem[tid]+=sMem[tid+4];
		if(blockSize>=4)
			sMem[tid]+=sMem[tid+2];
		if(blockSize>=2)
			sMem[tid]+=sMem[tid+1];
	}
	if(tid==0)
	{
		Type *outputRow;
		if(outMultiPitch)
		{
			outputRow=(Type *)((char *)output+blockIdx.y*outputPitch);
		}
		else
		{
			outputRow=output+blockIdx.y;
		}
		outputRow[blockIdx.x]=sData[0];
	}
}

template<typename Type>
__global__ void broadcastValue(Type *val, Type *outArr)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;

	outArr[i]=*val;
}

//**---------------end common kernels---------**


//**---------------random kernels---------**

template <typename randState>
__global__ void curandSetupKernel(randState *state, uint32_t seed)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* every thread gets same seed, different sequence number,
	   no offset */
	curand_init(seed, id, 0, &state[id]);
}

template <typename randState>
__global__ void curandGenerateUniformsKernel(randState *state, float *randoms, uint32_t rand_offset)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	curandStateMRG32k3a localState = state[i];
	randoms[i + rand_offset] = curand_uniform(&localState);
	state[i] = localState;
}

//**---------------end random kernels---------**


//**---------------kernel calls---------------**

void callTestKernel(cudaStream_t &st, float *a, float *b, float *c)
{
	testKernel<<<1, 128>>>(a, b, c);
}

void callGRActKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *vGPU, float *gKCaGPU, float *gLeakGPU, float *gNMDAGRGPU, float *gNMDAIncGRGPU,
		float *threshGPU, uint32_t *apBufGPU, uint8_t *apOutGRGPU, uint32_t *apGRGPU,
		int *apMFtoGRGPU, float *gESumGPU, float *gISumGPU, float eLeak,
		float eGOIn, float gAMPAInc, float threshBase, float threshMax, float threshDecay)
{
	calcActivityGRGPU<<<numBlocks, numGRPerBlock, 0, st>>>(vGPU, gKCaGPU, gLeakGPU, gNMDAGRGPU,
		  gNMDAIncGRGPU, threshGPU, apBufGPU, apOutGRGPU, apGRGPU, apMFtoGRGPU, gESumGPU, 
		  gISumGPU, eLeak, eGOIn, gAMPAInc, threshBase, threshMax, threshDecay);
}

template<typename Type, bool inMultiP, bool outMultiP>
void callSumKernel(cudaStream_t &st, Type *inGPU, size_t inGPUP, Type *outSumGPU, size_t outSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength)
{
	unsigned int numElementsPerBlock;
	dim3 dimGrid(nOutCols, nOutCells);

	numElementsPerBlock=rowLength/nOutCols;

	if(numElementsPerBlock>=2048)
	{
		sumInputsNew<Type, 512, 512, inMultiP, outMultiP><<<dimGrid, 512, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
	else if(numElementsPerBlock>=512)
	{
		sumInputsNew<Type, 128, 128, inMultiP, outMultiP><<<dimGrid, 128, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
	else if(numElementsPerBlock>=128)
	{
		sumInputsNew<Type, 32, 64, inMultiP, outMultiP><<<dimGrid, 32, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
	else if(numElementsPerBlock>=32)
	{
		sumInputsNew<Type, 8, 64, inMultiP, outMultiP><<<dimGrid, 8, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
	else
	{
		sumInputsNew<Type, 2, 64, inMultiP, outMultiP><<<dimGrid, 2, 0, st>>>
				(inGPU, inGPUP, outSumGPU, outSumGPUP, rowLength);
	}
}

template<typename Type>
void callBroadcastKernel(cudaStream_t &st, Type *broadcastVal, Type *outArray,
		unsigned int nBlocks, unsigned int rowLength)
{
	broadcastValue<Type><<<nBlocks, rowLength/nBlocks, 0, st>>>(broadcastVal, outArray);
}

template <typename randState, typename blockDims, typename threadDims>
void callCurandSetupKernel(cudaStream_t &st, randState *state, uint32_t seed,
						   blockDims &block_dim, threadDims &thread_dim)
{
	curandSetupKernel<randState><<<block_dim, thread_dim, 0, st>>>(state, seed);
}

template <typename randState>
void callCurandGenerateUniformKernel(cudaStream_t &st, randState *state, uint32_t block_dim,
	  uint32_t thread_dim, float *randoms, uint32_t rand_offset)
{
	curandGenerateUniformsKernel<randState><<<block_dim, thread_dim, 0, st>>>(state, randoms, rand_offset);
}

void callSumGRGOOutKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGOPerBlock,
		unsigned int numGROutRows, uint32_t *grInGOGPU,  size_t grInGOGPUPitch, uint32_t *grInGOSGPU)
{
	sumGRGOOutGPU<<<numBlocks, numGOPerBlock, 0, st>>>(numGROutRows, grInGOGPU, grInGOGPUPitch, grInGOSGPU);
}
void callSumGRBCOutKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numBCPerBlock,
		unsigned int numGROutRows, uint32_t *grInBCGPU,  size_t grInBCGPUPitch, uint32_t *grInBCSGPU)
{
	sumGRBCOutGPU<<<numBlocks, numBCPerBlock, 0, st>>>(numGROutRows, grInBCGPU, grInBCGPUPitch, grInBCSGPU);
}

void callUpdateInGROPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		unsigned int numInCells, uint32_t *apInGPU, float *dynamicAmpGPU, float *gGPU, size_t gGPUP,
		uint32_t *conInGRGPU, size_t conInGRGPUP,
		int32_t *numInPerGRGPU, float *gSumGPU, float *gDirectGPU, float *gSpilloverGPU, 
		float gDecayD, float gIncD, float gDecayS, float gIncFracS)
{
	updateGRInOPGPU<<<numBlocks, numGRPerBlock, numInCells*sizeof(uint32_t), st>>>
			(numInCells/numGRPerBlock, apInGPU, dynamicAmpGPU, 
			gGPU, gGPUP, conInGRGPU, conInGRGPUP, numInPerGRGPU,
			gSumGPU, gDirectGPU, gSpilloverGPU, 
			gDecayD, gIncD, gDecayS, gIncFracS);
}

void callUpdateMFInGRDepressionOPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
	 unsigned int numInCells, float *depAmpGPU, uint32_t *conInGRGPU, size_t conInGRGPUP, 
	 int32_t *numInPerGRGPU, int *numMFperGR, float *depAmpMFGRGPU)
{
	updateMFGRDepressionInOPGPU<<<numBlocks, numGRPerBlock, numInCells * sizeof(uint32_t), st>>>
	   (numInCells / numGRPerBlock, depAmpGPU, conInGRGPU, conInGRGPUP, numInPerGRGPU, numMFperGR, depAmpMFGRGPU);
}

void callUpdateGOInGRDepressionOPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
	  unsigned int numInCells, float *depAmpGPU, uint32_t *conInGRGPU, size_t conInGRGPUP, 
	  int32_t *numInPerGRGPU, float *depAmpGOGRGPU)
{
	updateGOGRDepressionInOPGPU<<<numBlocks, numGRPerBlock, numInCells * sizeof(uint32_t), st>>>
	   (numInCells / numGRPerBlock, depAmpGPU, conInGRGPU, conInGRGPUP, numInPerGRGPU, depAmpGOGRGPU);
}

void callUpdateGOInGRDynamicSpillOPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		unsigned int numInCells, float *dynamicAmpGPU,
		uint32_t *conInGRGPU, size_t conInGRGPUP,
		int32_t *numInPerGRGPU, float *dynamicAmpGOGRGPU)
{
	updateGOGRDynamicSpillInOPGPU<<<numBlocks, numGRPerBlock, numInCells*sizeof(uint32_t), st>>>
			(numInCells/numGRPerBlock, dynamicAmpGPU, conInGRGPU, conInGRGPUP, numInPerGRGPU, dynamicAmpGOGRGPU);
}

void callUpdateUBCInGROPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		unsigned int numInCells, uint32_t *apInGPU, float *depAmpGPU, float *gGPU, size_t gGPUP,
		uint32_t *conInGRGPU, size_t conInGRGPUP,
		int32_t *numInPerGRGPU, int *apUBCtoGRGPU, float *gSumGPU, float *gDirectGPU, float *gSpilloverGPU,  
		float gDecayDirect, float gIncDirect, float gDecaySpill, float gIncFracSpill)
{
	updateUBCGRInOPGPU<<<numBlocks, numGRPerBlock, numInCells*sizeof(uint32_t), st>>>
			(numInCells/numGRPerBlock, apInGPU, depAmpGPU,  gGPU, gGPUP, conInGRGPU, conInGRGPUP, numInPerGRGPU,
			apUBCtoGRGPU, gSumGPU, gDirectGPU, gSpilloverGPU, 
			gDecayDirect, gIncDirect, gDecaySpill, gIncFracSpill);
}

void callUpdateMFInGROPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		unsigned int numInCells, uint32_t *apInGPU, float *depAmp, float *gGPU, size_t gGPUP,
		uint32_t *conInGRGPU, size_t conInGRGPUP,
		int32_t *numInPerGRGPU, int *apMFtoGRGPU, float *gSumGPU, float *gDirectGPU, float *gSpilloverGPU,  
		float gDecayDirect, float gIncDirect, float gDecaySpill, float gIncFracSpill)
{
	updateMFGRInOPGPU<<<numBlocks, numGRPerBlock, numInCells*sizeof(uint32_t), st>>>
			(numInCells/numGRPerBlock, apInGPU, depAmp, gGPU, gGPUP, conInGRGPU, conInGRGPUP, numInPerGRGPU,
			apMFtoGRGPU, gSumGPU, gDirectGPU, gSpilloverGPU, 
			gDecayDirect, gIncDirect, gDecaySpill, gIncFracSpill);
}


void callUpdatePFBCSCOutKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		uint32_t *apBufGPU, uint32_t *delayMaskGPU,
		uint32_t *inPFBCGPU, size_t inPFBCGPUPitch, unsigned int numPFInPerBCP2,
		uint32_t *inPFSCGPU, size_t inPFSCGPUPitch, unsigned int numPFInPerSCP2)
{
	updatePFBCSCOutGPU<<<numBlocks, numGRPerBlock, 0, st>>>(apBufGPU, delayMaskGPU,
			inPFBCGPU, inPFBCGPUPitch, 1<<numPFInPerBCP2, numPFInPerBCP2,
			inPFSCGPU, inPFSCGPUPitch, 1<<numPFInPerSCP2, numPFInPerSCP2);
}

void callUpdatePFPCOutKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		uint32_t *apBufGPU, uint32_t *delayMaskGPU,
		float *pfPCSynWGPU, float *inPFPCGPU, size_t inPFPCGPUPitch, unsigned int numPFInPerPCP2)
{
	updatePFPCOutGPU<<<numBlocks, numGRPerBlock, 0, st>>>(apBufGPU, delayMaskGPU, pfPCSynWGPU,
			inPFPCGPU, inPFPCGPUPitch, 1<<numPFInPerPCP2, numPFInPerPCP2);
}

void callUpdateGROutGOKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock, unsigned int numGO,
		uint32_t *apBufGPU, uint32_t *grInGOGPU, uint32_t grInGOGPUPitch,
		uint32_t *delayMasksGPU, uint32_t delayMasksGPUPitch,
		uint32_t *conGRtoGOGPU, size_t conGRtoGOGPUPitch,
		int32_t *numGOPerGRGPU)
{
	updateGRGOOutGPU<<<numBlocks, numGRPerBlock, numGO*sizeof(uint32_t), st>>>(apBufGPU, grInGOGPU, grInGOGPUPitch,
			delayMasksGPU, delayMasksGPUPitch, conGRtoGOGPU, conGRtoGOGPUPitch, numGOPerGRGPU, numGO/numGRPerBlock);
}

void callUpdateGROutBCKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock, unsigned int numBC,
		uint32_t *apBufGPU, uint32_t *grInBCGPU, uint32_t grInBCGPUPitch,
		uint32_t *delayMasksGPU, uint32_t delayMasksGPUPitch,
		uint32_t *conGRtoBCGPU, size_t conGRtoBCGPUPitch,
		int32_t *numBCPerGRGPU)
{
	updateGRBCOutGPU<<<numBlocks, numGRPerBlock, numBC*sizeof(uint32_t), st>>>(apBufGPU, grInBCGPU, grInBCGPUPitch,
		delayMasksGPU, delayMasksGPUPitch, conGRtoBCGPU, conGRtoBCGPUPitch, numBCPerGRGPU, numBC/numGRPerBlock);
}

void callUpdateGRHistKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		uint32_t *apBufGPU, uint64_t *historyGPU, uint32_t apBufGRHistMask)
{
		updateGRHistory<<<numBlocks, numGRPerBlock, 0, st>>>(apBufGPU, historyGPU, apBufGRHistMask);
}

template <typename randState>
void callPFPCBinaryPlastKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *synWeightGPU, uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float pfPCPlastStep, float synWLow, float synWHigh, float trans_prob, float *randoms)
{
	uint64_t mask = ((uint64_t)1)<<(pastBinNToCheck-1);
	updatePFPCBinarySynWeightKernel<randState><<<numBlocks, numGRPerBlock, 0, st>>>(synWeightGPU, historyGPU,
		mask, offSet, pfPCPlastStep, synWLow, synWHigh, trans_prob, randoms);
}

template<typename randState>
void callPFPCAbbottCascadeLTDPlastKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *synWeightGPU, uint8_t *synStatesGPU, uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float synWLow, float trans_prob_base, float *randoms)
{
	uint64_t mask = ((uint64_t)1)<<(pastBinNToCheck-1);
	updatePFPCAbbottCascadeLTDPlastKernel<randState><<<numBlocks, numGRPerBlock, 0, st>>>(synWeightGPU, synStatesGPU,
		historyGPU, mask, offSet, synWLow, trans_prob_base, randoms);
}

template<typename randState>
void callPFPCAbbottCascadeLTPPlastKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *synWeightGPU, uint8_t *synStatesGPU, uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float synWHigh, float trans_prob_base, float *randoms)
{
	uint64_t mask = ((uint64_t)1)<<(pastBinNToCheck-1);
	updatePFPCAbbottCascadeLTPPlastKernel<randState><<<numBlocks, numGRPerBlock, 0, st>>>(synWeightGPU, synStatesGPU,
		historyGPU, mask, offSet, synWHigh, trans_prob_base, randoms);
}

template<typename randState>
void callPFPCMaukCascadeLTDPlastKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *synWeightGPU, uint8_t *synStatesGPU, uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float synWLow, float trans_prob_base, float *randoms)
{
	uint64_t mask = ((uint64_t)1)<<(pastBinNToCheck-1);
	updatePFPCMaukCascadeLTDPlastKernel<randState><<<numBlocks, numGRPerBlock, 0, st>>>(synWeightGPU, synStatesGPU,
		historyGPU, mask, offSet, synWLow, trans_prob_base, randoms);
}

template<typename randState>
void callPFPCMaukCascadeLTPPlastKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *synWeightGPU, uint8_t *synStatesGPU, uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float synWHigh, float trans_prob_base, float *randoms)
{
	uint64_t mask = ((uint64_t)1)<<(pastBinNToCheck-1);
	updatePFPCMaukCascadeLTPPlastKernel<randState><<<numBlocks, numGRPerBlock, 0, st>>>(synWeightGPU, synStatesGPU,
		historyGPU, mask, offSet, synWHigh, trans_prob_base, randoms);
}

void callPFPCGradedPlastKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *synWeightGPU, uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float pfPCPlastStep)
{
	uint64_t mask = ((uint64_t)1)<<(pastBinNToCheck-1);
		updatePFPCGradedSynWeightKernel<<<numBlocks, numGRPerBlock, 0, st>>>(synWeightGPU, historyGPU,
				mask, offSet, pfPCPlastStep);
}

void callPFPCSTPKernel(cudaStream_t &st, uint32_t numBlocks, uint32_t numGRPerBlock, uint32_t use_cs, uint32_t use_us,
	float grEligBase, float grEligMax, float grEligExpScale, float grEligDecay, float grStpDecay, float grStpInc,
	float *grEligGPU, float *pfpcSTPsGPU, uint32_t *apBufGPU, uint32_t *delayMaskGPU)
{
	// TODO: write in numBlocks and numGRPerBlock vars
	updatePFPCSTPKernel<<<numBlocks, numGRPerBlock, 0, st>>>(use_cs, use_cs, grEligBase, grEligMax,
		  grEligExpScale, grEligDecay, grStpDecay, grStpInc, grEligGPU, pfpcSTPsGPU, apBufGPU, delayMaskGPU);
}
//**---------------end kernel calls------------**

// template initializations

template void callSumKernel<float, true, false>
		(cudaStream_t &st, float *inPFGPU, size_t inPFGPUP, float *outPFSumGPU, size_t outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);

template void callSumKernel<uint32_t, true, false>
		(cudaStream_t &st, uint32_t *inPFGPU, size_t inPFGPUP, uint32_t *outPFSumGPU, size_t outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);

template void callSumKernel<uint32_t, false, false>
		(cudaStream_t &st, uint32_t *inPFGPU, size_t inPFGPUP, uint32_t *outPFSumGPU, size_t outPFSumGPUP,
		unsigned int nOutCells, unsigned int nOutCols, unsigned int rowLength);

template void callBroadcastKernel<uint32_t>
(cudaStream_t &st, uint32_t *broadCastVal, uint32_t *outArray, unsigned int nBlocks, unsigned int rowLength);

template void callCurandSetupKernel<curandStateMRG32k3a, dim3, dim3>
(cudaStream_t &st, curandStateMRG32k3a *state, uint32_t seed, dim3 &block_dim, dim3 &thread_dim);

template void callCurandGenerateUniformKernel<curandStateMRG32k3a>(cudaStream_t &st, curandStateMRG32k3a *state,
		uint32_t block_dim, uint32_t thread_dim, float *randoms, uint32_t rand_offset);

template void callPFPCBinaryPlastKernel<curandStateMRG32k3a>(cudaStream_t &st, unsigned int numBlocks,
		unsigned int numGRPerBlock, float *synWeightGPU, uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float pfPCPlastStep, float synWLow, float synWHigh, float trans_prob, float *randoms);

template void callPFPCAbbottCascadeLTDPlastKernel<curandStateMRG32k3a>(cudaStream_t &st, unsigned int numBlocks,
		unsigned int numGRPerBlock, float *synWeightGPU, uint8_t *synStatesGPU, uint64_t *historyGPU,
		unsigned int pastBinNToCheck, int offSet, float synWLow, float trans_prob_base, float *randoms);

template void callPFPCAbbottCascadeLTPPlastKernel<curandStateMRG32k3a>(cudaStream_t &st, unsigned int numBlocks,
		unsigned int numGRPerBlock, float *synWeightGPU, uint8_t *synStatesGPU, uint64_t *historyGPU,
		unsigned int pastBinNToCheck, int offSet, float synWHigh, float trans_prob_base, float *randoms);

template void callPFPCMaukCascadeLTDPlastKernel<curandStateMRG32k3a>(cudaStream_t &st, unsigned int numBlocks,
		unsigned int numGRPerBlock, float *synWeightGPU, uint8_t *synStatesGPU, uint64_t *historyGPU,
		unsigned int pastBinNToCheck, int offSet, float synWLow, float trans_prob_base, float *randoms);

template void callPFPCMaukCascadeLTPPlastKernel<curandStateMRG32k3a>(cudaStream_t &st, unsigned int numBlocks,
		unsigned int numGRPerBlock, float *synWeightGPU, uint8_t *synStatesGPU, uint64_t *historyGPU,
		unsigned int pastBinNToCheck, int offSet, float synWHigh, float trans_prob_base, float *randoms);

