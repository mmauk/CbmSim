 /*
 * kernels.cu
 *
 *  Created on: Jun 6, 2011
 *      Author: consciousness
 */

#include "kernels.h"

extern __shared__ uint32_t sharedIOBufGR[];
extern __shared__ float  sharedIOBufGRfloat[];

extern __shared__ uint32_t sharedIOBufGO[];
extern __shared__ float sharedIOBufGOfloat[];

__global__ void testKernel(float *a, float *b, float *c)
 {
 	int i = blockIdx.x * blockDim.x + threadIdx.x;
 	c[i] = a[i] + b[i];
 }

/*
  Copyright (c) 2015-2021, Norbert Juffa
  All rights reserved.

  Redistribution and use in source and binary forms, with or without 
  modification, are permitted provided that the following conditions
  are met:

  1. Redistributions of source code must retain the above copyright 
     notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* this is meant to be used when we do not specify any compiler flags
 like -ftz=true or -use_fast_math when invoking nvcc. See this
 thread for further information:

https://forums.developer.nvidia.com/t/a-more-accurate-performance-competitive-implementation-of-expf/47528

*/
__device__ float my_expf(float a)
{
    float f, r, j, s, t;
    int i, ia;

    // exp(a) = 2**i * exp(f); i = rintf (a / log(2))
    j = fmaf (1.442695f, a, 12582912.f) - 12582912.f; // 0x1.715476p0, 0x1.8p23
    f = fmaf (j, -6.93145752e-1f, a); // -0x1.62e400p-1  // log_2_hi 
    f = fmaf (j, -1.42860677e-6f, f); // -0x1.7f7d1cp-20 // log_2_lo 
    i = (int)j;
    // approximate r = exp(f) on interval [-log(2)/2, +log(2)/2]
    r =             1.37805939e-3f;  // 0x1.694000p-10
    r = fmaf (r, f, 8.37312452e-3f); // 0x1.125edcp-7
    r = fmaf (r, f, 4.16695364e-2f); // 0x1.555b5ap-5
    r = fmaf (r, f, 1.66664720e-1f); // 0x1.555450p-3
    r = fmaf (r, f, 4.99999851e-1f); // 0x1.fffff6p-2
    r = fmaf (r, f, 1.00000000e+0f); // 0x1.000000p+0
    r = fmaf (r, f, 1.00000000e+0f); // 0x1.000000p+0
    // exp(a) = 2**i * r
    ia = (i > 0) ?  0 : 0x83000000;
    s = __int_as_float (0x7f000000 + ia);
    t = __int_as_float ((i << 23) - ia);
    r = r * s;
    r = r * t;
    // handle special cases: severe overflow / underflow
    if (fabsf (a) >= 104.0f) r = s * s;
    return r;
}

//**-----------------GR Kernels------------------**

__global__ void calcActivityGRGPU(float *vm, float *gKCa, float *gLeak, float *gNMDA, float *gNMDAInc,
	float *thresh, uint32_t *apBuf, uint8_t *apOutGR, uint32_t *apGR, int *apMFtoGR,
	float *gESum, float *gISum, float eLeak, float eGOIn, float gAMPAInc, 
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


	tempV = tempV + gLeak[i] * (eLeak - tempV) - gESum[i] * tempV 
	   	  - gNMDA[i] * tempV + gISum[i] * (eGOIn - tempV); 

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

//TODO: these update functions do the same thing. Might be time to refactor ie 
// create a more general update fnctn, take variadic args at the end in case 
// we are updating more than just the input...gpu arr
__global__ void updateMFInGOOPGPU(uint32_t inNLoads, uint32_t *apIn, uint32_t *conFromIn,
		size_t conFromInPitch, int32_t *numInPerGO, uint32_t *inputMFGOGPU)
{
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int *conRow;
	int tempNSyn = numInPerGO[index];
	int tempApInSum = 0;

	for (int i = 0; i < inNLoads; i++)
	{
		sharedIOBufGO[tid + i * blockDim.x] = apIn[tid + i * blockDim.x];
	}
	__syncthreads();

	for (int i = 0; i < tempNSyn; i++)
	{
		conRow = (uint32_t *)((char *)conFromIn + i * conFromInPitch);
		tempApInSum += sharedIOBufGO[conRow[index]];
	}
	inputMFGOGPU[index] = tempApInSum;
}

// keep in mind: this kernel runs for every POST-synaptic GO. thus why tid + i *blockDim.x 
// is used to index the input spikes while index (ie absolute location in gpu memory of this
// post-syn GO) is used to index conRow
__global__ void updateGOInGOGOOPGPU(uint32_t inNLoads, uint32_t *apIn, uint32_t *conFromIn,
		size_t conFromInPitch, int32_t *numInPerGO, uint32_t *inputGOGOGPU)
{
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int *conRow;
	int tempNSyn = numInPerGO[index];
	int tempApInSum = 0;

	for (int i = 0; i < inNLoads; i++)
	{
		sharedIOBufGO[tid + i * blockDim.x] = apIn[tid + i * blockDim.x];
	}
	__syncthreads(); // necessary because sharedIOBufGO for tempNSyn in any given
                     // instance of this kernel may be read (below) at an index which was written
                     // to by another thread!

	for (int i = 0; i < tempNSyn; i++)
	{
		conRow = (uint32_t *)((char *)conFromIn + i * conFromInPitch);
		tempApInSum += sharedIOBufGO[conRow[index]]; // no need to atomic add, i no understand
	}
	inputGOGOGPU[index] = tempApInSum;
}

__global__ void updateGOCoupInGOGOOPGPU(uint32_t inNLoads, float *vGO, uint32_t *conFromIn,
		size_t conFromInPitch, int32_t *numInCoupPerGO, float coupleRiRjRatioGO,
		float *coupIn, size_t coupInPitch, float *vCoupleGO)
{
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t *conRow;
	float * coupCoefRow;
	int tempNGJ = numInCoupPerGO[index];
	float localVGO = vGO[index];

	for (int i = 0; i < inNLoads; i++)
	{
		sharedIOBufGOfloat[tid + i * blockDim.x] = vGO[tid + i * blockDim.x];
	}
	__syncthreads();

	// could be a source of slow-down as 0 <= tempNGJ < 81
	for (int i = 0; i < tempNGJ; i++)
	{
		conRow = (uint32_t *)((char *)conFromIn + i * conFromInPitch);
		coupCoefRow = (float *)((char *)coupIn + i * coupInPitch);
		vCoupleGO[index] += coupleRiRjRatioGO * coupCoefRow[index] * (sharedIOBufGOfloat[conRow[index]] - localVGO);
	}
}

__global__ void updateGOGRDynamicSpillOutGPU(float spillFrac, float gIncFracSpilloverGOtoGR,
	float gogrW, uint32_t *apGO, uint32_t *isiCounter, float *dynamicAmpGOOut)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float scalerGOGR = gogrW * gIncFracSpilloverGOtoGR * 1.4;
	float halfShift = 12.0;
	float steepness = 20.0;
	float baselvl = spillFrac * gogrW;

	dynamicAmpGOOut[index] = baselvl + scalerGOGR * (1 / (1 + my_expf((isiCounter[index] - halfShift) / steepness)));
	isiCounter[index] = (1 - apGO[index]) * isiCounter[index] + 1;
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
		sharedIOBufGOfloat[tid+i*blockDim.x]=dynamicAmp[tid+i*blockDim.x];
	}
	__syncthreads();
	
	for(int i=0; i<tempNSyn; i++)
	{
		conRow=(unsigned int *)((char *)conFromIn+i*conFromInPitch);
		tempDynamicAmpSum += sharedIOBufGOfloat[conRow[index]];
	}

	dynamicAmpGOGRGPU[index] = tempDynamicAmpSum / 3;
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

//**---------------GO Kernels-------------------**

__global__ void calcActivityGOGPU(float *vGO, float *vCoupleGOGO, float *threshGO, uint32_t *apBufGO,
  uint32_t *apGO, uint32_t *inputMFGO, uint32_t *inputGOGO, uint32_t *inputGRGO, float *gSum_MFGO, float *gSum_GOGO,
  float *synWScalerGOtoGO, float *synWScalerGRtoGO, float NMDA_AMPARatioMFGO, float *gNMDAMFGO, 
  float *gNMDAIncMFGO, float *gGRGO, float *gGRGO_NMDA, float gLeakGO, float eLeakGO, float threshRestGO, float threshMaxGO,
  float threshDecGO, float mfgoW, float gogoW, float grgoW, float gDecMFtoGO, float gGABADecGOtoGO, float gDecMFtoGONMDA,
  float gDecGRtoGO, float eGABAGO)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float tempThreshGO = threshGO[i];
	uint8_t tempAPGO = apGO[i];
	float tempVGO = vGO[i];

		//NMDA Low
	float gNMDAIncGRGO = (0.00000082263 * tempVGO * tempVGO * tempVGO)
					   + (0.00021653 * tempVGO * tempVGO)
					   + (0.0195 * tempVGO)
					   + 0.6117; 

	//NMDA High
	gNMDAIncMFGO[i] = (0.00000011969 * tempVGO * tempVGO * tempVGO)
						+ (0.000089369 * tempVGO * tempVGO)
						+ (0.0151 * tempVGO)
						+ 0.7713;

	gSum_MFGO[i]  = (inputMFGO[i] * mfgoW) + gSum_MFGO[i] * gDecMFtoGO;
	gSum_GOGO[i]  = (inputGOGO[i] * gogoW * synWScalerGOtoGO[i]) + gSum_GOGO[i] * gGABADecGOtoGO;
	gNMDAMFGO[i]  = (inputMFGO[i] * mfgoW * NMDA_AMPARatioMFGO * gNMDAIncMFGO[i])
					+ gNMDAMFGO[i] * gDecMFtoGONMDA;
	gGRGO[i]      = (inputGRGO[i] * grgoW * synWScalerGRtoGO[i]) + gGRGO[i] * gDecGRtoGO;
	gGRGO_NMDA[i] = (inputGRGO[i] * grgoW * synWScalerGRtoGO[i] * 0.6 * gNMDAIncGRGO)
					+ gGRGO_NMDA[i] * gDecMFtoGONMDA;

	tempThreshGO += (threshRestGO - tempThreshGO) * threshDecGO;

	// the balance of excitatory MF and GR input to inhibitory GO input is wayyy off.
	tempVGO += gLeakGO * (eLeakGO - tempVGO) + gSum_GOGO[i] * (eGABAGO - tempVGO)
				- (gSum_MFGO[i] + gGRGO[i] + gNMDAMFGO[i] + gGRGO_NMDA[i]) * tempVGO
				- vCoupleGOGO[i] * tempVGO;

  if (tempVGO > threshMaxGO) tempVGO = threshMaxGO;

  tempAPGO = tempVGO > tempThreshGO;
  apBufGO[i] = (apBufGO[i] << 1) | (tempAPGO * 0x00000001);

  tempThreshGO = tempAPGO * threshMaxGO + (1 - tempAPGO) * tempThreshGO;

  apGO[i] = tempAPGO;
  vGO[i] = tempVGO;
  threshGO[i] = tempThreshGO;
}

//**---------------end GO Kernels-------------------**

//**---------------IO kernels-----------------**

__global__ void updatePFPCSynIO(float *synWPFPC, uint64_t *historyGR, uint64_t plastCheckMask,
		unsigned int offset, float plastStep)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x+offset;
	synWPFPC[i]=synWPFPC[i]+((historyGR[i]&plastCheckMask)>0)*plastStep;

	synWPFPC[i]=(synWPFPC[i]>0)*synWPFPC[i];
	synWPFPC[i]=(synWPFPC[i]>1)+(synWPFPC[i]<=1)*synWPFPC[i];
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

void callUpdateMFInGOOPKernel(cudaStream_t &st, uint32_t numBlocks, uint32_t numGOPerBlock,
		uint32_t numInCells, uint32_t *apInGPU, uint32_t *conInGOGPU, size_t conInGOGPUP,
		int32_t *numInPerGOGPU, uint32_t *inputMFGOGPU)
{
	updateMFInGOOPGPU<<<numBlocks, numGOPerBlock, numInCells*sizeof(uint32_t), st>>>
			(numInCells/numGOPerBlock, apInGPU, conInGOGPU, conInGOGPUP, numInPerGOGPU, inputMFGOGPU);
}

void callUpdateGOInGOOPKernel(cudaStream_t &st, uint32_t numBlocks, uint32_t numGOPerBlock,
		uint32_t numInCells, uint32_t *apInGPU, uint32_t *conInGOGPU, size_t conInGOGPUP,
		int32_t *numInPerGOGPU, uint32_t *inputGOGOGPU)
{
	updateGOInGOGOOPGPU<<<numBlocks, numGOPerBlock, numInCells*sizeof(uint32_t), st>>>
			(numInCells/numGOPerBlock, apInGPU, conInGOGPU, conInGOGPUP, numInPerGOGPU, inputGOGOGPU);
}

void callUpdateGOCoupInGOOPKernel(cudaStream_t &st, uint32_t numBlocks, uint32_t numGOPerBlock,
		uint32_t numInCells, float *vGO, uint32_t *conFromIn, size_t conFromInPitch, int32_t *numInCoupPerGO,
		float coupleRiRjRatioGO, float *coupIn, size_t coupInPitch, float *vCoupleGO)
{
	updateGOCoupInGOGOOPGPU<<<numBlocks, numGOPerBlock, numInCells * sizeof(float), st>>>(numInCells / numGOPerBlock,
			vGO, conFromIn, conFromInPitch, numInCoupPerGO, coupleRiRjRatioGO, coupIn, coupInPitch, vCoupleGO);
}

void callUpdateGOInGRDepressionOPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
	  unsigned int numInCells, float *depAmpGPU, uint32_t *conInGRGPU, size_t conInGRGPUP, 
	  int32_t *numInPerGRGPU, float *depAmpGOGRGPU)
{
	updateGOGRDepressionInOPGPU<<<numBlocks, numGRPerBlock, numInCells * sizeof(uint32_t), st>>>
	   (numInCells / numGRPerBlock, depAmpGPU, conInGRGPU, conInGRGPUP, numInPerGRGPU, depAmpGOGRGPU);
}

void callUpdateGOOutGRDynamicSpillOPKernel(cudaStream_t &st, uint32_t numBlocks, uint32_t numGOPerBlock,
	float spillFrac, float gIncFracSpilloverGOtoGR, float gogrW, uint32_t *apGO,
	uint32_t *isiCounter, float *dynamicAmpGOOut)
{
	updateGOGRDynamicSpillOutGPU<<<numBlocks, numGOPerBlock, 0, st>>>(spillFrac, gIncFracSpilloverGOtoGR,
		gogrW, apGO, isiCounter, dynamicAmpGOOut);
}

void callUpdateGOInGRDynamicSpillOPKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		unsigned int numInCells, float *dynamicAmpGPU,
		uint32_t *conInGRGPU, size_t conInGRGPUP,
		int32_t *numInPerGRGPU, float *dynamicAmpGOGRGPU)
{
	updateGOGRDynamicSpillInOPGPU<<<numBlocks, numGRPerBlock, numInCells*sizeof(float), st>>>
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

void callGOActKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGOPerBlock,
	float *vGO, float *vCoupleGOGO, float *threshGO, uint32_t *apBufGO,
	uint32_t *apGO, uint32_t *inputMFGO, uint32_t *inputGOGO, uint32_t *inputGRGO, float *gSum_MFGO, float *gSum_GOGO,
	float *synWScalerGOtoGO, float *synWScalerGRtoGO, float NMDA_AMPARatioMFGO, float *gNMDAMFGO, 
	float *gNMDAIncMFGO, float *gGRGO, float *gGRGO_NMDA, float gLeakGO, float eLeakGO, float threshRestGO, float threshMaxGO,
	float threshDecGO, float mfgoW, float gogoW, float grgoW, float gDecMFtoGO, float gGABADecGOtoGO, float gDecMFtoGONMDA,
	float gDecGRtoGO, float eGABAGO)
{
	calcActivityGOGPU<<<numBlocks, numGOPerBlock, 0, st>>>(vGO, vCoupleGOGO, threshGO, apBufGO,
		apGO, inputMFGO, inputGOGO, inputGRGO, gSum_MFGO, gSum_GOGO, synWScalerGOtoGO, synWScalerGRtoGO,
		NMDA_AMPARatioMFGO, gNMDAMFGO, gNMDAIncMFGO, gGRGO, gGRGO_NMDA, gLeakGO, eLeakGO, threshRestGO, threshMaxGO,
		threshDecGO, mfgoW, gogoW, grgoW, gDecMFtoGO, gGABADecGOtoGO, gDecMFtoGONMDA, gDecGRtoGO, eGABAGO);
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

void callUpdatePFPCPlasticityIOKernel(cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
		float *synWeightGPU, uint64_t *historyGPU, unsigned int pastBinNToCheck,
		int offSet, float pfPCPlastStep)
{
	uint64_t mask = ((uint64_t)1)<<(pastBinNToCheck-1);
		updatePFPCSynIO<<<numBlocks, numGRPerBlock, 0, st>>>(synWeightGPU, historyGPU,
				mask, offSet, pfPCPlastStep);
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
