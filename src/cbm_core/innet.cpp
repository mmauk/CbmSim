/*
 *
 * innet.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 */

#include <iostream>
#include <math.h>

#include "activityparams.h"
#include "connectivityparams.h"
#include "dynamic2darray.h"
#include "innet.h"
#include "logger.h"

InNet::InNet() {}

InNet::InNet(InNetConnectivityState *cs, InNetActivityState *as,
             int gpuIndStart, int numGPUs) {
  // all of below are shallow-copying the pointers
  // thus we don't necessarily have ownership over
  // them, so caller deletes them. we assume the innet
  // object goes out of scope before or at the time of
  // the calling class.
  this->cs = cs;
  this->as = as;

  this->gpuIndStart = gpuIndStart;
  this->numGPUs = numGPUs;

  // use transpose arrays as makes for better layout in gpu memory
  // when copied over to device
  gGOGRT = allocate2DArray<float>(max_num_p_gr_from_go_to_gr, num_gr);
  gMFGRT = allocate2DArray<float>(max_num_p_gr_from_mf_to_gr, num_gr);

  pGRDelayfromGRtoGOT =
      allocate2DArray<uint32_t>(max_num_p_gr_from_gr_to_go, num_gr);
  pGRfromMFtoGRT =
      allocate2DArray<uint32_t>(max_num_p_gr_from_mf_to_gr, num_gr);
  pGRfromGOtoGRT =
      allocate2DArray<uint32_t>(max_num_p_gr_from_go_to_gr, num_gr);
  pGRfromGRtoGOT =
      allocate2DArray<uint32_t>(max_num_p_gr_from_gr_to_go, num_gr);

  apBufGRHistMask = (1 << (int)tsPerHistBinGR) - 1;

  sumGRInputGO = new uint32_t[num_go];
  sumInputGOGABASynDepGO = new float[num_go];

  initCUDA();
}

InNet::~InNet() {
  LOG_DEBUG("Deleting innet gpu arrays.");

  // gr external to initCUDA
  delete2DArray<float>(gGOGRT);
  delete2DArray<float>(gMFGRT);

  delete2DArray<uint32_t>(pGRDelayfromGRtoGOT);
  delete2DArray<uint32_t>(pGRfromMFtoGRT);
  delete2DArray<uint32_t>(pGRfromGOtoGRT);
  delete2DArray<uint32_t>(pGRfromGRtoGOT);

  // go, external to initCUDA
  delete[] sumGRInputGO;
  delete[] sumInputGOGABASynDepGO;

  // MF CUDA
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);

    // mf variables
    cudaFree(apMFGPU[i]);
    cudaFreeHost(apMFH[i]);
    cudaFree(depAmpMFGPU[i]);
    cudaFreeHost(depAmpMFH[i]);

    cudaDeviceSynchronize();
  }

  delete[] apMFGPU;
  delete[] apMFH;
  delete[] depAmpMFH;
  delete[] depAmpMFGPU;

  // GR CUDA
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);

    cudaFree(outputGRGPU[i]);
    cudaFree(apGRGPU[i]);
    cudaFree(vGRGPU[i]);
    cudaFree(gKCaGRGPU[i]);
    cudaFree(gLeakGRGPU[i]);
    cudaFree(gNMDAGRGPU[i]);
    cudaFree(gNMDAIncGRGPU[i]);
    cudaFree(gEGRGPU[i]);
    cudaFree(gEGRSumGPU[i]);
    cudaFree(gEDirectGPU[i]);
    cudaFree(gESpilloverGPU[i]);
    cudaFree(apMFtoGRGPU[i]);
    cudaFree(numMFperGR[i]);
    cudaFree(depAmpMFGRGPU[i]);
    cudaFree(depAmpGOGRGPU[i]);
    cudaFree(dynamicAmpGOGRGPU[i]);
    cudaFree(gIGRGPU[i]);
    cudaFree(gIGRSumGPU[i]);
    cudaFree(gIDirectGPU[i]);
    cudaFree(gISpilloverGPU[i]);
    cudaFree(apBufGRGPU[i]);
    cudaFree(threshGRGPU[i]);
    cudaFree(delayGOMasksGRGPU[i]);

    cudaFree(grConGROutGOGPU[i]);
    cudaFree(numGOOutPerGRGPU[i]);
    cudaFree(grConGOOutGRGPU[i]);
    cudaFree(numGOInPerGRGPU[i]);
    cudaFree(grConMFOutGRGPU[i]);
    cudaFree(numMFInPerGRGPU[i]);
    cudaFree(historyGRGPU[i]);

    cudaDeviceSynchronize();
  }

  // GR CUDA
  delete[] gEGRGPU;
  delete[] gEGRGPUP;
  delete[] gEGRSumGPU;
  delete[] gEDirectGPU;
  delete[] gESpilloverGPU;
  delete[] apMFtoGRGPU;
  delete[] numMFperGR;
  delete[] depAmpMFGRGPU;
  delete[] depAmpGOGRGPU;
  delete[] dynamicAmpGOGRGPU;

  delete[] gIGRGPU;
  delete[] gIGRGPUP;
  delete[] gIGRSumGPU;
  delete[] gIDirectGPU;
  delete[] gISpilloverGPU;

  delete[] apBufGRGPU;
  delete[] outputGRGPU;
  delete[] apGRGPU;

  delete[] threshGRGPU;
  delete[] vGRGPU;
  delete[] gKCaGRGPU;
  delete[] gLeakGRGPU;
  delete[] gNMDAGRGPU;
  delete[] gNMDAIncGRGPU;
  delete[] historyGRGPU;

  delete[] delayGOMasksGRGPU;
  delete[] delayGOMasksGRGPUP;

  delete[] numGOOutPerGRGPU;
  delete[] grConGROutGOGPU;
  delete[] grConGROutGOGPUP;

  delete[] numGOInPerGRGPU;
  delete[] grConGOOutGRGPU;
  delete[] grConGOOutGRGPUP;

  delete[] numMFInPerGRGPU;
  delete[] grConMFOutGRGPU;
  delete[] grConMFOutGRGPUP;

  delete[] outputGRH;
  // cudaFreeHost(outputGRH);

  // GO CUDA
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);

    cudaFreeHost(grInputGOSumH[i]);
    cudaFreeHost(apGOH[i]);
    cudaFreeHost(depAmpGOH[i]);
    cudaFreeHost(dynamicAmpGOH[i]);

    cudaFree(apGOGPU[i]);
    cudaFree(depAmpGOGPU[i]);
    cudaFree(dynamicAmpGOGPU[i]);
    cudaFree(grInputGOGPU[i]);
    cudaFree(grInputGOSumGPU[i]);

    cudaDeviceSynchronize();
  }

  delete[] grInputGOSumH;
  delete[] apGOH;
  delete[] apGOGPU;
  delete[] grInputGOGPU;
  delete[] grInputGOGPUP;
  delete[] grInputGOSumGPU;
  delete[] depAmpGOH;
  delete[] depAmpGOGPU;
  delete[] dynamicAmpGOH;
  delete[] dynamicAmpGOGPU;

  delete[] counter;

  LOG_DEBUG("Finished deleting innet gpu arrays.");
}

void InNet::writeToState() {
  cudaError_t error;
  // GR variables
  //  WARNING THIS IS A HORRIBLE IDEA. IF YOU GET BUGS CONSIDER THIS!
  //  Reason: the apGR is a unique_ptr. it should only be modifed in the scope
  //  that it is defined in.
  getGRGPUData<uint8_t>(outputGRGPU, as->apGR.get());
  getGRGPUData<uint32_t>(apBufGRGPU, as->apBufGR.get());
  getGRGPUData<float>(gEGRSumGPU, as->gMFSumGR.get());
  getGRGPUData<float>(gIGRSumGPU, as->gGOSumGR.get());

  getGRGPUData<float>(threshGRGPU, as->threshGR.get());
  getGRGPUData<float>(vGRGPU, as->vGR.get());
  getGRGPUData<float>(gKCaGRGPU, as->gKCaGR.get());
  getGRGPUData<uint64_t>(historyGRGPU, as->historyGR.get());
  for (int i = 0; i < numGPUs; i++) {
    int cpyStartInd;
    int cpySize;

    cpyStartInd = numGRPerGPU * i;
    cpySize = numGRPerGPU;

    cudaSetDevice(i + gpuIndStart);

    for (int j = 0; j < max_num_p_gr_from_mf_to_gr; j++) {
      error = cudaMemcpy(&gMFGRT[j][cpyStartInd],
                         (void *)((char *)gEGRGPU[i] + j * gEGRGPUP[i]),
                         cpySize * sizeof(float), cudaMemcpyDeviceToHost);
    }

    for (int j = 0; j < max_num_p_gr_from_go_to_gr; j++) {
      error = cudaMemcpy(&gGOGRT[j][cpyStartInd],
                         (void *)((char *)gIGRGPU[i] + j * gIGRGPUP[i]),
                         cpySize * sizeof(float), cudaMemcpyDeviceToHost);
    }
  }

  for (int i = 0; i < max_num_p_gr_from_mf_to_gr; i++) {
    for (int j = 0; j < num_gr; j++) {
      // NOTE: gMFGR now 1D array.
      as->gMFGR[j * max_num_p_gr_from_mf_to_gr + i] = gMFGRT[i][j];
    }
  }

  for (int i = 0; i < max_num_p_gr_from_go_to_gr; i++) {
    for (int j = 0; j < num_gr; j++) {
      as->gGOGR[j * max_num_p_gr_from_go_to_gr + i] = gGOGRT[i][j];
    }
  }
}

// void InNet::grStim(int startGRStim, int numGRStim)
//{
//	// might be a useless operation. would the state of these arrays
//	// on cpu be the same as on gpu at the time we modify the segment below?
//	// if so, no need to get gpu data. if not, need to get gpu data
//	getGRGPUData<uint8_t>(outputGRGPU, as->apGR.get());
//	getGRGPUData<uint32_t>(apBufGRGPU, as->apBufGR.get());
//	for (int j = startGRStim; j <= startGRStim + numGRStim; j++)
//	{
//		/* as-> apBufGR[j] |= 1u; // try this to see if we get the same
// result */ 		as->apBufGR[j] = as->apBufGR[j] | 1u;
// outputGRH[j] = true;
//	}
//
//	for (int i = 0; i < numGPUs; i++)
//	{
//		int cpyStartInd;
//		int cpySize;
//
//		cpyStartInd=numGRPerGPU*i;//numGR*i/numGPUs;
//		cpySize=numGRPerGPU;
//		cudaSetDevice(i+gpuIndStart);
//
//		cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]),
//				cpySize*sizeof(uint32_t),
//cudaMemcpyHostToDevice); 		cudaMemcpy(outputGRGPU[i], &outputGRH[cpyStartInd],
//				cpySize*sizeof(uint8_t),
//cudaMemcpyHostToDevice);
//	}
// }

const uint8_t *InNet::exportAPGO() { return (const uint8_t *)as->apGO.get(); }

const uint8_t *InNet::exportAPMF() { return (const uint8_t *)apMFOut; }

const uint8_t *InNet::exportHistMF() {
  return (const uint8_t *)as->histMF.get();
}

const uint8_t *InNet::exportAPGR() {
  cudaError_t error = getGRGPUData<uint8_t>(outputGRGPU, outputGRH);
  return (const uint8_t *)outputGRH;
}

const uint32_t *InNet::exportSumGRInputGO() {
  return (const uint32_t *)sumGRInputGO;
}

const float *InNet::exportSumGOInputGO() {
  return (const float *)sumInputGOGABASynDepGO;
}

uint32_t **InNet::getApBufGRGPUPointer() { return apBufGRGPU; }

// YIKES this should be deprecated: control class should not be able
// to interact directly with the gpu pointers here!!!
uint64_t **InNet::getHistGRGPUPointer() { return historyGRGPU; }

uint32_t **InNet::getGRInputGOSumHPointer() { return grInputGOSumH; }

const float *InNet::exportGESumGR() {
  getGRGPUData<float>(gEGRSumGPU, as->gMFSumGR.get());
  return (const float *)as->gMFSumGR.get();
}

const float *InNet::exportGISumGR() {
  getGRGPUData<float>(gIGRSumGPU, as->gGOSumGR.get());
  return (const float *)as->gGOSumGR.get();
}

const float *InNet::exportgSum_MFGO() {
  return (const float *)as->gSum_MFGO.get();
}

const float *InNet::exportgSum_GRGO() { return (const float *)as->gGRGO.get(); }

void InNet::updateMFActivties(const uint8_t *actInMF) {
  apMFOut = actInMF;
  for (int i = 0; i < num_mf; i++) {
    as->histMF[i] = as->histMF[i] || (actInMF[i] > 0);
    for (int j = 0; j < numGPUs; j++) {
      apMFH[j][i] = (actInMF[i] > 0);
    }
    as->apBufMF[i] = (as->apBufMF[i] << 1) | ((actInMF[i] > 0) * 0x00000001);
  }
}

void InNet::calcGOActivities() {
#pragma omp parallel for // an attempt at parallelization using openmp haha
  for (int i = 0; i < num_go; i++) {
    // gather gr -> go input sums copied from all devices into one host array
    sumGRInputGO[i] = 0;
    for (int j = 0; j < numGPUs; j++) {
      sumGRInputGO[i] += grInputGOSumH[j][i];
    }
    // temp var as faster to access local memory than dereference array over and
    // over (or so I hear)
    float tempVGO = as->vGO[i];

    // NMDA Low (voltage-dep step size for nmda conductance update gr -> go)
    float gNMDAIncGRGO = (0.00000082263 * tempVGO * tempVGO * tempVGO) +
                         (0.00021653 * tempVGO * tempVGO) + (0.0195 * tempVGO) +
                         0.6117;

    // NMDA High (voltage-dep step size for nmda conductance update mf -> go)
    as->gNMDAIncMFGO[i] = (0.00000011969 * tempVGO * tempVGO * tempVGO) +
                          (0.000089369 * tempVGO * tempVGO) +
                          (0.0151 * tempVGO) + 0.7713;

    // update total mf -> go input conductance
    as->gSum_MFGO[i] =
        (as->inputMFGO[i] * mfgoW) + as->gSum_MFGO[i] * gDecMFtoGO;
    // update total go -> go input conductance. (notice the weight gogoW)
    as->gSum_GOGO[i] = (as->inputGOGO[i] * gogoW * as->synWscalerGOtoGO[i]) +
                       as->gSum_GOGO[i] * gGABADecGOtoGO;
    // update mf -> go nmda conductance. (notice the weight mfgoW and NMDA/AMPA
    // ratio)
    as->gNMDAMFGO[i] =
        as->inputMFGO[i] * (mfgoW * NMDA_AMPAratioMFGO * as->gNMDAIncMFGO[i]) +
        as->gNMDAMFGO[i] * gDecayMFtoGONMDA;

    // update gr -> go input conductance. (notice the weight grgoW)
    as->gGRGO[i] = (sumGRInputGO[i] * grgoW * as->synWscalerGRtoGO[i]) +
                   as->gGRGO[i] * gDecGRtoGO;
    // update gr -> go nmda conductance. (also notice the grgoW and syn scale
    // value) value of 0.6 appears to be a fudge-factor
    as->gGRGO_NMDA[i] = sumGRInputGO[i] * ((grgoW * as->synWscalerGRtoGO[i]) *
                                           0.6 * gNMDAIncGRGO) +
                        as->gGRGO_NMDA[i] * gDecayMFtoGONMDA;

    // update voltage threshold variable
    as->threshCurGO[i] += (threshRestGO - as->threshCurGO[i]) * threshDecGO;

    // use all updated conductances to update the voltage
    tempVGO += (gLeakGO * (eLeakGO - tempVGO)) +
               (as->gSum_GOGO[i] * (eGABAGO - tempVGO)) -
               (as->gSum_MFGO[i] + as->gGRGO[i] + as->gNMDAMFGO[i] +
                as->gGRGO_NMDA[i]) *
                   tempVGO -
               (as->vCoupleGO[i] * tempVGO);

    // set voltage to threshold if supersedes it
    if (tempVGO > threshMaxGO)
      tempVGO = threshMaxGO;

    // spike if above threshold
    as->apGO[i] = tempVGO > as->threshCurGO[i];
    // update the buffer of previous spikes (used in updating history I think)
    as->apBufGO[i] = (as->apBufGO[i] << 1) | (as->apGO[i] * 0x00000001);

    // set the threshold back to max if we spiked
    as->threshCurGO[i] =
        as->apGO[i] * threshMaxGO + (1 - as->apGO[i]) * as->threshCurGO[i];

    // reset inputs
    as->inputMFGO[i] = 0;
    as->inputGOGO[i] = 0;
    // make final assignment to activity state's voltage array
    as->vGO[i] = tempVGO;

    // copy that activity states voltage array to the 2d buffer which
    // will be used to copy spikes to devices
    for (int j = 0; j < numGPUs; j++) {
      apGOH[j][i] = as->apGO[i];
    }
  }
}

void InNet::updateMFtoGROut() {
  float recoveryRate = 1 / recoveryTauMF;

  for (int i = 0; i < num_mf; i++) {
    // update the mf -> gr depression amplitude
    as->depAmpMFGR[i] =
        apMFH[0][i] * as->depAmpMFGR[i] * fracDepMF +
        (!apMFH[0][i]) *
            (as->depAmpMFGR[i] + recoveryRate * (1 - as->depAmpMFGR[i]));
    // copy result to 2d buffer, used to make repeats on each device
    for (int j = 0; j < numGPUs; j++) {
      depAmpMFH[j][i] = as->depAmpMFGR[i];
    }
  }
}

void InNet::updateMFtoGOOut() {
  for (int i = 0; i < num_mf; i++) {
    // if mf spiked, then update the mf -> go input
    // for all go this mf is connected with
    if (apMFH[0][i]) {
      for (int j = 0; j < cs->numpMFfromMFtoGO[i]; j++) {
        as->inputMFGO[cs->pMFfromMFtoGO[i][j]]++;
      }
    }
  }
}

void InNet::updateGOtoGROutParameters(float spillFrac) {
  // TODO: place these in the build file as well (one day)
  float scalerGOGR = gogrW * gIncFracSpilloverGOtoGR * 1.4;
  float halfShift = 12.0; // shift;
  float steepness = 20.0; // steep;
  float recoveryRate = 1 / recoveryTauGO;
  float baselvl = spillFrac * gogrW;

#pragma omp parallel for
  for (int i = 0; i < num_go; i++) {
    // reset depression amplitudes
    as->depAmpGOGR[i] = 1;
    // update dynamic amplitudes (uses a sigmoid function on how long since go
    // last spiked)
    as->dynamicAmpGOGR[i] =
        baselvl +
        (scalerGOGR * (1 / (1 + (exp((counter[i] - halfShift) / steepness)))));
    counter[i] = (1 - as->apGO[i]) * counter[i] + 1;

    for (int j = 0; j < numGPUs; j++) {
      depAmpGOH[j][i] = 1;
      dynamicAmpGOH[j][i] = as->dynamicAmpGOGR[i];
    }
  }
}

void InNet::updateGOtoGOOut() {

#pragma omp parallel for
  for (int i = 0; i < num_go; i++) {
    // update all go this go is connected to if this go spiked
    if (as->apGO[i]) {
      for (int j = 0; j < cs->numpGOGABAOutGOGO[i]; j++) {
        as->inputGOGO[cs->pGOGABAOutGOGO[i][j]]++;
      }
    }
  }

#pragma omp parallel for
  for (int i = 0; i < num_go; i++) {
    for (int j = 0; j < cs->numpGOCoupInGOGO[i]; j++) {
      // update the coupling voltage for every go
      as->vCoupleGO[i] += (as->vGO[cs->pGOCoupInGOGO[i][j]] - as->vGO[i]) *
                          coupleRiRjRatioGO * cs->pGOCoupInGOGOCCoeff[i][j];
    }
  }
}

void InNet::resetMFHist(uint32_t t) {
  // for every numTSinMFHist, make the whole histMF array false
  // (used in mf -> nc plasticity)
  if (t % (uint32_t)numTSinMFHist == 0) {
    for (int i = 0; i < num_mf; i++) {
      as->histMF[i] = false;
    }
  }
}

void InNet::runGRActivitiesCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;

  // total ampa conductance increment is the direct plus fraction
  // that is spillover (feel like this should be:
  //         (1 - gIncFracSpilloverMFtoGR) * gIncDirectMFtoGR +
  //          gIncDirectMFtoGR * gIncFracSpilloverMFtoGR
  //
  //  but I didn't write this part :-)
  float gAMPAInc =
      gIncDirectMFtoGR + gIncDirectMFtoGR * gIncFracSpilloverMFtoGR;

  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    callGRActKernel(sts[i][streamN], calcGRActNumBlocks, calcGRActNumGRPerB,
                    vGRGPU[i], gKCaGRGPU[i], gLeakGRGPU[i], gNMDAGRGPU[i],
                    gNMDAIncGRGPU[i], threshGRGPU[i], apBufGRGPU[i],
                    outputGRGPU[i], apGRGPU[i], apMFtoGRGPU[i], gEGRSumGPU[i],
                    gIGRSumGPU[i], eLeakGR, eGOGR, gAMPAInc, threshRestGR,
                    threshMaxGR, threshDecGR);
#ifdef DEBUGOUT
    error = cudaGetLastError();
    cerr << "grActivityCUDA: kernel launch for gpu #" << i << ": "
         << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::runSumGRGOOutCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    callSumGRGOOutKernel(sts[i][streamN], sumGRGOOutNumBlocks,
                         sumGRGOOutNumGOPerB, updateGRGOOutNumGRRows,
                         grInputGOGPU[i], grInputGOGPUP[i], grInputGOSumGPU[i]);
  }
}

void InNet::cpyDepAmpMFHosttoGPUCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    error =
        cudaMemcpyAsync(depAmpMFGPU[i], depAmpMFH[i], num_mf * sizeof(float),
                        cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
    cerr << "cpyAPMFHosttoGPUCUDA: async copy for gpu #" << i << ": "
         << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::cpyAPMFHosttoGPUCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    error = cudaMemcpyAsync(apMFGPU[i], apMFH[i], num_mf * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
    cerr << "cpyAPMFHosttoGPUCUDA: async copy for gpu #" << i << ": "
         << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::cpyDepAmpUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN) {}

void InNet::cpyAPUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN) {}

void InNet::cpyDepAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN) {}

void InNet::cpyDynamicAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    error = cudaMemcpyAsync(dynamicAmpGOGPU[i], dynamicAmpGOH[i],
                            num_go * sizeof(float), cudaMemcpyHostToDevice,
                            sts[i][streamN]);
#ifdef DEBUGOUT
    cerr << "cpyDynamicAmpGOGRHosttoGPUCUDA: async copy for gpu #" << i << ": "
         << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::cpyAPGOHosttoGPUCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    error = cudaMemcpyAsync(apGOGPU[i], apGOH[i], num_go * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, sts[i][streamN]);
#ifdef DEBUGOUT
    cerr << "cpyAPGOHosttoGPUCUDA: async copy for gpu #" << i << ": "
         << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::runUpdateMFInGRCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    callUpdateMFInGROPKernel(
        sts[i][streamN], updateMFInGRNumBlocks, updateMFInGRNumGRPerB, num_mf,
        apMFGPU[i], depAmpMFGRGPU[i], gEGRGPU[i], gEGRGPUP[i],
        grConMFOutGRGPU[i], grConMFOutGRGPUP[i], numMFInPerGRGPU[i],
        apMFtoGRGPU[i], gEGRSumGPU[i], gEDirectGPU[i], gESpilloverGPU[i],
        gDirectDecMFtoGR, gIncDirectMFtoGR, gSpilloverDecMFtoGR,
        gIncFracSpilloverMFtoGR);
#ifdef DEBUGOUT
    error = cudaGetLastError();
    cerr << "runUpdateMFInGRCUDA: kernel launch for gpu #" << i << ": "
         << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::runUpdateMFInGRDepressionCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    callUpdateMFInGRDepressionOPKernel(
        sts[i][streamN], updateMFInGRNumBlocks, updateMFInGRNumGRPerB, num_mf,
        depAmpMFGPU[i], grConMFOutGRGPU[i], grConMFOutGRGPUP[i],
        numMFInPerGRGPU[i], numMFperGR[i], depAmpMFGRGPU[i]);
#ifdef DEBUGOUT
    error = cudaGetLastError();
    cerr << "runUpdateMFInGRDepressionCUDA: kernel launch for gpu #" << i
         << ": " << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::runUpdateUBCInGRCUDA(cudaStream_t **sts, int streamN) {}

void InNet::runUpdateUBCInGRDepressionCUDA(cudaStream_t **sts, int streamN) {}

void InNet::runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    callUpdateInGROPKernel(
        sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB, num_go,
        apGOGPU[i], dynamicAmpGOGRGPU[i], gIGRGPU[i], gIGRGPUP[i],
        grConGOOutGRGPU[i], grConGOOutGRGPUP[i], numGOInPerGRGPU[i],
        gIGRSumGPU[i], gIDirectGPU[i], gISpilloverGPU[i], gDirectDecGOtoGR,
        gogrW, gSpilloverDecGOtoGR, gIncFracSpilloverGOtoGR);
#ifdef DEBUGOUT
    error = cudaGetLastError();
    cerr << "runUpdateGOInGRCUDA: kernel launch for gpu #" << i << ": "
         << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::runUpdateGOInGRDepressionCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    callUpdateGOInGRDepressionOPKernel(
        sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB, num_mf,
        depAmpGOGPU[i], grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
        numGOInPerGRGPU[i], depAmpGOGRGPU[i]);
#ifdef DEBUGOUT
    error = cudaGetLastError();
    cerr << "runUpdateMFInGRDepressionCUDA: kernel launch for gpu #" << i
         << ": " << cudaGetErrorString(error) << endl;
#endif
  }
}
void InNet::runUpdateGOInGRDynamicSpillCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    callUpdateGOInGRDynamicSpillOPKernel(
        sts[i][streamN], updateGOInGRNumBlocks, updateGOInGRNumGRPerB, num_mf,
        dynamicAmpGOGPU[i], grConGOOutGRGPU[i], grConGOOutGRGPUP[i],
        numGOInPerGRGPU[i], dynamicAmpGOGRGPU[i]);
#ifdef DEBUGOUT
    error = cudaGetLastError();
    cerr << "runUpdateMFInGRDynamicSpillCUDA: kernel launch for gpu #" << i
         << ": " << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    callUpdateGROutGOKernel(
        sts[i][streamN], updateGRGOOutNumGRRows, updateGRGOOutNumGRPerR, num_go,
        apBufGRGPU[i], grInputGOGPU[i], grInputGOGPUP[i], delayGOMasksGRGPU[i],
        delayGOMasksGRGPUP[i], grConGROutGOGPU[i], grConGROutGOGPUP[i],
        numGOOutPerGRGPU[i]);
#ifdef DEBUGOUT
    error = cudaGetLastError();
    cerr << "runUpdateGROutGOCUDA: kernel launch for gpu #" << i << ": "
         << cudaGetErrorString(error) << endl;
#endif
  }
}

void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN) {
  cpyGRGOSumGPUtoHostCUDA(sts, streamN, grInputGOSumH);
}

void InNet::cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN,
                                    uint32_t **grInputGOSumHost) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);

    error = cudaMemcpyAsync(grInputGOSumHost[i], grInputGOSumGPU[i],
                            num_go * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                            sts[i][streamN]);
  }
}

void InNet::runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN,
                                   uint32_t t) {
  cudaError_t error;
  for (int i = 0; i < numGPUs; i++) {
    error = cudaSetDevice(i + gpuIndStart);
    if (t % (uint32_t)tsPerHistBinGR == 0) {
      callUpdateGRHistKernel(sts[i][streamN], updateGRHistNumBlocks,
                             updateGRHistNumGRPerB, apBufGRGPU[i],
                             historyGRGPU[i], apBufGRHistMask);
    }
  }
}

/* =========================== PROTECTED FUNCTIONS =============================
 */

void InNet::initCUDA() {
  cudaError_t error;
  int maxNumGPUs;

  error = cudaGetDeviceCount(&maxNumGPUs);
  LOG_DEBUG("Maximum number of CUDA devices: %d", maxNumGPUs);
  LOG_DEBUG("%s", cudaGetErrorString(error));
  LOG_DEBUG("Number of CUDA devices actually used: %d", numGPUs);
  LOG_DEBUG("Lowest CUDA device index: %d", gpuIndStart);

  numGRPerGPU = num_gr / numGPUs;

  // fiddle with these limits in just the right way and you slow the sim
  // down greatly, or speed it up a little bit
  calcGRActNumGRPerB = 512;
  calcGRActNumBlocks = numGRPerGPU / calcGRActNumGRPerB;

  updateGRGOOutNumGRPerR = 512 * (num_go > 512) + num_go * (num_go <= 512);
  updateGRGOOutNumGRRows = numGRPerGPU / updateGRGOOutNumGRPerR;

  sumGRGOOutNumGOPerB = 1024 * (num_go > 1024) + num_go * (num_go <= 1024);
  sumGRGOOutNumBlocks = num_go / sumGRGOOutNumGOPerB;

  updateMFInGRNumGRPerB = 1024 * (num_mf > 1024) + (num_mf <= 1024) * num_mf;
  updateMFInGRNumBlocks = numGRPerGPU / updateMFInGRNumGRPerB;

  updateUBCInGRNumGRPerB =
      1024 * (num_ubc > 1024) + (num_ubc <= 1024) * num_ubc;
  updateUBCInGRNumBlocks = numGRPerGPU / updateUBCInGRNumGRPerB;

  updateGOInGRNumGRPerB = 1024 * (num_go >= 1024) + (num_go < 1024) * num_go;
  updateGOInGRNumBlocks = numGRPerGPU / updateGOInGRNumGRPerB;

  updateGRHistNumGRPerB = 1024;
  updateGRHistNumBlocks = numGRPerGPU / updateGRHistNumGRPerB;

  LOG_DEBUG("Initializing per-cell cuda vars...");

  // allocate gpu memory, initialize or copy from host vars
  initMFCUDA();
  LOG_DEBUG("Initialized MF CUDA");
  LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));
  initGRCUDA();
  LOG_DEBUG("Initialized GR CUDA");
  LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));
  initGOCUDA();
  LOG_DEBUG("Initialized GO CUDA");
  LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));
  LOG_DEBUG("Finished initializing per-cell cuda variables.");
}

void InNet::initMFCUDA() {
  apMFH = new uint32_t *[numGPUs];
  depAmpMFH = new float *[numGPUs];
  apMFGPU = new uint32_t *[numGPUs];
  depAmpMFGPU = new float *[numGPUs];

  LOG_DEBUG("Allocating MF cuda variables...");
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMalloc((void **)&apMFGPU[i], num_mf * sizeof(uint32_t));
    cudaMallocHost((void **)&apMFH[i], num_mf * sizeof(uint32_t));
    cudaMalloc((void **)&(depAmpMFGPU[i]), num_mf * sizeof(float));
    cudaMallocHost((void **)&depAmpMFH[i], num_mf * sizeof(float));
    cudaDeviceSynchronize();
  }
  LOG_DEBUG("Finished MF variable cuda allocation");
  LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

  // initialize MF GPU variables
  LOG_DEBUG("Initializing MF cuda variables...");
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMemset(apMFH[i], 0, num_mf * sizeof(uint32_t));
    cudaMemset(depAmpMFH[i], 1, num_mf * sizeof(float));
    cudaMemset(apMFGPU[i], 0, num_mf * sizeof(uint32_t));
    cudaMemset(depAmpMFGPU[i], 1, num_mf * sizeof(float));
    cudaDeviceSynchronize();
  }
  // end copying to GPU
  LOG_DEBUG("Finished initializing MF cuda variables.");
}

void InNet::initGRCUDA() {
  gEGRGPU = new float *[numGPUs];
  gEGRGPUP = new size_t[numGPUs];
  gEGRSumGPU = new float *[numGPUs];
  gEDirectGPU = new float *[numGPUs];
  gESpilloverGPU = new float *[numGPUs];
  apMFtoGRGPU = new int *[numGPUs];
  numMFperGR = new int *[numGPUs];
  depAmpMFGRGPU = new float *[numGPUs];
  depAmpGOGRGPU = new float *[numGPUs];
  dynamicAmpGOGRGPU = new float *[numGPUs];

  gIGRGPU = new float *[numGPUs];
  gIGRGPUP = new size_t[numGPUs];
  gIGRSumGPU = new float *[numGPUs];
  gIDirectGPU = new float *[numGPUs];
  gISpilloverGPU = new float *[numGPUs];

  apBufGRGPU = new uint32_t *[numGPUs];
  outputGRGPU = new uint8_t *[numGPUs];
  apGRGPU = new uint32_t *[numGPUs];

  threshGRGPU = new float *[numGPUs];
  vGRGPU = new float *[numGPUs];
  gKCaGRGPU = new float *[numGPUs];
  gLeakGRGPU = new float *[numGPUs];
  gNMDAGRGPU = new float *[numGPUs];
  gNMDAIncGRGPU = new float *[numGPUs];
  historyGRGPU = new uint64_t *[numGPUs];

  delayGOMasksGRGPU = new uint32_t *[numGPUs];
  delayGOMasksGRGPUP = new size_t[numGPUs];

  numGOOutPerGRGPU = new int32_t *[numGPUs];
  grConGROutGOGPU = new uint32_t *[numGPUs];
  grConGROutGOGPUP = new size_t[numGPUs];

  numGOInPerGRGPU = new int32_t *[numGPUs];
  grConGOOutGRGPU = new uint32_t *[numGPUs];
  grConGOOutGRGPUP = new size_t[numGPUs];

  numMFInPerGRGPU = new int32_t *[numGPUs];
  grConMFOutGRGPU = new uint32_t *[numGPUs];
  grConMFOutGRGPUP = new size_t[numGPUs];

  // NOTE: debating whether to make this page-locked mem or not (06/25/2022)
  outputGRH = new uint8_t[num_gr];
  memset(outputGRH, 0, num_gr * sizeof(uint8_t));

  LOG_DEBUG("Allocating GR cuda variables...");

  // allocate memory for GPU
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMallocPitch((void **)&gEGRGPU[i], (size_t *)&gEGRGPUP[i],
                    numGRPerGPU * sizeof(float), max_num_p_gr_from_mf_to_gr);
    cudaMalloc((void **)&gEGRSumGPU[i], numGRPerGPU * sizeof(float));
    cudaMalloc((void **)&gEDirectGPU[i], numGRPerGPU * sizeof(float));
    cudaMalloc((void **)&gESpilloverGPU[i], numGRPerGPU * sizeof(float));
    cudaMalloc((void **)&apMFtoGRGPU[i], numGRPerGPU * sizeof(int));
    cudaMalloc((void **)&numMFperGR[i], numGRPerGPU * sizeof(int));
    cudaMalloc((void **)&depAmpMFGRGPU[i], numGRPerGPU * sizeof(float));
    cudaMalloc((void **)&depAmpGOGRGPU[i], numGRPerGPU * sizeof(float));
    cudaMalloc((void **)&dynamicAmpGOGRGPU[i], numGRPerGPU * sizeof(float));

    cudaMallocPitch((void **)&gIGRGPU[i], (size_t *)&gIGRGPUP[i],
                    numGRPerGPU * sizeof(float), max_num_p_gr_from_go_to_gr);
    cudaMalloc((void **)&gIGRSumGPU[i], numGRPerGPU * sizeof(float));
    cudaMalloc((void **)&gIDirectGPU[i], numGRPerGPU * sizeof(float));
    cudaMalloc((void **)&gISpilloverGPU[i], numGRPerGPU * sizeof(float));

    cudaMalloc((void **)&apGRGPU[i], numGRPerGPU * sizeof(uint32_t));
    cudaMalloc((void **)&apBufGRGPU[i], numGRPerGPU * sizeof(uint32_t));
    cudaMalloc((void **)&outputGRGPU[i], numGRPerGPU * sizeof(uint8_t));

    cudaMalloc((void **)&threshGRGPU[i], numGRPerGPU * sizeof(float));
    cudaMalloc<float>(&(vGRGPU[i]), numGRPerGPU * sizeof(float));
    cudaMalloc<float>(&(gKCaGRGPU[i]), numGRPerGPU * sizeof(float));
    cudaMalloc<float>(&(gLeakGRGPU[i]), numGRPerGPU * sizeof(float));
    cudaMalloc<float>(&(gNMDAGRGPU[i]), numGRPerGPU * sizeof(float));
    cudaMalloc<float>(&(gNMDAIncGRGPU[i]), numGRPerGPU * sizeof(float));
    cudaMalloc((void **)&historyGRGPU[i], numGRPerGPU * sizeof(uint64_t));

    // variables for conduction delays
    cudaMallocPitch((void **)&delayGOMasksGRGPU[i],
                    (size_t *)&delayGOMasksGRGPUP[i],
                    numGRPerGPU * sizeof(uint32_t), max_num_p_gr_from_gr_to_go);
    // end conduction delay

    // connectivity
    cudaMalloc((void **)&numGOOutPerGRGPU[i], numGRPerGPU * sizeof(int32_t));
    cudaMallocPitch((void **)&grConGROutGOGPU[i],
                    (size_t *)&grConGROutGOGPUP[i],
                    numGRPerGPU * sizeof(uint32_t), max_num_p_gr_from_gr_to_go);

    cudaMalloc((void **)&numGOInPerGRGPU[i], numGRPerGPU * sizeof(int32_t));
    cudaMallocPitch((void **)&grConGOOutGRGPU[i],
                    (size_t *)&grConGOOutGRGPUP[i],
                    numGRPerGPU * sizeof(uint32_t), max_num_p_gr_from_go_to_gr);

    cudaMalloc((void **)&numMFInPerGRGPU[i], numGRPerGPU * sizeof(int32_t));
    cudaMallocPitch((void **)&grConMFOutGRGPU[i],
                    (size_t *)&grConMFOutGRGPUP[i],
                    numGRPerGPU * sizeof(uint32_t), max_num_p_gr_from_mf_to_gr);
    // end connectivity

    cudaDeviceSynchronize();
  }
  LOG_DEBUG("Finished GR variable cuda allocation");
  LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

  LOG_DEBUG(
      "Initializing transposed copies of act state and con state vars...");

  // create a transposed copy of the matrices from activity state and
  // connectivity
  for (int i = 0; i < max_num_p_gr_from_go_to_gr; i++) {
    for (int j = 0; j < num_gr; j++) {
      gGOGRT[i][j] = as->gGOGR[j * max_num_p_gr_from_go_to_gr + i];
      pGRfromGOtoGRT[i][j] = cs->pGRfromGOtoGR[j][i];
    }
  }

  for (int i = 0; i < max_num_p_gr_from_mf_to_gr; i++) {
    for (int j = 0; j < num_gr; j++) {
      gMFGRT[i][j] = as->gMFGR[j * max_num_p_gr_from_mf_to_gr + i];
      pGRfromMFtoGRT[i][j] = cs->pGRfromMFtoGR[j][i];
    }
  }

  for (int i = 0; i < max_num_p_gr_from_gr_to_go; i++) {
    for (int j = 0; j < num_gr; j++) {
      pGRDelayfromGRtoGOT[i][j] = cs->pGRDelayMaskfromGRtoGO[j][i];
      pGRfromGRtoGOT[i][j] = cs->pGRfromGRtoGO[j][i];
    }
  }

  LOG_DEBUG("Finished transposition of act state and con state vars.");

  // initialize GR GPU variables
  LOG_DEBUG("Initializing GR cuda variables...");

  for (int i = 0; i < numGPUs; i++) {
    int cpyStartInd = numGRPerGPU * i;
    int cpySize = numGRPerGPU;
    cudaSetDevice(i + gpuIndStart);

    cudaMemcpy(gKCaGRGPU[i], &(as->gKCaGR[cpyStartInd]),
               cpySize * sizeof(float), cudaMemcpyHostToDevice);

    for (int j = 0; j < max_num_p_gr_from_mf_to_gr; j++) {
      cudaMemcpy((void *)((char *)gEGRGPU[i] + j * gEGRGPUP[i]),
                 &gMFGRT[j][cpyStartInd], cpySize * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy((void *)((char *)grConMFOutGRGPU[i] + j * grConMFOutGRGPUP[i]),
                 &pGRfromMFtoGRT[j][cpyStartInd], cpySize * sizeof(uint32_t),
                 cudaMemcpyHostToDevice);
    }

    cudaMemcpy(vGRGPU[i], &(as->vGR[cpyStartInd]), cpySize * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gEGRSumGPU[i], &(as->gMFSumGR[cpyStartInd]),
               cpySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gEDirectGPU[i], 0.0, cpySize * sizeof(float));
    cudaMemset(gESpilloverGPU[i], 0.0, cpySize * sizeof(float));
    cudaMemcpy(apMFtoGRGPU[i], &(as->apMFtoGR[cpyStartInd]),
               cpySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(numMFperGR[i], &(cs->numpGRfromMFtoGR[cpyStartInd]),
               cpySize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(depAmpMFGRGPU[i], 1.0, cpySize * sizeof(float));
    cudaMemset(depAmpGOGRGPU[i], 1.0, cpySize * sizeof(float));
    cudaMemset(dynamicAmpGOGRGPU[i], 0.0, cpySize * sizeof(float));

    for (int j = 0; j < max_num_p_gr_from_go_to_gr; j++) {
      cudaMemcpy((void *)((char *)gIGRGPU[i] + j * gIGRGPUP[i]),
                 &gGOGRT[j][cpyStartInd], cpySize * sizeof(float),
                 cudaMemcpyHostToDevice);
      cudaMemcpy((void *)((char *)grConGOOutGRGPU[i] + j * grConGOOutGRGPUP[i]),
                 &pGRfromGOtoGRT[j][cpyStartInd], cpySize * sizeof(uint32_t),
                 cudaMemcpyHostToDevice);
    }

    cudaMemcpy(gIGRSumGPU[i], &(as->gGOSumGR[cpyStartInd]),
               cpySize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(gIDirectGPU[i], 0.0, cpySize * sizeof(float));
    cudaMemset(gISpilloverGPU[i], 0.0, cpySize * sizeof(float));

    cudaMemcpy(apBufGRGPU[i], &(as->apBufGR[cpyStartInd]),
               cpySize * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(threshGRGPU[i], &(as->threshGR[cpyStartInd]),
               cpySize * sizeof(float), cudaMemcpyHostToDevice);

    // TODO: place initial value of gLeak into activityparams file and actually
    // use that (since its not default val of float)
    cudaMemset(gLeakGRGPU[i], 0.11, cpySize * sizeof(float));
    cudaMemset(gNMDAGRGPU[i], 0.0, cpySize * sizeof(float));
    cudaMemset(gNMDAIncGRGPU[i], 0.0, cpySize * sizeof(float));

    for (int j = 0; j < max_num_p_gr_from_gr_to_go; j++) {
      cudaMemcpy(
          (void *)((char *)delayGOMasksGRGPU[i] + j * delayGOMasksGRGPUP[i]),
          &pGRDelayfromGRtoGOT[j][cpyStartInd], cpySize * sizeof(float),
          cudaMemcpyHostToDevice);
      cudaMemcpy((void *)((char *)grConGROutGOGPU[i] + j * grConGROutGOGPUP[i]),
                 &pGRfromGRtoGOT[j][cpyStartInd],
                 cpySize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    // Basket cell stuff
    cudaMemcpy(numGOOutPerGRGPU[i], &(cs->numpGRfromGRtoGO[cpyStartInd]),
               cpySize * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaMemcpy(numGOInPerGRGPU[i], &(cs->numpGRfromGOtoGR[cpyStartInd]),
               cpySize * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaMemcpy(numMFInPerGRGPU[i], &(cs->numpGRfromMFtoGR[cpyStartInd]),
               cpySize * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(historyGRGPU[i], &(as->historyGR[cpyStartInd]),
               cpySize * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaMemset(outputGRGPU[i], 0, cpySize * sizeof(uint8_t));
    cudaMemset(apGRGPU[i], 0, cpySize * sizeof(uint32_t));

    cudaDeviceSynchronize();
  }
  // end copying to GPU
  LOG_DEBUG("Finished initializing GR cuda variables.");
}

void InNet::initGOCUDA() {
  // FIXME: change the types of some of these arrays (see joe's biasManip sim)
  grInputGOSumH = new uint32_t *[numGPUs];
  apGOH = new uint32_t *[numGPUs];
  apGOGPU = new uint32_t *[numGPUs];
  grInputGOGPU = new uint32_t *[numGPUs];
  grInputGOGPUP = new size_t[numGPUs];
  grInputGOSumGPU = new uint32_t *[numGPUs];
  depAmpGOH = new float *[numGPUs];
  depAmpGOGPU = new float *[numGPUs];
  dynamicAmpGOH = new float *[numGPUs];
  dynamicAmpGOGPU = new float *[numGPUs];

  LOG_DEBUG("Allocating GO cuda variables...");
  counter = new int[num_go];
  memset(counter, 0, num_go * sizeof(int));

  // allocate host and device memory
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMallocHost((void **)&grInputGOSumH[i], num_go * sizeof(uint32_t));
    cudaMallocHost((void **)&apGOH[i], num_go * sizeof(uint32_t));
    cudaMallocHost((void **)&depAmpGOH[i], num_go * sizeof(float));
    cudaMallocHost((void **)&dynamicAmpGOH[i], num_go * sizeof(float));

    // allocate gpu memory
    cudaMalloc((void **)&apGOGPU[i], num_go * sizeof(uint32_t));
    cudaMalloc((void **)&depAmpGOGPU[i], num_go * sizeof(float));
    cudaMalloc((void **)&dynamicAmpGOGPU[i], num_go * sizeof(float));

    cudaMallocPitch((void **)&grInputGOGPU[i], (size_t *)&grInputGOGPUP[i],
                    num_go * sizeof(uint32_t), updateGRGOOutNumGRRows);
    cudaMalloc((void **)&grInputGOSumGPU[i], num_go * sizeof(uint32_t));

    cudaDeviceSynchronize();
  }
  LOG_DEBUG("Finished GO variable cuda allocation");
  LOG_DEBUG("Last error: %s", cudaGetErrorString(cudaGetLastError()));

  // initialize GO vars
  LOG_DEBUG("Initializing GO cuda variables...");
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMemset(apGOH[i], 0, num_go * sizeof(uint32_t));
    cudaMemset(depAmpGOH[i], 1, num_go * sizeof(float));
    cudaMemset(dynamicAmpGOH[i], 1, num_go * sizeof(float));
    cudaMemset(grInputGOSumH[i], 0, num_go * sizeof(uint32_t));

    cudaMemset(apGOGPU[i], 0, num_go * sizeof(uint32_t));
    cudaMemset(depAmpGOGPU[i], 1, num_go * sizeof(float));
    cudaMemset(dynamicAmpGOGPU[i], 1, num_go * sizeof(float));

    for (int j = 0; j < updateGRGOOutNumGRRows; j++) {
      cudaMemset(((char *)grInputGOGPU[i] + j * grInputGOGPUP[i]), 0,
                 num_go * sizeof(uint32_t));
    }
    cudaMemset(grInputGOSumGPU[i], 0, num_go * sizeof(uint32_t));

    cudaDeviceSynchronize();
  }
  LOG_DEBUG("Finished initializing GO cuda variables.");
}

/* =========================== PRIVATE FUNCTIONS =============================
 */

template <typename Type>
cudaError_t InNet::getGRGPUData(Type **gpuData, Type *hostData) {
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i + gpuIndStart);
    cudaMemcpyAsync((void *)&hostData[i * numGRPerGPU], gpuData[i],
                    numGRPerGPU * sizeof(Type), cudaMemcpyDeviceToHost);
  }
  return cudaGetLastError();
}
