/*
 * InNet.h
 *
 *  Created on: Jun 21, 2011
 *      Author: consciousness
 *
 *  InNet class is used to update spiking activity within the input network
 *  of the simulation, which includes mf, gr, and go cells. mf spikes are
 *  poisson generated separately and are copied in to this class. connectivity
 *  arrays are used to relay information originating from mf to go and gr, and
 *  from gr to go
 */

#ifndef INNET_H_
#define INNET_H_

#include <cuda.h>
#include <omp.h>

#include "innetactivitystate.h"
#include "innetconnectivitystate.h"
#include "kernels.h"
#include <cstdint>

class InNet {
public:
  InNet();
  InNet(InNetConnectivityState *cs, InNetActivityState *as, int gpuIndStart,
        int numGPUs);
  ~InNet();

  void writeToState();

  // silly little functions used to send a const (immutable) pointer to the
  // caller to read this data
  const uint8_t *exportAPGO();
  const uint8_t *exportHistMF();
  const uint8_t *exportAPGR();

  const uint32_t *exportSumGRInputGO();
  const float *exportSumGOInputGO();

  // used when initializing mzone
  uint32_t **getApBufGRGPUPointer();
  uint64_t **getHistGRGPUPointer();

  uint32_t **getGRInputGOSumHPointer();

  const float *exportGESumGR();
  const float *exportGISumGR();
  const float *exportgSum_MFGO();
  const float *exportgSum_GRGO();
  const float *exportgSum_GRGONMDA();

  void updateMFActivties(const uint8_t *actInMF);
  void calcGOActivities(bool inCS);

  void updateMFtoGROut();
  void updateMFtoGOOut();
  void updateGOtoGROutParameters(float spillFrac);
  void updateGOtoGOOut();
  void resetMFHist(uint32_t t);

  void runGRActivitiesCUDA(cudaStream_t **sts, int streamN);
  void runSumGRGOOutCUDA(cudaStream_t **sts, int streamN);
  void cpyDepAmpMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);
  void cpyAPMFHosttoGPUCUDA(cudaStream_t **sts, int streamN);

  void cpyDepAmpUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN);
  void cpyAPUBCHosttoGPUCUDA(cudaStream_t **sts, int streamN);

  void cpyDepAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN);
  void cpyDynamicAmpGOGRHosttoGPUCUDA(cudaStream_t **sts, int streamN);
  void cpyAPGOHosttoGPUCUDA(cudaStream_t **sts, int streamN);
  void runUpdateMFInGRCUDA(cudaStream_t **sts, int streamN);
  void runUpdateMFInGRDepressionCUDA(cudaStream_t **sts, int streamN);

  void runUpdateUBCInGRDepressionCUDA(cudaStream_t **sts, int streamN);
  void runUpdateUBCInGRCUDA(cudaStream_t **sts, int streamN);

  void runUpdateGOInGRCUDA(cudaStream_t **sts, int streamN);
  void runUpdateGOInGRDepressionCUDA(cudaStream_t **sts, int streamN);
  void runUpdateGOInGRDynamicSpillCUDA(cudaStream_t **sts, int streamN);

  void runUpdateGROutGOCUDA(cudaStream_t **sts, int streamN);

  void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN);
  void cpyGRGOSumGPUtoHostCUDA(cudaStream_t **sts, int streamN,
                               uint32_t **grInputGOSumHost);
  void runUpdateGRHistoryCUDA(cudaStream_t **sts, int streamN, uint32_t t);

protected:
  // following two pointers are copied in during construction of the class
  InNetConnectivityState *cs;
  InNetActivityState *as;

  int gpuIndStart;
  int numGPUs;
  int numGRPerGPU;

  // cuda related variables. some are redundant
  unsigned int calcGRActNumGRPerB;
  unsigned int calcGRActNumBlocks;

  unsigned int updateGRGOOutNumGRPerR;
  unsigned int updateGRGOOutNumGRRows;

  unsigned int sumGRGOOutNumGOPerB;
  unsigned int sumGRGOOutNumBlocks;

  unsigned int updateMFInGRNumGRPerB;
  unsigned int updateMFInGRNumBlocks;

  unsigned int updateUBCInGRNumGRPerB;
  unsigned int updateUBCInGRNumBlocks;

  unsigned int updateGOInGRNumGRPerB;
  unsigned int updateGOInGRNumBlocks;

  unsigned int updateGRHistNumGRPerB;
  unsigned int updateGRHistNumBlocks;

  // UBC
  uint32_t **apUBCH;
  uint32_t **apUBCGPU;

  float **depAmpUBCH;
  float **depAmpUBCGPU;
  float **depAmpUBCGRGPU;

  // mossy fibers
  const uint8_t *apMFOut;

  // gpu related variables
  uint32_t **apMFH;
  uint32_t **apMFGPU;

  float **depAmpMFH;
  float **depAmpMFGPU;
  float **depAmpMFGRGPU;

  int **numMFperGR;
  int **numUBCperGR;
  // end gpu related variables

  // golgi cell variables
  // gpu related variables

  uint32_t **apGOH;
  uint32_t **grInputGOSumH;

  float **depAmpGOH;
  float **dynamicAmpGOH;
  float *biasGOBkgdH;
  float *biasGOCSH;

  int *counter;

  uint32_t **apGOGPU;
  uint32_t **grInputGOGPU;
  uint32_t **grInputGOSumGPU;

  size_t *grInputGOGPUP;

  float **depAmpGOGRGPU;
  float **depAmpGOGPU;
  float **dynamicAmpGOGPU;
  float **dynamicAmpGOGRGPU;
  // end gpu variables

  uint32_t *sumGRInputGO;

  float *sumInputGOGABASynDepGO;
  float tempGIncGRtoGO;
  // end golgi cell variables

  // granule cell variables
  float **gMFGRT;
  float **gUBCGRT;
  float **gGOGRT;

  uint32_t **pGRDelayfromGRtoGOT;
  uint32_t **pGRfromMFtoGRT;
  uint32_t **pGRfromUBCtoGRT;
  uint32_t **pGRfromGOtoGRT;
  uint32_t **pGRfromGRtoGOT;

  uint32_t apBufGRHistMask;
  // gpu related variables
  // host variables
  uint8_t *outputGRH;
  // end host variables

  float **gEGRGPU;
  size_t *gEGRGPUP;
  float **gEGRSumGPU;
  float **gEDirectGPU;
  float **gESpilloverGPU;
  float **gIDirectGPU;
  float **gISpilloverGPU;
  int **apMFtoGRGPU;
  int **apUBCtoGRGPU;
  float **gUBC_EGRGPU;
  size_t *gUBC_EGRGPUP;
  float **gUBC_EGRSumGPU;
  float **gUBC_EDirectGPU;
  float **gUBC_ESpilloverGPU;

  float **gIGRGPU;
  size_t *gIGRGPUP;
  float **gIGRSumGPU;

  uint32_t **apBufGRGPU;
  uint8_t **outputGRGPU;
  uint32_t **apGRGPU;

  float **threshGRGPU;
  float **vGRGPU;
  float **gLeakGRGPU;
  float **gNMDAGRGPU;
  float **gNMDAIncGRGPU;
  float **gKCaGRGPU;
  uint64_t **historyGRGPU;

  // conduction delays
  uint32_t **delayGOMasksGRGPU;
  size_t *delayGOMasksGRGPUP;

  // connectivity
  int contVar = 1;
  int contVarOther = 1;
  float sumGOFR = 0;
  float sumExScaler = 0;
  int slowCounter = 0;
  int timeStepPOne;
  int thingCounter = 0;

  int32_t **numGOOutPerGRGPU;
  uint32_t **grConGROutGOGPU;
  size_t *grConGROutGOGPUP;

  int32_t **numGOInPerGRGPU;
  uint32_t **grConGOOutGRGPU;
  size_t *grConGOOutGRGPUP;

  int32_t **numMFInPerGRGPU;
  uint32_t **grConMFOutGRGPU;
  size_t *grConMFOutGRGPUP;

  int32_t **numUBCInPerGRGPU;
  uint32_t **grConUBCOutGRGPU;
  size_t *grConUBCOutGRGPUP;

  // end gpu variables
  // end granule cell variables

  void initCUDA();
  void initMFCUDA();
  void initGRCUDA();
  void initGOCUDA();
  void initSCCUDA();

private:
  template <typename Type>
  cudaError_t getGRGPUData(Type **gpuData, Type *hostData);
};

#endif /* INNET_H_ */
