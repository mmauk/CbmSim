/*
 * kernels.h
 *
 *  Created on: Jun 6, 2011
 *      Author: consciousness
 */

#ifndef KERNELS_H_
#define KERNELS_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include <cstdint>

void callTestKernel(cudaStream_t &st, float *a, float *b, float *c);

void callGRActKernel(cudaStream_t &st, unsigned int numBlocks,
                     unsigned int numGRPerBlock, float *vGPU, float *gKCaGPU,
                     float *gLeakGRPGU, float *gNMDAGRGPU, float *gNMDAIncGRGPU,
                     float *threshGPU, uint32_t *apBufGPU, uint8_t *apOutGRGPU,
                     uint32_t *apGRGPU, int *apMFtoGRGPU, float *gESumGPU,
                     float *gISumGPU, float eLeak, float eGOIn, float gAMPAInc,
                     float threshBase, float threshMax, float threshDecay);

template <typename Type, bool inMultiP, bool outMultiP>
void callSumKernel(cudaStream_t &st, Type *inGPU, size_t inGPUP,
                   Type *outSumGPU, size_t outSumGPUP, unsigned int nOutCells,
                   unsigned int nOutCols, unsigned int rowLength);

template <typename Type>
void callBroadcastKernel(cudaStream_t &st, Type *broadCastVal, Type *outArray,
                         unsigned int nBlocks, unsigned int rowLength);

void callUpdateGROutGOKernel(cudaStream_t &st, unsigned int numBlocks,
                             unsigned int numGRPerBlock, unsigned int numGO,
                             uint32_t *apBufGPU, uint32_t *grInGOGPU,
                             uint32_t grInGOGPUPitch, uint32_t *delayMasksGPU,
                             uint32_t delayMasksGPUPitch,
                             uint32_t *conGRtoGOGPU, size_t conGRtoGOGPUPitch,
                             int32_t *numGOPerGRGPU);
void callUpdateGROutBCKernel(cudaStream_t &st, unsigned int numBlocks,
                             unsigned int numGRPerBlock, unsigned int numBC,
                             uint32_t *apBufGPU, uint32_t *grInBCGPU,
                             uint32_t grInBCGPUPitch, uint32_t *delayMasksGPU,
                             uint32_t delayMasksGPUPitch,
                             uint32_t *conGRtoBCGPU, size_t conGRtoBCGPUPitch,
                             int32_t *numBCPerGRGPU);

void callSumGRGOOutKernel(cudaStream_t &st, unsigned int numBlocks,
                          unsigned int numGOPerBlock, unsigned int numGROutRows,
                          uint32_t *grInGOGPU, size_t grInGOGPUPitch,
                          uint32_t *grInGOSGPU);
void callSumGRBCOutKernel(cudaStream_t &st, unsigned int numBlocks,
                          unsigned int numGOPerBlock, unsigned int numGROutRows,
                          uint32_t *grInBCGPU, size_t grInBCGPUPitch,
                          uint32_t *grInBCSGPU);

void callUpdateGOInGRDepressionOPKernel(
    cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
    unsigned int numInCells, float *depAmpGPU, uint32_t *conInGRGPU,
    size_t conInGRGPUP, int32_t *numInPerGRGPU, float *depAmpGOGRGPU);

void callUpdateMFInGRDepressionOPKernel(
    cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
    unsigned int numInCells, float *depAmpGPU, uint32_t *conInGRGPU,
    size_t conInGRGPUP, int32_t *numInPerGRGPU, int *numMFperGR,
    float *depAmpMFGRGPU);

void callUpdateGOInGRDynamicSpillOPKernel(
    cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
    unsigned int numInCells, float *dynamicAmpGPU, uint32_t *conInGRGPU,
    size_t conInGRGPUP, int32_t *numInPerGRGPU, float *dynamicAmpGOGRGPU);

void callUpdateInGROPKernel(cudaStream_t &st, unsigned int numBlocks,
                            unsigned int numGRPerBlock, unsigned int numInCells,
                            uint32_t *apInGPU, float *dynamicAmpGPU,
                            float *gGPU, size_t gGPUP, uint32_t *conInGRGPU,
                            size_t conInGRGPUP, int32_t *numInPerGRGPU,
                            float *gSumGPU, float *gDirectGPU,
                            float *gSpilloverGPU, float gDecayD, float gIncD,
                            float gDecayS, float gIncFracS);

void callUpdateUBCInGROPKernel(
    cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
    unsigned int numInCells, uint32_t *apInGPU, float *depAmpGPU, float *gGPU,
    size_t gGPUP, uint32_t *conInGRGPU, size_t conInGRGPUP,
    int32_t *numInPerGRGPU, int *apUBCtoGRGPU, float *gSumGPU,
    float *gDirectGPU, float *gSpilloverGPU, float gDecayDirect,
    float gIncDirect, float gDecaySpill, float gIncFracSpill);

void callUpdateMFInGROPKernel(
    cudaStream_t &st, unsigned int numBlocks, unsigned int numGRPerBlock,
    unsigned int numInCells, uint32_t *apInGPU, float *depAmp, float *gGPU,
    size_t gGPUP, uint32_t *conInGRGPU, size_t conInGRGPUP,
    int32_t *numInPerGRGPU, int *apMFtoGRGPU, float *gSumGPU, float *gDirectGPU,
    float *gSpilloverGPU, float gDecayDirect, float gIncDirect,
    float gDecaySpill, float gIncFracSpill);

void callUpdatePFBCSCOutKernel(cudaStream_t &st, unsigned int numBlocks,
                               unsigned int numGRPerBlock, uint32_t *apBufGPU,
                               uint32_t *delayMaskGPU, uint32_t *inPFBCGPU,
                               size_t inPFBCGPUPitch,
                               unsigned int numPFInPerBCP2, uint32_t *inPFSCGPU,
                               size_t inPFSCGPUPitch,
                               unsigned int numPFInPerSCP2);

void callUpdatePFPCOutKernel(cudaStream_t &st, unsigned int numBlocks,
                             unsigned int numGRPerBlock, uint32_t *apBufGPU,
                             uint32_t *delayMaskGPU, float *pfPCSynWGPU,
                             float *inPFPCGPU, size_t inPFPCGPUPitch,
                             unsigned int numPFInPerPCP2);

void callUpdateGRHistKernel(cudaStream_t &st, unsigned int numBlocks,
                            unsigned int numGRPerBlock, uint32_t *apBufGPU,
                            uint64_t *historyGPU, uint32_t apBufGRHistMask);

void callUpdatePFPCPlasticityIOKernel(cudaStream_t &st, unsigned int numBlocks,
                                      unsigned int numGRPerBlock,
                                      float *synWeightGPU, uint64_t *historyGPU,
                                      unsigned int pastBinNToCheck, int offSet,
                                      float pfPCPlastStep);

#endif /* KERNELS_H_ */
