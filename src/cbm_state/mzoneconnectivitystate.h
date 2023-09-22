/*
 * mzoneconnectivitystate.h
 *
 *  Created on: Nov 21, 2012
 *      Author: consciousness
 */

#ifndef MZONECONNECTIVITYSTATE_H_
#define MZONECONNECTIVITYSTATE_H_

#include <cstdint>
#include <fstream>

class MZoneConnectivityState {
public:
  MZoneConnectivityState();
  MZoneConnectivityState(int randSeed);
  MZoneConnectivityState(std::fstream &infile);
  ~MZoneConnectivityState();

  void readState(std::fstream &infile);
  void writeState(std::fstream &outfile);

  // granule cells
  uint32_t *pGRDelayMaskfromGRtoBSP;

  // basket cells
  uint32_t **pBCfromBCtoPC;
  uint32_t **pBCfromPCtoBC;

  // stellate cells
  uint32_t **pSCfromSCtoPC;

  // purkinje cells
  uint32_t **pPCfromBCtoPC;
  uint32_t **pPCfromPCtoBC;
  uint32_t **pPCfromSCtoPC;
  uint32_t **pPCfromPCtoNC;
  uint32_t *pPCfromIOtoPC;

  // nucleus cells
  uint32_t **pNCfromPCtoNC;
  uint32_t **pNCfromNCtoIO;
  uint32_t **pNCfromMFtoNC;

  // inferior olivary cells
  uint32_t **pIOfromIOtoPC;
  uint32_t **pIOfromNCtoIO;
  uint32_t **pIOInIOIO;
  uint32_t **pIOOutIOIO;

private:
  void allocateMemory();
  void initializeVals();
  void deallocMemory();
  void stateRW(bool read, std::fstream &file);

  void assignGRDelays();
  void connectBCtoPC();
  void connectPCtoBC();
  void connectSCtoPC();
  void connectPCtoNC(int randSeed);
  void connectNCtoIO();
  void connectMFtoNC();
  void connectIOtoPC();
  void connectIOtoIO();
};

#endif /* MZONECONNECTIVITYSTATE_H_ */
