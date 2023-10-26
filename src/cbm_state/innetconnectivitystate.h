/*
 * innetconnectivitystate.h
 *
 *  Created on: Nov 6, 2012
 *      Author: consciousness
 *
 *  Provides algorithms for making the following connections:
 *
 *    mf -> gl
 *    gl -> gr
 *    gl -> go
 *    go -> gl
 *
 *  as well as the translations to the "direct" connections:
 *
 *    mf -> gr
 *    mf -> go
 *    gr -> go
 *    go -> gr
 */

#ifndef INNETCONNECTIVITYSTATE_H_
#define INNETCONNECTIVITYSTATE_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <sstream>
#include <string.h>

#include "dynamic2darray.h"
#include "file_utility.h"
#include "sfmt.h"
#include <cstdint>

class InNetConnectivityState {
public:
  InNetConnectivityState();
  InNetConnectivityState(int randSeed);
  InNetConnectivityState(std::fstream &infile);
  ~InNetConnectivityState();

  void readState(std::fstream &infile);
  void writeState(std::fstream &outfile);

  // file IO arrays

  // who the MFs connect to...
  void numPMFfromMFtoGRRW(std::fstream &file, bool read);
  void pMFfromMFtoGRRW(std::fstream &file, bool read);

  void numPMFfromMFtoGORW(std::fstream &file, bool read);
  void pMFfromMFtoGORW(std::fstream &file, bool read);

  // who the GOs connect to...
  void numPGOfromMFtoGORW(std::fstream &file, bool read);
  void pGOfromMFtoGORW(std::fstream &file, bool read);

  void numPGOfromGOtoGRRW(std::fstream &file, bool read);
  void pGOfromGOtoGRRW(std::fstream &file, bool read);

  void numPGOfromGRtoGORW(std::fstream &file, bool read);
  void pGOfromGRtoGORW(std::fstream &file, bool read);

  // receiving GOs ...
  void numPGOInfromGOtoGORW(std::fstream &file, bool read);
  void pGOInfromGOtoGORW(std::fstream &file, bool read);

  // outgoing GOs...
  void numPGOOutfromGOtoGORW(std::fstream &file, bool read);
  void pGOOutfromGOtoGORW(std::fstream &file, bool read);

  // receiving GO coupling...
  void numPGOCoupInfromGOtoGORW(std::fstream &file, bool read);
  void pGOCoupInfromGOtoGORW(std::fstream &file, bool read);

  // outgoing GO coupling...
  void numPGOCoupOutfromGOtoGORW(std::fstream &file, bool read);
  void pGOCoupOutfromGOtoGORW(std::fstream &file, bool read);

  // receiving GO coupling *coefficients*...
  void pGOCoupOutGOGOCCoeffRW(std::fstream &file, bool read);

  // outgoing GO coupling *coefficients*...
  void pGOCoupInGOGOCCoeffRW(std::fstream &file, bool read);

  // who the GRs connect to...
  void numPGRfromGRtoGORW(std::fstream &file, bool read);
  void pGRfromGRtoGORW(std::fstream &file, bool read);
  void numPGRfromGOtoGRRW(std::fstream &file, bool read);
  void pGRfromGOtoGRRW(std::fstream &file, bool read);
  void numPGRfromMFtoGRRW(std::fstream &file, bool read);
  void pGRfromMFtoGRRW(std::fstream &file, bool read);

  // glomerulus
  bool *haspGLfromMFtoGL;
  int *numpGLfromGLtoGO;
  int **pGLfromGLtoGO;
  int *numpGLfromGOtoGL;
  int **pGLfromGOtoGL;
  int *numpGLfromGLtoGR;
  int **pGLfromGLtoGR;
  int *pGLfromMFtoGL;
  int *numpMFfromMFtoGL;
  int **pMFfromMFtoGL;
  int *numpMFfromMFtoGR;
  int **pMFfromMFtoGR;
  int *numpMFfromMFtoGO;
  int **pMFfromMFtoGO;

  // golgi
  int *numpGOfromGLtoGO;
  int **pGOfromGLtoGO;
  int *numpGOfromGOtoGL;
  int **pGOfromGOtoGL;
  int *numpGOfromMFtoGO;
  int **pGOfromMFtoGO;
  int *numpGOfromGOtoGR;
  int **pGOfromGOtoGR;
  int *numpGOfromGRtoGO;
  int **pGOfromGRtoGO;

  // coincidentally, numcongotogo == maxnumpgogabaingogo
  int *numpGOGABAInGOGO;
  int **pGOGABAInGOGO;
  int *numpGOGABAOutGOGO;
  int **pGOGABAOutGOGO;

  // go <-> go gap junctions
  int *numpGOCoupInGOGO;
  int **pGOCoupInGOGO;
  int *numpGOCoupOutGOGO;
  int **pGOCoupOutGOGO;
  float **pGOCoupOutGOGOCCoeff;
  float **pGOCoupInGOGOCCoeff;

  // granule
  int *numpGRfromGLtoGR;
  int **pGRfromGLtoGR;
  int *numpGRfromGRtoGO;
  int **pGRfromGRtoGO;
  int **pGRDelayMaskfromGRtoGO;
  int *numpGRfromGOtoGR;
  int **pGRfromGOtoGR;
  int *numpGRfromMFtoGR;
  int **pGRfromMFtoGR;

protected:
  void allocateMemory();
  void initializeVals();
  void deallocMemory();
  void stateRW(bool read, std::fstream &file);

  void connectMFGL_noUBC();
  void connectGLGR(CRandomSFMT &randGen);
  void connectGRGO();
  void connectGOGL(CRandomSFMT &randGen);
  void connectGOGODecayP(CRandomSFMT &randGen);
  void connectGOGO_GJ(CRandomSFMT &randGen);
  void translateMFGL();
  void translateGOGL();
  void assignGRDelays();
};

#endif /* INNETCONNECTIVITYSTATE_H_ */
