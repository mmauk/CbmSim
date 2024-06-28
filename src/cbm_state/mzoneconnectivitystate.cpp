/*
 * mzoneconnectivitystate.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: consciousness
 */

#include <algorithm>
#include <iostream>
#include <limits.h>

#include "connectivityparams.h"
#include "dynamic2darray.h"
#include "file_utility.h"
#include "logger.h"
#include "mzoneconnectivitystate.h"
#include "sfmt.h"

MZoneConnectivityState::MZoneConnectivityState(int randSeed) {
  LOG_DEBUG("Allocating and initializing mzone connectivity arrays...");
  allocateMemory();
  initializeVals();
  LOG_DEBUG("Initializing mzone connections...");
  LOG_DEBUG("Assigning GR delays");
  assignGRDelays();
  LOG_DEBUG("Connecting BC to PC");
  connectBCtoPC();
  LOG_DEBUG("Connecting PC to BC");
  connectPCtoBC();

  LOG_DEBUG("Connecting SC to Compartment");
  connectSCtoCompart();
  LOG_DEBUG("Connecting Compartment to PC");
  connectCompartToPC();

  LOG_DEBUG("Connecting PC and NC");
  connectPCtoNC(randSeed);
  LOG_DEBUG("Connecting NC and IO");
  connectNCtoIO();
  LOG_DEBUG("Connecting MF and NC");
  connectMFtoNC();
  LOG_DEBUG("Connecting IO and PC");
  connectIOtoPC();
  LOG_DEBUG("Connecting IO and IO");
  connectIOtoIO();
  LOG_DEBUG("Finished making mzone connections.");
}

MZoneConnectivityState::MZoneConnectivityState(std::fstream &infile) {
  allocateMemory();
  stateRW(true, infile);
}

MZoneConnectivityState::~MZoneConnectivityState() { deallocMemory(); }

void MZoneConnectivityState::readState(std::fstream &infile) {
  stateRW(true, infile);
}

void MZoneConnectivityState::writeState(std::fstream &outfile) {
  stateRW(false, outfile);
}

void MZoneConnectivityState::pPCfromPCtoBCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pPCfromPCtoBC[0],
             num_pc * num_p_pc_from_pc_to_bc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pPCfromBCtoPCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pPCfromBCtoPC[0],
             num_pc * num_p_pc_from_bc_to_pc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pPCfromPCtoNCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pPCfromPCtoNC[0],
             num_pc * num_p_pc_from_pc_to_nc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pPCfromIOtoPCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pPCfromIOtoPC, num_pc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pBCfromBCtoPCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pBCfromBCtoPC[0],
             num_bc * num_p_bc_from_bc_to_pc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pBCfromPCtoBCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pBCfromPCtoBC[0],
             num_bc * num_p_bc_from_pc_to_bc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pNCfromPCtoNCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pNCfromPCtoNC[0],
             num_nc * num_p_nc_from_pc_to_nc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pNCfromNCtoIORW(std::fstream &file, bool read) {
  rawBytesRW((char *)pNCfromNCtoIO[0],
             num_nc * num_p_nc_from_nc_to_io * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pNCfromMFtoNCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pNCfromMFtoNC[0],
             num_nc * num_p_nc_from_mf_to_nc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pIOfromIOtoPCRW(std::fstream &file, bool read) {
  rawBytesRW((char *)pIOfromIOtoPC[0],
             num_io * num_p_io_from_io_to_pc * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pIOfromNCtoIORW(std::fstream &file, bool read) {
  rawBytesRW((char *)pIOfromNCtoIO[0],
             num_io * num_p_io_from_nc_to_io * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pIOInIOIORW(std::fstream &file, bool read) {
  rawBytesRW((char *)pIOInIOIO[0],
             num_io * num_p_io_in_io_to_io * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::pIOOutIOIORW(std::fstream &file, bool read) {
  rawBytesRW((char *)pIOOutIOIO[0],
             num_io * num_p_io_out_io_to_io * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::allocateMemory() {
  // granule cells
  pGRDelayMaskfromGRtoBSP = new uint32_t[num_gr];

  // basket cells
  pBCfromBCtoPC = allocate2DArray<uint32_t>(num_bc, num_p_bc_from_bc_to_pc);
  pBCfromPCtoBC = allocate2DArray<uint32_t>(num_bc, num_p_bc_from_pc_to_bc);

  // stellate cells
  pSCfromSCtoCompart =
      allocate2DArray<uint32_t>(num_sc, num_p_sc_from_sc_to_compart);
  pCompartfromSCtoCompart =
      allocate2DArray<uint32_t>(num_compart, num_p_compart_from_sc_to_compart);
  pCompartfromCompartToPC = new uint32_t[num_compart]();
  pPCfromCompartToPC =
      allocate2DArray<uint32_t>(num_pc, max_num_p_pc_from_compart_to_pc);
  numpPCfromCompartToPC = new uint32_t[num_pc]();

  // purkinje cells
  pPCfromBCtoPC = allocate2DArray<uint32_t>(num_pc, num_p_pc_from_bc_to_pc);
  pPCfromPCtoBC = allocate2DArray<uint32_t>(num_pc, num_p_pc_from_pc_to_bc);
  pPCfromPCtoNC = allocate2DArray<uint32_t>(num_pc, num_p_pc_from_pc_to_nc);
  pPCfromIOtoPC = new uint32_t[num_pc]();

  // nucleus cells
  pNCfromPCtoNC = allocate2DArray<uint32_t>(num_nc, num_p_nc_from_pc_to_nc);
  pNCfromNCtoIO = allocate2DArray<uint32_t>(num_nc, num_p_nc_from_nc_to_io);
  pNCfromMFtoNC = allocate2DArray<uint32_t>(num_nc, num_p_nc_from_mf_to_nc);

  // inferior olivary cells
  pIOfromIOtoPC = allocate2DArray<uint32_t>(num_io, num_p_io_from_io_to_pc);
  pIOfromNCtoIO = allocate2DArray<uint32_t>(num_io, num_p_io_from_nc_to_io);
  pIOInIOIO = allocate2DArray<uint32_t>(num_io, num_p_io_in_io_to_io);
  pIOOutIOIO = allocate2DArray<uint32_t>(num_io, num_p_io_out_io_to_io);
}

void MZoneConnectivityState::initializeVals() {
  // granule cells
  std::fill(pGRDelayMaskfromGRtoBSP, pGRDelayMaskfromGRtoBSP + num_gr, 0);

  // basket cells
  std::fill(pBCfromBCtoPC[0],
            pBCfromBCtoPC[0] + num_bc * num_p_bc_from_bc_to_pc, 0);
  std::fill(pBCfromPCtoBC[0],
            pBCfromPCtoBC[0] + num_bc * num_p_bc_from_pc_to_bc, 0);

  // stellate cells
  std::fill(pSCfromSCtoCompart[0],
            pSCfromSCtoCompart[0] + num_sc * num_p_sc_from_sc_to_compart, 0);
  std::fill(pCompartfromSCtoCompart[0],
            pCompartfromSCtoCompart[0] +
                num_compart * num_p_compart_from_sc_to_compart,
            0);
  std::fill(pPCfromCompartToPC[0],
            pPCfromCompartToPC[0] + num_pc * max_num_p_pc_from_compart_to_pc,
            0);

  // purkinje cells
  std::fill(pPCfromBCtoPC[0],
            pPCfromBCtoPC[0] + num_pc * num_p_pc_from_bc_to_pc, 0);
  std::fill(pPCfromPCtoBC[0],
            pPCfromPCtoBC[0] + num_pc * num_p_pc_from_pc_to_bc, 0);
  std::fill(pPCfromPCtoNC[0],
            pPCfromPCtoNC[0] + num_pc * num_p_pc_from_pc_to_nc, 0);

  // nucleus cells
  std::fill(pNCfromPCtoNC[0],
            pNCfromPCtoNC[0] + num_nc * num_p_nc_from_pc_to_nc, 0);
  std::fill(pNCfromNCtoIO[0],
            pNCfromNCtoIO[0] + num_nc * num_p_nc_from_nc_to_io, 0);
  std::fill(pNCfromMFtoNC[0],
            pNCfromMFtoNC[0] + num_nc * num_p_nc_from_mf_to_nc, 0);

  // inferior olivary cells
  std::fill(pIOfromIOtoPC[0],
            pIOfromIOtoPC[0] + num_io * num_p_io_from_io_to_pc, 0);
  std::fill(pIOfromNCtoIO[0],
            pIOfromNCtoIO[0] + num_io * num_p_io_from_nc_to_io, 0);
  std::fill(pIOInIOIO[0], pIOInIOIO[0] + num_io * num_p_io_in_io_to_io, 0);
  std::fill(pIOOutIOIO[0], pIOOutIOIO[0] + num_io * num_p_io_out_io_to_io, 0);
}

void MZoneConnectivityState::deallocMemory() {
  // granule cells
  delete[] pGRDelayMaskfromGRtoBSP;

  // basket cells
  delete2DArray<uint32_t>(pBCfromBCtoPC);
  delete2DArray<uint32_t>(pBCfromPCtoBC);

  // stellate cells
  delete2DArray<uint32_t>(pSCfromSCtoCompart);
  delete2DArray<uint32_t>(pCompartfromSCtoCompart);
  delete[] pCompartfromCompartToPC;
  delete2DArray<uint32_t>(pPCfromCompartToPC);
  delete[] numpPCfromCompartToPC;

  // purkinje cells
  delete2DArray<uint32_t>(pPCfromBCtoPC);
  delete2DArray<uint32_t>(pPCfromPCtoBC);
  delete2DArray<uint32_t>(pPCfromPCtoNC);
  delete[] pPCfromIOtoPC;

  // nucleus cells
  delete2DArray<uint32_t>(pNCfromPCtoNC);
  delete2DArray<uint32_t>(pNCfromNCtoIO);
  delete2DArray<uint32_t>(pNCfromMFtoNC);

  // inferior olivary cells
  delete2DArray<uint32_t>(pIOfromIOtoPC);
  delete2DArray<uint32_t>(pIOfromNCtoIO);
  delete2DArray<uint32_t>(pIOInIOIO);
  delete2DArray<uint32_t>(pIOOutIOIO);
}

void MZoneConnectivityState::stateRW(bool read, std::fstream &file) {
  // granule cells
  rawBytesRW((char *)pGRDelayMaskfromGRtoBSP, num_gr * sizeof(uint32_t), read,
             file);

  // basket cells
  rawBytesRW((char *)pBCfromBCtoPC[0],
             num_bc * num_p_bc_from_bc_to_pc * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pBCfromPCtoBC[0],
             num_bc * num_p_bc_from_pc_to_bc * sizeof(uint32_t), read, file);

  // stellate cells
  rawBytesRW((char *)pSCfromSCtoCompart[0],
             num_sc * num_p_sc_from_sc_to_compart * sizeof(uint32_t), read,
             file);
  rawBytesRW((char *)pCompartfromSCtoCompart[0],
             num_compart * num_p_compart_from_sc_to_compart * sizeof(uint32_t),
             read, file);

  rawBytesRW((char *)pCompartfromCompartToPC, num_compart * sizeof(uint32_t),
             read, file);

  rawBytesRW((char *)pPCfromCompartToPC[0],
             num_pc * max_num_p_pc_from_compart_to_pc * sizeof(uint32_t), read,
             file);
  rawBytesRW((char *)numpPCfromCompartToPC, num_pc * sizeof(uint32_t), read,
             file);

  // purkinje cells
  rawBytesRW((char *)pPCfromBCtoPC[0],
             num_pc * num_p_pc_from_bc_to_pc * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pPCfromPCtoBC[0],
             num_pc * num_p_pc_from_pc_to_bc * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pPCfromPCtoNC[0],
             num_pc * num_p_pc_from_pc_to_nc * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pPCfromIOtoPC, num_pc * sizeof(uint32_t), read, file);

  // nucleus cells
  rawBytesRW((char *)pNCfromPCtoNC[0],
             num_nc * num_p_nc_from_pc_to_nc * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pNCfromNCtoIO[0],
             num_nc * num_p_nc_from_nc_to_io * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pNCfromMFtoNC[0],
             num_nc * num_p_nc_from_mf_to_nc * sizeof(uint32_t), read, file);

  // inferior olivary cells
  rawBytesRW((char *)pIOfromIOtoPC[0],
             num_io * num_p_io_from_io_to_pc * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pIOfromNCtoIO[0],
             num_io * num_p_io_from_nc_to_io * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pIOInIOIO[0],
             num_io * num_p_io_in_io_to_io * sizeof(uint32_t), read, file);
  rawBytesRW((char *)pIOOutIOIO[0],
             num_io * num_p_io_out_io_to_io * sizeof(uint32_t), read, file);
}

void MZoneConnectivityState::assignGRDelays() {
  for (int i = 0; i < num_gr; i++) {
    // calculate x coordinate of GR position
    int grPosX = i % gr_x;

    // calculate distance of GR (assume soma) to BC, PC, and SC (aa + pf
    // distance) and assign time delay.
    int grBCPCSCDist = abs((int)(gr_x / 2 - grPosX));
    pGRDelayMaskfromGRtoBSP[i] =
        0x1 << (int)((grBCPCSCDist / gr_pf_vel_in_gr_x_per_t_step +
                      gr_af_delay_in_t_step) /
                     msPerTimeStep);
  }
}

void MZoneConnectivityState::connectBCtoPC() {
  int bcToPCRatio = num_bc / num_pc;

  for (int i = 0; i < num_pc; i++) {
    pBCfromBCtoPC[i * bcToPCRatio][0] = ((i + 1) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio][1] = ((i - 1) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio][2] = ((i + 2) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio][3] = ((i - 2) % num_pc + num_pc) % num_pc;

    pBCfromBCtoPC[i * bcToPCRatio + 1][0] =
        ((i + 1) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 1][1] =
        ((i - 1) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 1][2] =
        ((i + 3) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 1][3] =
        ((i - 3) % num_pc + num_pc) % num_pc;

    pBCfromBCtoPC[i * bcToPCRatio + 2][0] =
        ((i + 3) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 2][1] =
        ((i - 3) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 2][2] =
        ((i + 6) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 2][3] =
        ((i - 6) % num_pc + num_pc) % num_pc;

    pBCfromBCtoPC[i * bcToPCRatio + 3][0] =
        ((i + 4) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 3][1] =
        ((i - 4) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 3][2] =
        ((i + 9) % num_pc + num_pc) % num_pc;
    pBCfromBCtoPC[i * bcToPCRatio + 3][3] =
        ((i - 9) % num_pc + num_pc) % num_pc;
  }
}

void MZoneConnectivityState::connectPCtoBC() {
  int bcToPCRatio = num_bc / num_pc;

  for (int i = 0; i < num_pc; i++) {
    for (int j = 0; j < num_p_pc_from_pc_to_bc; j++) {
      int bcInd = i * bcToPCRatio - 6 + j;
      pPCfromPCtoBC[i][j] = (bcInd % num_bc + num_bc) % num_bc;
    }
  }
}

void MZoneConnectivityState::connectSCtoCompart() {
  // naive: fully connected
  for (size_t i = 0; i < num_sc; i++) {
    for (size_t j = 0; j < num_p_sc_from_sc_to_compart; j++) {
      int compartId = i / num_p_compart_from_sc_to_compart;
      pSCfromSCtoCompart[i][j] = compartId;
      pCompartfromSCtoCompart[compartId][i % num_p_compart_from_sc_to_compart] =
          i;
    }
  }
}

void MZoneConnectivityState::connectCompartToPC() {
  CRandomSFMT0 randGen(time(0));
  bool compartAssigned[num_compart];
  for (size_t i = 0; i < num_compart; i++) {
    compartAssigned[i] = false;
  }
  uint32_t total = 0;
  uint32_t pc_id = 0;
  uint32_t total_assigned = 0;
  while (total < num_compart) {

    uint32_t num_to_assign = randGen.IRandomX(min_num_p_pc_from_compart_to_pc,
                                              max_num_p_pc_from_compart_to_pc);
    if (total + num_to_assign > num_compart) {
      num_to_assign = num_compart - total;
    } else if (total + num_to_assign == num_compart) {
      num_to_assign--;
    }
    uint32_t num_assigned = 0;
    while (num_assigned < num_to_assign) {
      uint32_t compart_id = randGen.IRandomX(0, num_compart - 1);
      if (!compartAssigned[compart_id]) {
        pCompartfromCompartToPC[compart_id] = pc_id;
        pPCfromCompartToPC[pc_id][num_assigned] = compart_id;
        num_assigned++;
        total_assigned++;
        compartAssigned[compart_id] = true;
      }
    }
    numpPCfromCompartToPC[pc_id] = num_to_assign;
    total += num_to_assign;
    pc_id++;
  }
}

void MZoneConnectivityState::connectPCtoNC(int randSeed) {
  int pcNumConnected[num_pc] = {0};
  std::fill(pcNumConnected, pcNumConnected + num_pc, 1);

  CRandomSFMT0 randGen(randSeed);

  for (int i = 0; i < num_nc; i++) {
    for (int j = 0; j < num_pc / num_nc; j++) {
      int pcInd = i * (num_pc / num_nc) + j;
      pNCfromPCtoNC[i][j] = pcInd;
      pPCfromPCtoNC[pcInd][0] = i;
    }
  }

  for (int i = 0; i < num_nc - 1; i++) {
    for (int j = num_p_nc_from_pc_to_nc / 3; j < num_p_nc_from_pc_to_nc; j++) {
      int countPCNC = 0;
      int pcInd;
      while (true) {
        bool connect = true;

        pcInd = randGen.IRandomX(0, num_pc - 1);

        if (pcNumConnected[pcInd] >= num_p_pc_from_pc_to_nc)
          continue;
        for (int k = 0; k < pcNumConnected[pcInd]; k++) {
          if (pPCfromPCtoNC[pcInd][k] == i) {
            connect = false;
            break;
          }
        }
        if (connect || countPCNC > 100)
          break;
        countPCNC++;
      }

      pNCfromPCtoNC[i][j] = pcInd;
      pPCfromPCtoNC[pcInd][pcNumConnected[pcInd]] = i;

      pcNumConnected[pcInd]++;
    }
  }

  // static cast?
  unsigned int numSyn = num_pc / num_nc;

  for (int h = 1; h < num_p_pc_from_pc_to_nc; h++) {
    for (int i = 0; i < num_pc; i++) {
      if (pcNumConnected[i] < num_p_pc_from_pc_to_nc) {
        pNCfromPCtoNC[num_nc - 1][numSyn] = i;
        pPCfromPCtoNC[i][pcNumConnected[i]] = num_nc - 1;

        pcNumConnected[i]++;
        numSyn++;
      }
    }
  }
}

void MZoneConnectivityState::connectNCtoIO() {
  for (int i = 0; i < num_io; i++) {
    for (int j = 0; j < num_p_io_from_nc_to_io; j++) {
      pIOfromNCtoIO[i][j] = j;
      pNCfromNCtoIO[j][i] = i;
    }
  }
}

void MZoneConnectivityState::connectMFtoNC() {
  for (int i = 0; i < num_mf; i++) {
    for (int j = 0; j < num_p_mf_from_mf_to_nc; j++) {
      pNCfromMFtoNC[i / num_p_nc_from_mf_to_nc][i % num_p_nc_from_mf_to_nc] = i;
    }
  }
}

void MZoneConnectivityState::connectIOtoPC() {
  for (int i = 0; i < num_io; i++) {
    for (int j = 0; j < num_p_io_from_io_to_pc; j++) {
      int pcInd = i * num_p_io_from_io_to_pc + j;

      pIOfromIOtoPC[i][j] = pcInd;
      pPCfromIOtoPC[pcInd] = i;
    }
  }
}

void MZoneConnectivityState::connectIOtoIO() {
  for (int i = 0; i < num_io; i++) {
    int inInd = 0;
    for (int j = 0; j < num_p_io_in_io_to_io; j++) {
      if (inInd == i)
        inInd++;
      pIOInIOIO[i][j] = inInd;
      inInd++;
    }
  }
}
