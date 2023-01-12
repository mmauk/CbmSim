#!/usr/bin/bash

cd ../build

#Making the bunny
./cbm_sim -b build_file_tune_09262022.bld -o b1_NullZoneTest1.sim

#Running SIM
echo "STARTING WITH CONTROL GROUP"
    #Acquisition
./cbm_sim -i b1_NullZoneTest1.sim -s Acquisition_NullZoneTest1_0.sess --raster PC,NullZoneTest1_PC_raster_Acq_Control.bin \
    GR,NullZoneTest1_GR_raster_Acq_Control.bin --psth MF,NullZoneTest1_MF_PSTH_Acq_Control.bin --weights PFPC,NullZoneTest1_weights_PFPC_Acq_Control.bin -o b1_NullZoneTest1_control_trained.sim

cp ../data/inputs/b1_NullZoneTest1_control_trained.sim ../data/inputs/b1_NullZoneTest1_control_readytoextinguish.sim
    #Extinguishing
./cbm_sim -i b1_NullZoneTest1_control_readytoextinguish.sim -s Extinction_NullZoneTest1_0.sess --raster PC,NullZoneTest1_PC_raster_Ext_Control.bin \
    GR,NullZoneTest1_GR_raster_Ext_Control.bin --psth MF,NullZoneTest1_MF_PSTH_Ext_Control.bin --weights PFPC,NullZoneTest1_weights_PFPC_Ext_Control.bin  -o b1_NullZoneTest1_control_extinguished.sim

echo "DONE WITH CONTROL GROUP"
mkdir -p /home/data/NullZoneTest1/ControlGrp/
mv -v ../data/outputs/* /home/data/NullZoneTest1/ControlGrp/

echo "STARTING WITH EXP GROUP 1"
    #Acquisition
./cbm_sim -i b1_NullZoneTest1.sim -s Acquisition_NullZoneTest1_1.sess --raster PC,NullZoneTest1_PC_raster_Acq_Exp1.bin \
    GR,NullZoneTest1_GR_raster_Acq_Exp1.bin --psth MF,NullZoneTest1_MF_PSTH_Acq_Exp1.bin --weights PFPC,NullZoneTest1_weights_PFPC_Acq_Exp1.bin -o b1_NullZoneTest1_Exp1_trained.sim

cp ../data/inputs/b1_NullZoneTest1_Exp1_trained.sim ../data/inputs/b1_NullZoneTest1_Exp1_readytoextinguish.sim
    #Extinguishing
./cbm_sim -i b1_NullZoneTest1_Exp1_readytoextinguish.sim -s Extinction_NullZoneTest1_1.sess --raster PC,NullZoneTest1_PC_raster_Ext_Exp1.bin \
    GR,NullZoneTest1_GR_raster_Ext_Exp1.bin --psth MF,NullZoneTest1_MF_PSTH_Ext_Exp1.bin --weights PFPC,NullZoneTest1_weights_PFPC_Ext_Exp1.bin  -o b1_NullZoneTest1_Exp1_extinguished.sim

echo "DONE WITH EXP GROUP 1"

mkdir -p /home/data/NullZoneTest1/EqualDelta_HigherStep/
mv -v ../data/outputs/* /home/data/NullZoneTest1/EqualDelta_HigherStep/

echo "STARTING WITH EXP GROUP 2"
    #Acquisition
./cbm_sim -i b1_NullZoneTest1.sim -s Acquisition_NullZoneTest1_2.sess --raster PC,NullZoneTest1_PC_raster_Acq_Exp2.bin \
    GR,NullZoneTest1_GR_raster_Acq_Exp2.bin --psth MF,NullZoneTest1_MF_PSTH_Acq_Exp2.bin --weights PFPC,NullZoneTest1_weights_PFPC_Acq_Exp2.bin -o b1_NullZoneTest1_Exp2_trained.sim

cp ../data/inputs/b1_NullZoneTest1_Exp2_trained.sim ../data/inputs/b1_NullZoneTest1_Exp2_readytoextinguish.sim
    #Extinguishing
./cbm_sim -i b1_NullZoneTest1_Exp2_readytoextinguish.sim -s Extinction_NullZoneTest1_2.sess --raster PC,NullZoneTest1_PC_raster_Ext_Exp2.bin \
    GR,NullZoneTest1_GR_raster_Ext_Exp2.bin --psth MF,NullZoneTest1_MF_PSTH_Ext_Exp2.bin --weights PFPC,NullZoneTest1_weights_PFPC_Ext_Exp2.bin  -o b1_NullZoneTest1_Exp2_extinguished.sim

echo "DONE WITH EXP GROUP 2"
mkdir -p /home/data/NullZoneTest1/EqualDelta_LowerStep/
mv -v ../data/outputs/* /home/data/NullZoneTest1/EqualDelta_LowerStep/
