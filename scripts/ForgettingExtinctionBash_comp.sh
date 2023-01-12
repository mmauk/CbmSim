#!/usr/bin/bash

cd ../build

cp ../data/inputs/bunny_1_trained.sim ../data/inputs/bunny_1_readytoextinguish.sim

    #Extinguishing
echo "STARTING WITH EXTINGUISHING"

./cbm_sim -i bunny_1_readytoextinguish.sim -s GS_test_extinction.sess --raster PC,bunny1_ISI1000_PC_raster_extinction.bin \
    GR,bunny1_ISI1000_GR_raster_extinction.bin --psth MF,bunny1_ISI1000_MF_raster_extinction.bin --weights PFPC,bunny1_ISI1000_weights_PFPC_extinction.bin  -o bunny_1_howyoudoing.sim

echo "DONE"

