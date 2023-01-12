#!/usr/bin/bash

cd ../build

#Making the bunny
./cbm_sim -b build_file_tune_09262022.bld -o bunny_1.sim

#Running SIM
    #Acquisition
echo "STARTING WITH ACQUISITION"
./cbm_sim -i bunny_1.sim -s GS_test_acquisition.sess --raster PC,bunny1_ISI500_PC_raster_Acq.bin \
    MF,bunny1_MF_raster_Acq.bin --weights PFPC,bunny1_ISI500_weights_PFPC_acq.bin -o bunny_1_trained.sim

echo "DONE WITH ACQUISITION"
cp ../data/inputs/bunny_1_trained.sim ../data/inputs/bunny_1_readytoforget.sim
cp ../data/inputs/bunny_1_trained.sim ../data/inputs/bunny_1_readytoextinguish.sim

    #Forgetting 
echo "STARTING WITH FORGETTING"
for k in {1..50}
do
    ./cbm_sim -i bunny_1_readytoforget.sim -s GS_test_forget100t.sess -o bunny_1_howyoudoing.sim
    rm ../data/inputs/bunny_1_readytoforget.sim

    ./cbm_sim -i bunny_1_howyoudoing.sim -s GS_test_probe.sess -p --raster PC,bunny1_ISI500_pc_raster_probe_${k}_forget.bin \
        MF,bunny1_ISI500_mf_raster_probe_${k}_forget.bin --weights PFPC,bunny1_ISI500_weights_PFPC_probe_${k}_forget.bin -o bunny_1_readytoforget.sim
    rm ../data/inputs/bunny_1_howyoudoing.sim
    echo "Done with iteration $k of forgetting"
done
echo "DONE WITH FORGETTING"

    #Extinguishing
echo "STARTING WITH EXTINGUISHING"

./cbm_sim -i bunny_1_readytoextinguish.sim -s GS_test_extinction.sess -o bunny_1_howyoudoing.sim
rm ../data/inputs/bunny_1_readytoextinguish.sim

echo "DONE"

