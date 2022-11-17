#!/usr/bin/bash

cd ../build

#Making the bunny
./cbm_sim -b build_file_tune_09262022.bld -o bunny_2.sim

#Running SIM
    #Acquisition
echo "STARTING WITH ACQUISITION"
for k in {1..5}
do
    ./cbm_sim -i bunny_2.sim -s GOGO_NoPlast_Acq.sess -p --raster GO,ISI2000_TCS_Test_GORaster_${k}.bin -o bunny_out_${k}.sim
done
echo "DONE WITH ACQUISITION"


echo "DONE"

