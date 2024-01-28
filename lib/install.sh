#!/usr/bin/bash

./install_headers.sh /home/s02200417_2309/lib_result
cd cpu_cluster
cmake -DCMAKE_CXX_COMPILER=mpiicpc -B .
make -j4 -B
mv ./*.so /home/s02200417_2309/lib_result


#cd ../single
#cmake -DCMAKE_CXX_COMPILER=icpx -B .
#make -j4 -B
#sudo mv ./*.so $2/.
