#!/usr/bin/bash

./install_headers.sh $1
cd cpu_cluster
cmake -DCMAKE_CXX_COMPILER=mpiicpx -B .
make -j4 -B
sudo mv ./*.so $2/.

cd ../single
cmake -DCMAKE_CXX_COMPILER=icpx -B .
make -j4 -B
sudo mv ./*.so $2/.
