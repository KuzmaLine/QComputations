#!/usr/bin/sh

cd cpu_cluster
make -j4 -B
sudo cp libQComputations_cpu_cluster.so $1/.
