#!/usr/bin/sh

cd cpu_cluster
make -B
sudo cp libQComputations_cpu_cluster.so $1/.
