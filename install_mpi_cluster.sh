#!/usr/bin/sh

cd mpi_cluster
make -B
sudo cp libQComputations_mpi_cluster.so $1/.
