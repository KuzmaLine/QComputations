#!/usr/bin/sh

make -B
sudo cp mpi_cluster/libQComputations_mpi_cluster.so $1/.
sudo cp mpi/libQComputations_mpi.so $1/.
sudo cp single/libQComputations_single.so $1/.
