#!/usr/bin/sh

cd mpi
make -B
sudo cp libQComputations_mpi.so $1/.
