#!/usr/bin/sh

cd single
make -B
sudo cp libQComputations_single.so $1/.
