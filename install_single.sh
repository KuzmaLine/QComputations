#!/usr/bin/sh

cd single
make -j4 -B
sudo cp libQComputations_single.so $1/.
