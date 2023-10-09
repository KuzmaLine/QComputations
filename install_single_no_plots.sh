#!/usr/bin/sh

cd single_no_plots
make -j4 -B
sudo cp libQComputations_single_no_plots.so $1/.
