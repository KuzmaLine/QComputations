#!/usr/bin/sh

cd cpu_cluster_no_plots
make -j4 -B
sudo cp libQComputations_cpu_cluster_no_plots.so $1/.
