#!/usr/bin/bash

./install_headers.sh $1
sudo cp src/QComputations/seaborn_config.json $1/QComputations/seaborn_config.json
sudo cp src/QComputations/seaborn_plot.py $1/QComputations/seaborn_plot.py
