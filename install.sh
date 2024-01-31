#!/usr/bin/bash

./install_headers.sh

# Read the .bashrc file into a variable
bashrc_content=$(cat ~/.bashrc)

# Check if the alias already exists in .bashrc
if echo "$bashrc_content" | grep -q "export SEABORN_PLOT"; then
    # The alias exists, so remove it and add the new one
    echo "$bashrc_content" | sed -i "s@export SEABORN_PLOT.*@export SEABORN_PLOT=\"$1\/mnt/scratch/users/$USER/lib_result/QComputations\/seaborn_plot.py\"@" ~/.bashrc
else
    # The alias doesn't exist, so add it to .bashrc
    echo -e "\nexport SEABORN_PLOT=\"$1/mnt/scratch/users/$USER/lib_result/QComputations/seaborn_plot.py\"" >> ~/.bashrc
fi

chmod +x $1/QComputations/seaborn_plot.py

cd cpu_cluster
cmake -DCMAKE_CXX_COMPILER=mpiicpc -B .
make -j4 -B
mv ./*.so ~/lib_result/.

#cd ../single
#cmake -DCMAKE_CXX_COMPILER=icpx -B .
#make -j4 -B
#sudo mv ./*.so $2/.

