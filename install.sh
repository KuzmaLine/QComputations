#!/usr/bin/bash

./clean_install.sh

./install_headers.sh $1

./update_plot_script.sh $1

# Read the .bashrc file into a variable
bashrc_content=$(cat ~/.bashrc)

# Check if the alias already exists in .bashrc
if echo "$bashrc_content" | grep -q "export SEABORN_PLOT"; then
    # The alias exists, so remove it and add the new one
    echo "$bashrc_content" | sed -i "s@export SEABORN_PLOT.*@export SEABORN_PLOT=\"$1\/QComputations\/seaborn_plot.py\"@" ~/.bashrc
else
    # The alias doesn't exist, so add it to .bashrc
    echo -e "\nexport SEABORN_PLOT=\"$1/QComputations/seaborn_plot.py\"" >> ~/.bashrc
fi

# Check if the alias already exists in .bashrc
if echo "$bashrc_content" | grep -q "export SEABORN_CONFIG"; then
    # The alias exists, so remove it and add the new one
    echo "$bashrc_content" | sed -i "s@export SEABORN_CONFIG.*@export SEABORN_CONFIG=\"$1\/QComputations\/seaborn_config.json\"@" ~/.bashrc
else
    # The alias doesn't exist, so add it to .bashrc
    echo -e "\nexport SEABORN_CONFIG=\"$1/QComputations/seaborn_config.json\"" >> ~/.bashrc
fi

sudo chmod +x $1/QComputations/seaborn_plot.py

cd cpu_cluster
cmake -DCMAKE_CXX_COMPILER=mpiicpx -B .
make -j4 -B
sudo mv ./*.so $2/.

cd ../single
cmake -DCMAKE_CXX_COMPILER=icpx -B .
make -j4 -B
sudo mv ./*.so $2/.

sudo apt install -y clang-format-15
