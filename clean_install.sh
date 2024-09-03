#!/usr/bin/bash

cd cpu_cluster
rm -rf `ls | grep -v "CMakeLists.txt"`
cd ../single
rm -rf `ls | grep -v "CMakeLists.txt"`
cd ..
