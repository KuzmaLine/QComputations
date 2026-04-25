#!/usr/bin/bash

cd cpu_cluster
rm -rf `ls | grep -v "CMakeLists.txt"`
cd ../single
rm -rf `ls | grep -v "CMakeLists.txt"`
cd ../cuda
rm -rf `ls | grep -v "CMakeLists.txt"`
cd ../blocked_cuda
rm -rf `ls | grep -v "CMakeLists.txt"`
cd ..
