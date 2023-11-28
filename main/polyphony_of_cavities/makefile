MPICC = mpiicpx

polyphony_of_cavities:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
	-I${PYTHON_H_INCLUDE_PATH} -o polyphony_of_cavities polyphony_of_cavities.cpp -lQComputations_cpu_cluster

periods:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
	-I${PYTHON_H_INCLUDE_PATH} -o periods periods.cpp -lQComputations_cpu_cluster

clean:
	rm -rf polyphony_of_cavities periods
