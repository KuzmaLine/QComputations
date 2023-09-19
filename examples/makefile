all: $1

pblas_main:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH}/../../lib/python3.9/site-packages/numpy/core/include -o pblas_main pblas_main.cpp -DDOUBLE_TYPE -lQComputations_mpi_cluster

clean:
	rm -rf pblas_main
