all: $1

pblas_main:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH}/../../lib/python3.9/site-packages/numpy/core/include -o pblas_main pblas_main.cpp -DDOUBLE_TYPE -lQComputations_mpi_cluster

blocked_matrix:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH}/../../lib/python3.9/site-packages/numpy/core/include -o blocked_matrix blocked_matrix.cpp -lQComputations_mpi_cluster

hamiltonian_blocked_example:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH}/../../lib/python3.9/site-packages/numpy/core/include -o hamiltonian_blocked_example hamiltonian_blocked_example.cpp -lQComputations_mpi_cluster

hamiltonian_single_example:
	 icpc -I${MKL_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH}/../../lib/python3.9/site-packages/numpy/core/include -o hamiltonian_single_example hamiltonian_single_example.cpp -lQComputations_single

clean:
	rm -rf pblas_main blocked_matrix hamiltonian_single_example hamiltonian_blocked_example
