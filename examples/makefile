all: $1

pblas_main:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o pblas_main pblas_main.cpp -DDOUBLE_TYPE -lQComputations_cpu_cluster_no_plots

blocked_matrix:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o blocked_matrix blocked_matrix.cpp -lQComputations_cpu_cluster_no_plots

hamiltonian_blocked_example:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o hamiltonian_blocked_example hamiltonian_blocked_example.cpp -lQComputations_cpu_cluster_no_plots

hamiltonian_single_example:
	 icpc -I${MKL_INCLUDE_PATH} -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
	-I${PYTHON_H_INCLUDE_PATH} -o hamiltonian_single_example hamiltonian_single_example.cpp -lQComputations_single

schrodinger:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
	-I${PYTHON_H_INCLUDE_PATH} -o schrodinger schrodinger.cpp -lQComputations_cpu_cluster

mpi_qcomp:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o mpi_qcomp mpi_qcomp.cpp -lQComputations_cpu_cluster_no_plots

my_file:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o my_file my_file.cpp -lQComputations_cpu_cluster_no_plots

runge_kutt:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o runge_kutt runge_kutt.cpp -lQComputations_cpu_cluster_no_plots

rho_dynamic:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o rho_dynamic rho_dynamic.cpp -lQComputations_cpu_cluster_no_plots

stocks:
	 mpiicpc -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o stocks stocks.cpp -lQComputations_cpu_cluster_no_plots

clean:
	rm -rf pblas_main schrodinger blocked_matrix hamiltonian_single_example stocks hamiltonian_blocked_example mpi_qcomp runge_kutt rho_dynamic my_file
