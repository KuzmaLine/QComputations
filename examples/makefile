all: $1

MPICC = mpiicpx
CC = icpx

pblas_main:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o pblas_main pblas_main.cpp -DDOUBLE_TYPE -lQComputations_cpu_cluster_no_plots

blocked_matrix:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o blocked_matrix blocked_matrix.cpp -lQComputations_cpu_cluster_no_plots

hamiltonian_blocked_example:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o hamiltonian_blocked_example hamiltonian_blocked_example.cpp -lQComputations_cpu_cluster_no_plots

hamiltonian_single_example:
	 ${CC} -I${MKL_INCLUDE_PATH} -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
	-I${PYTHON_H_INCLUDE_PATH} -o hamiltonian_single_example hamiltonian_single_example.cpp -lQComputations_single

schrodinger:
	 ${MPICC} -o schrodinger schrodinger.cpp -ldl -lQComputations_CPU_CLUSTER

exprtk_test:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
	-I${PYTHON_H_INCLUDE_PATH} -pedantic-errors -Wextra -Wno-long-long -o exprtk_test exprtk_test.cpp -lstdc++ -lm

mpi_qcomp:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o mpi_qcomp mpi_qcomp.cpp -lQComputations_cpu_cluster_no_plots

my_file:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o my_file my_file.cpp -lQComputations_cpu_cluster_no_plots

runge_kutt:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o runge_kutt runge_kutt.cpp -lQComputations_cpu_cluster_no_plots

rho_dynamic:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o rho_dynamic rho_dynamic.cpp -lQComputations_cpu_cluster_no_plots

stocks:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -o stocks stocks.cpp -lQComputations_cpu_cluster_no_plots

taylor_dynamics:
	 ${MPICC} -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
	-I${PYTHON_H_INCLUDE_PATH} -o taylor_dynamics taylor_dynamics.cpp -lQComputations_cpu_cluster_no_plots

clean:
	rm -rf taylor_dynamics exprtk_test pblas_main schrodinger blocked_matrix hamiltonian_single_example stocks hamiltonian_blocked_example mpi_qcomp runge_kutt rho_dynamic my_file a.out
