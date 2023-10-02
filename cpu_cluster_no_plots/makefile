vpath %.cpp ../src
vpath %.hpp ../src
CC=mpiicpc

CFLAGS = -Werror -fopenmp -diag-disable=10441\
		 -I${MKL_INCLUDE_PATH} \
		 -I${MPI_INCLUDE_PATH} \
		 -DENABLE_MPI -DENABLE_CLUSTER
LFLAGS = -Werror \
		 -lmkl_intel_lp64 \
		 -lmkl_scalapack_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 \
		 -lm -ldl \
		 -liomp5 -lpthread \
		 -DENABLE_MPI -DENABLE_CLUSTER
OBJ = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o mpi_functions.o functions.o dynamic.o plot.o main.o
OBJ_LIB = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o mpi_functions.o functions.o dynamic.o blocked_matrix.o hamiltonian_blocked.o
is_lib = "YES"

all: libmpi_cluster

prog:
	${OBJ}
	${CC} ${OBJ} ${LFLAGS} -o $@

.cpp.o:
ifeq (${is_lib},"YES")
	${CC} -g -c -fPIC $< ${CFLAGS} -o $@
else
	${CC} -g -c $< ${CFLAGS} -o $@
endif

main.o: matrix.hpp cavity_state.hpp graph.hpp big_uint.hpp state.hpp quantum_operators.hpp hamiltonian.hpp additional_operators.hpp mpi_functions.hpp functions.hpp dynamic.hpp blocked_matrix.hpp hamiltonian_blocked.hpp test.hpp config.hpp csr_matrix.hpp matplotlibcpp.hpp

libmpi_cluster: ${OBJ_LIB}
	${CC} -shared -o libQComputations_cpu_cluster_no_plots.so ${OBJ_LIB} ${LFLAGS}


clean:
	rm -rf *.o prog libQComputations_cpu_cluster_no_plots.so
