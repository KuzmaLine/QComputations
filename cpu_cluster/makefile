vpath %.cpp ../src
vpath %.hpp ../src
CC=mpiicpx

CFLAGS = -Werror -fopenmp -qmkl=parallel \
		 -I${MKL_INCLUDE_PATH} -I${PYTHON_H_INCLUDE_PATH} \
		 -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
		 -I${MPI_INCLUDE_PATH} \
		 -DENABLE_MPI -DENABLE_CLUSTER -DENABLE_MATPLOTLIB
LFLAGS = -qmkl=parallel -lmkl_scalapack_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 \
		 -lm -ldl \
		 -liomp5 -lpthread \
		 -L${PYTHON_LIB_PATH} \
		 -lpython3.9 \
		 -DENABLE_MPI -DENABLE_CLUSTER -DENABLE_MATPLOTLIB
OBJ = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o mpi_functions.o functions.o dynamic.o plot.o main.o
OBJ_LIB = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o mpi_functions.o functions.o dynamic.o blocked_matrix.o plot.o blocked_vector.o hamiltonian_blocked.o
is_lib = "YES"

all: libmpi_cluster

prog:
	${OBJ}
	${CC} ${LFLAGS} ${OBJ} -o $@

.cpp.o:
ifeq (${is_lib},"YES")
	${CC} ${CFLAGS} -fPIC -c $<  -o $@
else
	${CC} -g -c $< ${CFLAGS} -o $@
endif

main.o: matrix.hpp cavity_state.hpp graph.hpp big_uint.hpp state.hpp quantum_operators.hpp hamiltonian.hpp additional_operators.hpp mpi_functions.hpp functions.hpp dynamic.hpp blocked_matrix.hpp blocked_vector.hpp hamiltonian_blocked.hpp test.hpp config.hpp plot.hpp exprtk.o csr_matrix.hpp matplotlibcpp.hpp

libmpi_cluster: ${OBJ_LIB}
	${CC} -shared -o libQComputations_cpu_cluster.so ${OBJ_LIB} ${LFLAGS}


clean:
	rm -rf *.o prog libQComputations_cpu_cluster.so
