vpath %.cpp ../src
vpath %.hpp ../src
CC=mpicc

CFLAGS = -Werror -fopenmp -m64\
		 -I${MKL_INCLUDE_PATH} -I${MPI_INCLUDE_PATH} \
		 -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
		 -I${PYTHON_H_INCLUDE_PATH} \
		 -DENABLE_MPI -DENABLE_MATPLOTLIB
LFLAGS = -Werror \
		 -lmkl_intel_lp64 \
		 -lmkl_gnu_thread -lmkl_core \
		 -lm -ldl \
		 -lgomp -lpthread \
		 -L${PYTHON_LIB_PATH} \
		 -lpython3.9 \
		 -DENABLE_MPI -DENABLE_MATPLOTLIB
OBJ = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o mpi_functions.o functions.o dynamic.o plot.o main.o
OBJ_LIB = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o mpi_functions.o functions.o dynamic.o plot.o
is_lib = "YES"

all: libmpi

prog:
	${OBJ}
	${CC} ${OBJ} ${LFLAGS} -o $@

.cpp.o:
ifeq (${is_lib},"YES")
	${CC} -g -c -fPIC $< ${CFLAGS} -o $@
else
	${CC} -g -c $< ${CFLAGS} -o $@
endif

main.o: matrix.hpp cavity_state.hpp graph.hpp big_uint.hpp state.hpp quantum_operators.hpp hamiltonian.hpp additional_operators.hpp mpi_functions.hpp functions.hpp dynamic.hpp plot.hpp test.hpp config.hpp csr_matrix.hpp matplotlibcpp.hpp

libmpi: ${OBJ_LIB}
	${CC} -shared -o libQComputations_mpi.so ${OBJ_LIB} ${LFLAGS}


clean:
	rm -rf *.o prog libQComputations_mpi.so
