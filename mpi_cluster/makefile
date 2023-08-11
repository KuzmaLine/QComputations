vpath %.cpp ../src
vpath %.hpp ../src
CC=mpiicpc

CFLAGS = -Werror -fopenmp -diag-disable=10441 \
		 -I/opt/intel/oneapi/mkl -DENABLE_MPI -DENABLE_CLUSTER
LFLAGS = -Werror \
		 -lmkl_intel_lp64 \
		 -lmkl_scalapack_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 \
		 -lm -ldl \
		 -liomp5 -lpthread -DENABLE_MPI -DENABLE_CLUSTER
OBJ = matrix.o cavity_state.o graph.o state.o quantum_operators.o hamiltonian.o additional_operators.o mpi_functions.o functions.o dynamic.o plot.o main.o

all: prog

prog: $(OBJ)
	${CC} ${OBJ} ${LFLAGS} -o $@

.cpp.o:
	${CC} -g -c $< ${CFLAGS} -o $@

main.o: matrix.hpp cavity_state.hpp graph.hpp state.hpp quantum_operators.hpp hamiltonian.hpp additional_operators.hpp mpi_functions.hpp functions.hpp dynamic.hpp plot.hpp test.hpp config.hpp matplotlibcpp.hpp

clean:
	rm -rf *.o prog
