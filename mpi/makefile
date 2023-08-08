vpath %.cpp ../src
vpath %.hpp ../src
CC=mpiCC

CFLAGS = -std=c++11 -m64 -Werror -fopenmp \
		 -I/usr/include/python3.10 \
		 -I/usr/include/mkl -DENABLE_MPI
LFLAGS = -std=c++11 -Werror -fopenmp -lpython3.10 -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi \
		 -L/usr/lib/x86_64-linux-gnu \
		 -lmkl_intel_lp64 \
		 -lmkl_scalapack_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_openmpi_lp64 \
		 -lm -ldl \
		 -liomp5 -lpthread -DENABLE_MPI
OBJ = matrix.o cavity_state.o graph.o state.o quantum_operators.o hamiltonian.o additional_operators.o mpi_functions.o functions.o dynamic.o plot.o main.o

all: prog

prog: $(OBJ)
	${CC} ${OBJ} ${LFLAGS} -o $@

.cpp.o:
	${CC} -g -c $< ${CFLAGS} -o $@

main.o: matrix.hpp cavity_state.hpp graph.hpp state.hpp quantum_operators.hpp hamiltonian.hpp additional_operators.hpp mpi_functions.hpp functions.hpp dynamic.hpp plot.hpp test.hpp config.hpp matplotlibcpp.hpp

clean:
	rm -rf *.o prog
