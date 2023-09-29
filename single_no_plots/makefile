vpath %.cpp ../src
vpath %.hpp ../src
CC=g++

CFLAGS = -Werror -fopenmp \
		 -I${MKL_INCLUDE_PATH}
LFLAGS = -Werror \
		 -lmkl_intel_lp64 \
		 -lmkl_intel_thread -lmkl_core \
		 -lm -ldl \
		 -liomp5 -lpthread
OBJ = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o functions.o dynamic.o main.o
OBJ_LIB = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o functions.o dynamic.o
is_lib = "YES"

all: libQC

prog: ${OBJ}
	${CC} ${OBJ} ${LFLAGS} -o $@

.cpp.o:
ifeq (${is_lib},"YES")
	${CC} -g -c -fPIC $< ${CFLAGS} -o $@
else
	${CC} -g -c $< ${CFLAGS} -o $@
endif

main.o: matrix.hpp cavity_state.hpp graph.hpp big_uint.hpp state.hpp quantum_operators.hpp hamiltonian.hpp additional_operators.hpp functions.hpp dynamic.hpp test.hpp config.hpp csr_matrix.hpp matplotlibcpp.hpp

libQC: ${OBJ_LIB}
	${CC} -shared -o libQComputations_single_no_plots.so ${OBJ_LIB} ${LFLAGS}


clean:
	rm -rf *.o prog libQComputations_single_no_plots.so
