vpath %.cpp ../src
vpath %.hpp ../src
CC=g++

CFLAGS = -Werror -fopenmp \
		 -I/opt/intel/oneapi/mkl -I/opt/intel/oneapi/intelpython/python3.9/include/python3.9 \
		 -I/opt/intel/oneapi/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include \
		 -DENABLE_MATPLOTLIB
LFLAGS = -Werror \
		 -lmkl_intel_lp64 \
		 -lmkl_intel_thread -lmkl_core \
		 -lm -ldl \
		 -liomp5 -lpthread \
		 -L/opt/intel/oneapi/intelpython/python3.9/lib \
		 -lpython3.9 \
		 -DENABLE_MATPLOTLIB
OBJ = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o functions.o dynamic.o plot.o main.o
OBJ_LIB = matrix.o cavity_state.o graph.o big_uint.o state.o quantum_operators.o hamiltonian.o additional_operators.o functions.o dynamic.o plot.o
is_lib = "YES"

all: libQC

prog:
	${OBJ}
	${CC} ${OBJ} ${LFLAGS} -o $@

.cpp.o:
ifeq (${is_lib},"YES")
	${CC} -g -c -fPIC $< ${CFLAGS} -o $@
else
	${CC} -g -c $< ${CFLAGS} -o $@
endif

main.o: matrix.hpp cavity_state.hpp graph.hpp big_uint.hpp state.hpp quantum_operators.hpp hamiltonian.hpp additional_operators.hpp functions.hpp dynamic.hpp plot.hpp test.hpp config.hpp csr_matrix.hpp matplotlibcpp.hpp

libQC: ${OBJ_LIB}
	${CC} -shared -o libQComputations_single.so ${OBJ_LIB} ${LFLAGS}


clean:
	rm -rf *.o prog libQComputations_single.so
