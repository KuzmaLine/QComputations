vpath %.cpp ./src
vpath %.hpp ./src
CC=g++

LFLAGS = -std=c++11 -Werror -I/usr/include/python3.10 -I/usr/include/mkl -lpython3.10 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
CFLAGS = -std=c++11 -Werror -I/usr/include/python3.10 -I/usr/include/mkl -lpython3.10 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl
OBJ = matrix.o cavity_state.o graph.o state.o hamiltonian.o additional_operators.o functions.o dynamic.o plot.o main.o

all: prog

prog: $(OBJ)
	${CC} ${OBJ} ${LFLAGS} -o $@

.cpp.o:
	${CC} -g ${CFLAGS} -c $< -o $@

main.o: matrix.hpp cavity_state.hpp graph.hpp state.hpp hamiltonian.hpp additional_operators.hpp functions.hpp dynamic.hpp plot.hpp test.hpp config.hpp matplotlibcpp.hpp

clean:
	rm -rf *.o prog
