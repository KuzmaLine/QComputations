vpath %.cpp ./src
vpath %.hpp ./src
CC=g++

LFLAGS = -std=c++11 -Werror -I/usr/include/python3.10 -lpython3.10
CFLAGS = -std=c++11 -Werror -I/usr/include/python3.10 -lpython3.10
OBJ = matrix.o state.o graph.o hamiltonian.o additional_operators.o functions.o dynamic.o main.o

all: prog

prog: $(OBJ)
	${CC} ${OBJ} ${LFLAGS} -o $@

.cpp.o:
	${CC} -g ${CFLAGS} -c $< -o $@

main.o: matrix.hpp state.hpp test.hpp graph.hpp hamiltonian.hpp additional_operators.hpp functions.hpp dynamic.hpp config.hpp

clean:
	rm -rf *.o prog
