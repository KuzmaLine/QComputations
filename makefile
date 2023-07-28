default:
	${MAKE} -C ./single
	${MAKE} -C ./mpi

clean:
	${MAKE} clean -C ./single
	${MAKE} clean -C ./mpi
