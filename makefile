default:
	${MAKE} -C ./mpi
	${MAKE} -C ./single

clean:
	${MAKE} clean -C ./mpi
	${MAKE} clean -C ./single
