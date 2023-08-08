default:
	${MAKE} -C ./mpi
	${MAKE} -C ./single
	${MAKE} -C ./mpi_cluster

clean:
	${MAKE} clean -C ./mpi
	${MAKE} clean -C ./single
	${MAKE} clean -C ./mpi_cluster
