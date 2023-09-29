default:
	${MAKE} -C ./single
	${MAKE} -C ./single_no_plots
	${MAKE} -C ./cpu_cluster
	${MAKE} -C ./cpu_cluster_no_plots

clean:
	${MAKE} clean -C ./single
	${MAKE} clean -C ./single_no_plots
	${MAKE} clean -C ./cpu_cluster
	${MAKE} clean -C ./cpu_cluster_no_plots
