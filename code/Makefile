CXX = nvcc

standard: standard.cu
	$(CXX) $< -o $@
	sbatch job.sl

linear: linear.cu
	$(CXX) $< -o $@
	sbatch job.sl

init:
	module load nvidia-hpc-sdk

clean:
	rm -f *~ *.o linear mmult standard gpujob.out


# alias res='watch -n 1 ls'
# squeue -u johnheo