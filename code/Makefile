CXX = nvcc

init:
	module load nvidia-hpc-sdk

std: standard.cu
	$(CXX) $< -o $@
	sbatch job.sl

lienar: linear.cu
	$(CXX) $< -o $@
	sbatch job.sl

clean:
	rm -f *~ *.o linear mmult standard