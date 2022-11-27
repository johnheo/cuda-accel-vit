CXX = nvcc

init:
	module load nvidia-hpc-sdk

p1: linear.cu
	$(CXX) $< -o $@
	sbatch job.sl

clean:
	rm -f *~ *.o linear mmult standard