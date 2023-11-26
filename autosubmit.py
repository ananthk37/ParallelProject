import os

inputSize = [pow(2, 16), pow(2, 20), pow(2, 24)]
processCounts = [2,4,8,16,32,64,128]
cudaThreadCounts = [64,128,256,512,1024]
inputTypes = ["sorted", "reverse", "nearly", "random"]
size_per_thread = [pow(2, 6), pow(2, 10), pow(2, 14)]
size_per_proc = [pow(2, 9), pow(2, 13), pow(2, 17)]

def run_sequential():
  for inputType in inputTypes:
    for size in inputSize:
      #note change file name to match your job
      os.system("sbatch jobfile.sh {} {}".format(inputType, size))

def run_MPI():
  for inputType in inputTypes:
    for size in inputSize:
      for procs in processCounts:
        #note change file name to match your job
        os.system("sbatch mpi.grace_job {} {} {}".format(inputType, procs, size))

def run_CUDA():
  for inputType in inputTypes:
    for size in inputSize:
      for threads in cudaThreadCounts:
        #note change file name to match your job
        os.system("sbatch cuda.grace_job {} {} {}".format(inputType, threads, size))
        
def run_weak_MPI():
  for procs in processCounts:
      for size in size_per_proc:
          os.system("sbatch mpi.grace_job random {} {}".format(procs, size*procs))
            
def run_weak_CUDA():
  for threads in cudaThreadCounts:
      for size in size_per_thread:
          os.system("sbatch cuda.grace_job random {} {}".format(threads, size*threads))
            
def run_missing_MPI():
  # missing weak
  for type in ["sorted", "reverse", "nearly"]:
    for procs in processCounts:
      for size in size_per_proc:
        os.system("sbatch mpi.grace_job {} {} {}".format(type, procs, size*procs))
        
  # missing strong
  for type in ["sorted", "reverse", "nearly"]:
    for procs in processCounts:
      for size in [pow(2, 16), pow(2, 24)]:
        os.system("sbatch mpi.grace_job {} {} {}".format(type, procs, size))
            
def run_missing_CUDA():
  # missing weak
  for type in ["sorted", "reverse", "nearly"]:
    for threads in cudaThreadCounts:
      for size in size_per_thread:
        os.system("sbatch cuda.grace_job {} {} {}".format(type, threads, size*threads))
        
  # missing strong
  for type in ["sorted", "reverse", "nearly"]:
    for threads in cudaThreadCounts:
      for size in [pow(2, 16), pow(2, 24)]:
        os.system("sbatch cuda.grace_job {} {} {}".format(type, threads, size))

# run_sequential()
# run_MPI()
# run_CUDA()

# run_weak_MPI()
# run_weak_CUDA()

# run_missing_MPI()
# run_missing_CUDA()
