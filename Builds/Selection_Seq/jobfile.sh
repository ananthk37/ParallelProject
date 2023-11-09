#!/bin/bash
### Job commands start here
## echo '=====================JOB STARTING=========================='
#SBATCH --job-name=sequentialselection              ### Job Name
#SBATCH --output=output.%j        ### File in which to store job output
#SBATCH --time=0-00:02:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1               ### Node count required for the job, default = 1
#SBATCH --mem=8G                ### memory per node
#SBATCH --exclusive             ### no shared resources within a node
#SBATCH --partition=short       ### Which hardware partition to run on
time_now=$(date +%s)

# Set variables from input
input_type=$1
arr_size=$2

CALI_CONFIG="spot(output=SEQ-${input_type}-v${arr_size}.cali)" ./selection_seq $input_type $arr_size