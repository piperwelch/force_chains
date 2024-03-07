#!/bin/bash
# Specify a partition
#SBATCH --partition=bluemoon
# Request nodes 
#SBATCH --nodes=1
# Request some processor cores
#SBATCH --ntasks=1
# Maximum runtime
#SBATCH --time=0:15:00
# Name of job
#SBATCH --job-name=L_BATCH
# Output of this job, stderr and stdout are joined by default
# %x=job-name %j=jobid
#SBATCH --output=outputs/%x_%j.out

data_files=("${@:1:10}")
visualize=${11}
output_node=${12}
outputs=${13}

for data_file in "${data_files[@]}"; do
    id=$(echo "$data_file" | grep -oP 'data_id\K\d+')
    seed=$(echo "$data_file" | grep -oP 'seed\K\d+')    

    ./lmp_serial -in in.force_chain -var data_file ${data_file} \
    -var output_node ${output_node} -var id ${id} -var visualize ${visualize} \
    -var seed ${seed} -var dump_folder ../dumps/ >../outputs/${outputs}

done