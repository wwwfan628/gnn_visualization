#!/bin/bash

#SBATCH --mail-type=ALL                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=log/%j.out                 # where to store the output ( %j is the JOBID )
#SBATCH --error=log/%j.err                  # where to store error messages
#SBATCH --gres=gpu:1                        # use 1 GPU for the job

/bin/echo Running on host: `hostname`
/bin/echo In directory: `pwd`
/bin/echo Starting on: `date`
/bin/echo SLURM_JOB_ID: $SLURM_JOB_ID

# exit on errors
set -o errexit

# binary to execute
export PYTHONPATH=/home/yiflu/Desktop/gnn_visualization
export CUDA_VISIBLE_DEVICES=0
cd /home/yiflu/Desktop/gnn_visualization/src/run
python ./fixpoint.py cora graph_optimization --train

echo finished at: `date`
exit 0;