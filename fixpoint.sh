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
cd ./src/run
read -p "Choose dataset from cora, reddit-self-loop, ppi and tu: " dataset
read -p "Choose the way to find fixpoint from graph_optimization, node_graph_optimization and newton_method: " method
read -p "Training needed, i.e. checkpoint of network not available? True or False: " train_needed
if [ $train_needed==True ]
then
  python ./fixpoint.py $dataset $method --train
else
  python ./fixpoint.py $dataset $method --train
fi

echo finished at: `date`
exit 0;