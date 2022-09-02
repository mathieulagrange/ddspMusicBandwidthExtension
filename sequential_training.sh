#!/bin/bash

JID_JOB1=`sbatch ./slurm/$1.slurm | cut -d " " -f 4`
JID_JOB2=`sbatch --dependency=afterok:$JID_JOB1 ./slurm/$1.slurm | cut -d " " -f 4`
JID_JOB3=`sbatch --dependency=afterok:$JID_JOB2 ./slurm/$1.slurm | cut -d " " -f 4`
JID_JOB4=`sbatch --dependency=afterok:$JID_JOB3 ./slurm/$1.slurm | cut -d " " -f 4`
JID_JOB5=`sbatch --dependency=afterok:$JID_JOB4 ./slurm/$1.slurm | cut -d " " -f 4`
JID_JOB6=`sbatch --dependency=afterok:$JID_JOB5 ./slurm/$1.slurm | cut -d " " -f 4`
JID_JOB7=`sbatch --dependency=afterok:$JID_JOB6 ./slurm/$1.slurm | cut -d " " -f 4`
JID_JOB8=`sbatch --dependency=afterok:$JID_JOB7 ./slurm/$1.slurm | cut -d " " -f 4`
JID_JOB9=`sbatch --dependency=afterok:$JID_JOB8 ./slurm/$1.slurm | cut -d " " -f 4`
sbatch --dependency=afterok:$JID_JOB9 ./slurm/$1.slurm