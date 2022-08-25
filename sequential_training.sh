#!/bin/bash

JID_JOB1=`sbatch $1.slurm | cut -d " " -f 4`
JID_JOB2=`sbatch --dependency=afterok:$JID_JOB1 $1.slurm | cut -d " " -f 4`
JID_JOB3=`sbatch --dependency=afterok:$JID_JOB2 $1.slurm | cut -d " " -f 4`
JID_JOB4=`sbatch --dependency=afterok:$JID_JOB3 $1.slurm | cut -d " " -f 4`
JID_JOB5=`sbatch --dependency=afterok:$JID_JOB4 $1.slurm | cut -d " " -f 4`
JID_JOB6=`sbatch --dependency=afterok:$JID_JOB5 $1.slurm | cut -d " " -f 4`
sbatch --dependency=afterok:$JID_JOB6 $1.slurm