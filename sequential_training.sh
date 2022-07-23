#!/bin/bash
JID_JOB1=`sbatch launch_job.slurm | cut -d " " -f 4`
JID_JOB2=`sbatch --dependency=afterok:$JID_JOB1 launch_job.slurm | cut -d " " -f 4`
JID_JOB3=`sbatch --dependency=afterok:$JID_JOB2 launch_job.slurm | cut -d " " -f 4`
JID_JOB4=`sbatch --dependency=afterok:$JID_JOB3 launch_job.slurm | cut -d " " -f 4`
JID_JOB5=`sbatch --dependency=afterok:$JID_JOB4 launch_job.slurm | cut -d " " -f 4`
JID_JOB6=`sbatch --dependency=afterok:$JID_JOB5 launch_job.slurm | cut -d " " -f 4`
JID_JOB7=`sbatch --dependency=afterok:$JID_JOB6 launch_job.slurm | cut -d " " -f 4`
JID_JOB8=`sbatch --dependency=afterok:$JID_JOB7 launch_job.slurm | cut -d " " -f 4`
JID_JOB9=`sbatch --dependency=afterok:$JID_JOB8 launch_job.slurm | cut -d " " -f 4`
sbatch --dependency=afterok:$JID_JOB9 launch_job.slurm