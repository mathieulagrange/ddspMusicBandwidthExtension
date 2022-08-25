import os
import sys
import subprocess

exp_name = sys.argv[1]
if len(sys.argv)>2:
    exp_small_name = sys.argv[2]
else:
    exp_small_name = exp_name

# if the bash file for the sequential training script doesn't exist, we write it
if not os.path.isfile(f'./bash/{exp_name}.sh'):
    print('to do')

# if the slurm file doesn't exist, we write it
if not os.path.isfile(f'./slurm/{exp_name}.sh'):
    lines = []
    lines.append('#!/bin/bash\n\n')

    lines.append(f'#SBATCH --job-name={exp_small_name}\n')
    lines.append('##SBATCH --partition=gpu_p2\n')
    lines.append('#SBATCH --nodes=1\n')
    lines.append('#SBATCH --ntasks-per-node=1\n')
    lines.append('#SBATCH --gres=gpu:1\n')
    lines.append('#SBATCH --cpus-per-task=10\n')
    lines.append('##SBATCH --cpus-per-task=3\n\n')

    lines.append('#SBATCH --hint=nomultithread\n')
    lines.append('#SBATCH --time=20:00:00\n')
    lines.append('#SBATCH --output=outputs/{exp_name}.out\n')
    lines.append('#SBATCH --error=outputs/{exp_name}.out\n\n')
    
    lines.append('# nettoyage des modules charges en interactif et herites par defaut\n')
    lines.append('module purge\n\n')
    
    lines.append('# chargement des modules\n')
    lines.append('module load tensorflow-gpu/py3/2.8.0\n')
    lines.append('export PYTHONUSERBASE=$WORK/.idris_ddsp\n')
    lines.append('export PATH=$PYTHONUSERBASE/bin:$PATH\n\n')

    lines.append('# echo des commandes lancees\n')
    lines.append('set -x\n\n')
    
    lines.append('# execution du code\n')
    lines.append('python -u main.py -c -s alg=ddsp+data=sol\n')

    with open(f'slurm/{exp_name}.slurm', 'w') as f:
        f.writelines(lines)

# call sequential_training.sh
subprocess.run(['./sequential_training.sh', exp_name], stderr=subprocess.PIPE)