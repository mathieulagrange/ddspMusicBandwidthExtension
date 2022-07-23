#!/bin/bash

#SBATCH --job-name=test         # nom du job
##SBATCH --partition=gpu_p2          # de-commente pour la partition gpu_p2
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU (1/4 des GPU)
#SBATCH --cpus-per-task=10           # nombre de coeurs CPU par tache (1/4 du noeud 4-GPU)
##SBATCH --cpus-per-task=3           # nombre de coeurs CPU par tache (pour gpu_p2 : 1/8 du noeud 8-GPU)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=outputs/ddsp%j.out      # nom du fichier de sortie
#SBATCH --error=outputs/ddsp%j.out       # nom du fichier d'erreur (ici commun avec la sortie)
 
# nettoyage des modules charges en interactif et herites par defaut
module purge
 
# chargement des modules
module load tensorflow-gpu/py3/2.8.0  
export PYTHONUSERBASE=$WORK/.idris_ddsp
export PATH=$PYTHONUSERBASE/bin:$PATH

# echo des commandes lancees
set -x
 
# execution du code
python -u main.py -c -s alg=ddsp+data=sol