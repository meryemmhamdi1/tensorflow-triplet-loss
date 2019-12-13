# ! /bin/bash
qsub -N train_vae -P aida -cwd -pe mt 2 -l h_vmem=12G,gpu=1,h_rt=24:00:00 train_hard.sh