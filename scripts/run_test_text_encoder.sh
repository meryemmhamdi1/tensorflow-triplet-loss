# ! /bin/bash
qsub -N test_text_encode -P aida -cwd -pe mt 2 -l h_vmem=12G,gpu=1,h_rt=24:00:00 test_text_encoder.sh