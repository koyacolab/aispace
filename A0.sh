#!/bin/bash

python3 A0.py train_GPT -exp_name 'A0[embeds]' -TRAINER_RUN 1 -report_to 'wandb' -tokenize_it False -resume_from_checkpoint False 
