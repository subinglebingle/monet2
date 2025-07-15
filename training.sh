#!/bin/bash
cd /home/subin/unet/monet/

echo "" >>training_log.log

date +%Y-%m-%d_%T >>training_log.log

echo "**start: train_monet**">>training_log.log
python3 train_monet.py >>training_log.log
echo "**end**">>training_log.log

echo "**start: train.py**">>training_log.log
python3 train.py >>training_log.log
echo "**end**" >>training_log.log
