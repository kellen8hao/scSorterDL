#!/bin/bash
module load r/4.1.0
source ~/envs/enslda/bin/activate

rm -Rf *.pt ckpt_dataset1_5_8_final_fineaggre_nostrat.bz2
python /home/kbai/Belaid/web_ensembleLDA/scSorterDL_final/train.py --cell_sampling='uniform' --gene_sampling='uniform' --fine_aggr --global_reg --retrain --restart=0 --ldareg=0.000001 --optimizer=SGD --epochs=100 --numldas=300 --random_seed=8 --batch_size=200 --device=cuda --lr=0.05 --log_every=10 --cellsr=0.8 --train_data=/project/rrg-ubcxzh/kbai/singlecell_data/cross_platform/dataset1/wilcox_res/train.pkl --test_data=/project/rrg-ubcxzh/kbai/singlecell_data/cross_platform/dataset1/wilcox_res/test.pkl --runID=dataset1_5_8_final_fineaggre_nostrat --resX=10 --resY=10 2>&1 |egrep -v "Warning|warn"
#python /home/kbai/scratch_local/Belaid/ensembleLDA/nnLDA.py --retrain --restart=0 --ldareg=0.000001 --optimizer=SGD --epochs=100 --numldas=300 --random_seed=8 --batch_size=200 --device=cuda --lr=0.05 --log_every=10 --cellsr=0.8 --train_data=/project/rrg-ubcxzh/kbai/singlecell_data/cross_platform/dataset1/wilcox_res/train.pkl --test_data=/project/rrg-ubcxzh/kbai/singlecell_data/cross_platform/dataset1/wilcox_res/test.pkl --runID=dataset1_5_8_final_fineaggre_nostrat --resX=10 --resY=10
