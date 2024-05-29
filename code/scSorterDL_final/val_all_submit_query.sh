#!/bin/bash
folder=final_fineaggre_nostrat_smreg
ldareg=0.000001
epochs=3000
gene_sampling='uniform'
cell_sampling='uniform'
web_layer_sizes="" #"weblayers 13 13 13"
valResDir="/scratch/kbai/results/web_ensembleLDA/cross_validation"
valDataDir="/project/rrg-ubcxzh/kbai/singlecell_data/cross_validation"

for dataset in ${valDataDir}/dataset{4..5}/wilcox_res; # we need to do it
do
  qrydtset=$(basename $(dirname $dataset))
  for valfold in $dataset/screening/*
  do
       foldNum=$(basename $valfold)
       resFoldDir=${valResDir}/${qrydtset}/${foldNum}/$folder
       mkdir -p ${resFoldDir}/query_logs
       echo "..............Submitting job for dataset $dataset................";
       for i in 5 10 50 100 300
       do
          echo sbatch --job-name="val_${qrydtset}_$i" val_query $i
          output=${resFoldDir}/query_logs/slurm_${i}-%A_%a.out
          sbatch --job-name="val_${qrydtset}_$i" --time=6:00:00 --output=$output val_query.sh $i SGD "${resFoldDir}" "${valfold}" $ldareg $epochs ${gene_sampling} ${cell_sampling} ${web_layer_sizes}
       done
       for i in 600 1000
       do
          echo sbatch --job-name="val_${qrydtset}_$i" val_query $i
          output=${valResDir}/$qrydtset/$folder/query_logs/slurm_${i}-%A_%a.ou
          sbatch --job-name="val_${qrydtset}_$i" --mem=127000 --array=0-8:2 --time=6:00:00 --output=$output val_query.sh $i SGD ${resFoldDir}  "${valfold}" $ldareg $epochs ${gene_sampling} ${cell_sampling} ${web_layer_sizes} 
       done
       #break
  done
done

