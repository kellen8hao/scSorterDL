#!/bin/bash
reg="_smreg" # or "" for the bigreg
folder=scSorterDL${reg}
ldareg=0.000001
epochs=3000
gene_sampling='uniform'
cell_sampling='uniform'
web_layer_sizes="" #"weblayers 13 13 13"
pltResDir="/scratch/kbai/results/web_ensembleLDA/cross_platform"
pltDataDir="/project/rrg-ubcxzh/kbai/singlecell_data/cross_platform"

for dataset in ${pltDataDir}/dataset1 #{21..33};
do
  qrydtset=$(basename $dataset)
  mkdir -p ${pltResDir}/$qrydtset/$folder/query_logs
  echo "..............Submitting job for dataset $dataset................";
  for i in 5 10 50 100 300
  do
     echo sbatch --job-name="${qrydtset}_$i" plt_query $i
     output=${pltResDir}/$qrydtset/$folder/query_logs/slurm_${i}-%A_%a.out
     sbatch --job-name="${qrydtset}_$i$reg" --time=1:00:00 --mem=32000 --cpus-per-task=4 --output=$output plt_query.sh $i SGD "${pltResDir}/$qrydtset/$folder" "$dataset/wilcox_res" $ldareg $epochs ${gene_sampling} ${cell_sampling} ${web_layer_sizes} 
     exit 0
  done
  for i in 600 1000
  do
     echo sbatch --job-name="${qrydtset}_$i" plt_query $i
     output=${pltResDir}/$qrydtset/$folder/query_logs/slurm_${i}-%A_%a.ou
     sbatch --job-name="${qrydtset}_$i$reg" --mem=32000 --cpus-per-task=4 --array=0-8:2 --time=1:00:00 --output=$output plt_query.sh $i SGD ${pltResDir}/$qrydtset/$folder "$dataset/wilcox_res" $ldareg $epochs ${gene_sampling} ${cell_sampling} ${web_layer_sizes}
  done
  #break
done

