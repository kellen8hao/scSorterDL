#!/bin/bash
reg="_smreg" # or "" for the bigreg 
folder=final_fineaggre_nostrat${reg}
ldareg=0.000001
epochs=3000
gene_sampling='uniform'
cell_sampling='uniform'
web_layer_sizes="" #"weblayers 13 13 13"
pltResDir="/scratch/kbai/results/web_ensembleLDA/cross_platform"
pltDataDir="/project/rrg-ubcxzh/kbai/singlecell_data/cross_platform"

for dataset in ${pltDataDir}/dataset{1..33} #{21..33};
do
  qrydtset=$(basename $dataset)
  mkdir -p ${pltResDir}/$qrydtset/$folder/query_logs
  echo ".........Checking job status for dataset $qrydtset................";
  for i in 5 10 50 100 300 
  do
     echo sbatch --job-name="${qrydtset}_$i${reg}" plt_query $i
     output=${pltResDir}/$qrydtset/$folder/query_logs/slurm_${i}-%A_%a.out
     # check how many jobs in the queue
     num_jobs=`sq --account=rrg-ubcxzh_gpu -r | wc -l`
     if [ ${num_jobs} -gt 995 ] # each array job counts as 5 jobs in this case...
     then
       echo "Full queue - ${num_jobs} are already in the queue ..."
       exit 0
     fi
     # check whether results are already there
     num_res_files=0
     if [ -d "${pltResDir}/$qrydtset/$folder/ldas_${i}" ]
     then 
     	num_res_files=`ls -l ${pltResDir}/$qrydtset/$folder/ldas_${i}/QueryReportInit_* 2>/dev/null|wc -l`
     fi
     if [ ${num_res_files} -lt 10 ]
     then
       # check if the job is pending/running:
       if  [ -z "`sq --format="%.30j"|grep " ${qrydtset}_${i}${reg}\$"`" ]
       then
           echo "Submitting unfinished job for ${qrydtset}$reg and number of ldas $i"
	   #for seed in {0..9}
	   #do
	   #   grep ',100,' ${pltResDir}/$qrydtset/$folder/ldas_${i}/ValResults_${qrydtset}_${i}_${seed}_final_fineaggre_nostrat_smreg.csv
	   sbatch --job-name="${qrydtset}_$i${reg}" --time=8:00:00 --output=$output plt_query.sh $i SGD "${pltResDir}/$qrydtset/$folder" "$dataset/wilcox_res" $ldareg $epochs ${gene_sampling} ${cell_sampling} ${web_layer_sizes}
	   
       else
           echo "Jobs for ${qrydtset} and number of ldas $i are already queued...."
       fi
     else
        echo "Dataset $qrydtset finished ..."
     fi
    
  done
  for i in 600 1000
  do
     echo sbatch --job-name="${qrydtset}_$i${reg}" plt_query $i
     output=${pltResDir}/$qrydtset/$folder/query_logs/slurm_${i}-%A_%a.out
     num_jobs=`sq --account=rrg-ubcxzh_gpu -r | wc -l`
     if [ ${num_jobs} -gt 995 ]
     then
       echo 'Full queue - ${num_jobs} are already in the queue ...'
       exit 0
     fi
     # check whether results are already there
     num_res_files=0
     if [ -d "${pltResDir}/$qrydtset/$folder/ldas_${i}" ]
     then
         num_res_files=`ls -l ${pltResDir}/$qrydtset/$folder/ldas_${i}/QueryReportInit_* 2>/dev/null | wc -l`
     fi
     if [ ${num_res_files} -lt 10 ]
     then
       # check if the job is pending/running:
       if  [ -z "`sq --format="%.30j"|grep " ${qrydtset}_$i${reg}\$"`" ]
       then
	  echo "Submitting unfinished job for ${qrydtset}${reg} and number of ldas $i"
          sbatch --job-name="${qrydtset}_$i${reg}" --mem=127000 --array=0-8:2 --time=12:00:00 --output=$output plt_query.sh $i SGD "${pltResDir}/$qrydtset/$folder" "$dataset/wilcox_res" $ldareg $epochs ${gene_sampling} ${cell_sampling} ${web_layer_sizes}
       else
           echo "Jobs for ${qrydtset}$reg and number of ldas $i are already queued...."
       fi
     else
        echo "Dataset ${qrydtset}$reg finished ..."
     fi
  done
  #break
done

