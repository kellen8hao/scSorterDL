#!/bin/bash
reg="_smreg"
folder=final_fineaggre_nostrat${reg}
ldareg=0.000001
epochs=3000
gene_sampling='uniform'
cell_sampling='uniform'
web_layer_sizes="" #"weblayers 13 13 13"
valResDir="/scratch/kbai/results/web_ensembleLDA/cross_validation"
valDataDir="/project/def-ubcxzh/kbai/singlecell_data/cross_validation"

for dataset in ${valDataDir}/dataset11/wilcox_res; # we need to do it
do
  qrydtset=$(basename $(dirname $dataset))
  for valfold in $dataset/screening/*
  do
       foldNum=$(basename $valfold)
       resFoldDir=${valResDir}/${qrydtset}/${foldNum}/$folder
       mkdir -p ${resFoldDir}/query_logs
       echo "............. Checking $qrydtset on $valfold................";
       for i in 5 10 50 100 300
       do
          echo sbatch --job-name="val_${qrydtset}_$i${reg}" val_query $i
          output=${resFoldDir}/query_logs/slurm_${i}-%A_%a.out
	  # check how many jobs in the queue
     	  num_jobs=`sq --account=def-ubcxzh_gpu -r | wc -l`
     	  if [ ${num_jobs} -gt 2995 ] # each array job counts as 5 jobs in this case...
     	  then
       	  echo "Full queue - ${num_jobs} are already in the queue ..."
       	  exit 0
     	  fi
           # check whether results are already there
	  for seed in {0..9}
	  do
     	   num_res_files=0
     	   if [ -d "${resFoldDir}/ldas_${i}" ]
           then 
	      #echo "Results are located here: ${resFoldDir}/ldas_${i}/QueryReportInit_${qrydtset}_${foldNum}_${i}_${seed}_*.csv"
     	      num_res_files=`ls -l ${resFoldDir}/ldas_${i}/QueryReportInit_${qrydtset}_${foldNum}_${i}_${seed}_*.csv 2>/dev/null|wc -l`
     	   fi
     	   if [ ${num_res_files} -lt 1 ]
    	   then
             # check if the job is pending/running:
             if  [ -z "`sq -r --format="%.30i,%.50j"|grep " val_${qrydtset}_${foldNum}_${i}${reg}\$"|egrep "\d*_${seed}, "`" ]
             then
               echo "Submitting unfinished job for ${qrydtset}${reg} on $foldNum and number of ldas $i and random seed $seed"
	       sbatch --job-name="val_${qrydtset}_${foldNum}_$i${reg}" --time=12:00:00 --array=$seed --output=$output val_query.sh  $i SGD "${resFoldDir}" "${valfold}" $ldareg $epochs ${gene_sampling} ${cell_sampling} ${web_layer_sizes} 
             else
               echo "Jobs for ${qrydtset} on $foldNum and number of ldas $i are already queued...."
             fi
       
           else
             echo "Dataset $qrydtset finished for $foldNum, number of ldas $i, random seed $seed ..."
           fi
       done
   done
   for i in 600 1000
   do
	  echo sbatch --job-name="val_${qrydtset}_$i${reg}" val_query $i
          output=${resFoldDir}/query_logs/slurm_${i}-%A_%a.out
	  # check how many jobs in the queue
     	  num_jobs=`sq --account=def-ubcxzh_gpu -r | wc -l`
     	  if [ ${num_jobs} -gt 2995 ] # each array job counts as 5 jobs in this case...
     	  then
       	  echo "Full queue - ${num_jobs} are already in the queue ..."
       	  exit 0
     	  fi
           # check whether results are already there
	  for seed in {0..9}
	  do
     	   num_res_files=0
     	   if [ -d "${resFoldDir}/ldas_${i}" ]
           then 
	      #echo "Results are located here: ${resFoldDir}/ldas_${i}/QueryReportInit_${qrydtset}_${foldNum}_${i}_${seed}_*.csv"
     	      num_res_files=`ls -l ${resFoldDir}/ldas_${i}/QueryReportInit_${qrydtset}_${foldNum}_${i}_${seed}_*.csv 2>/dev/null|wc -l`
     	   fi
     	   if [ ${num_res_files} -lt 1 ]
    	   then
             # check if the job is pending/running:
             if  [ -z "`sq -r --format="%.30i,%.50j"|grep " val_${qrydtset}_${foldNum}_${i}${reg}\$"|egrep "\d*_${seed}, "`"  ]
             then
               echo "Submitting unfinished job for ${qrydtset}${reg} on $foldNum and number of ldas $i and random seed $seed"
	       sbatch --job-name="val_${qrydtset}_${foldNum}_$i${reg}" --time=12:00:00 --array=$seed --output=$output val_query.sh $i SGD "${resFoldDir}" "${valfold}" $ldareg $epochs ${gene_sampling} ${cell_sampling} ${web_layer_sizes}
             else
               echo "Jobs for ${qrydtset} on $foldNum and number of ldas $i are already queued...."
             fi
       
           else
             echo "Dataset $qrydtset finished for $foldNum, number of ldas $i, random seed $seed ..."
           fi
       done
   done
done
done
