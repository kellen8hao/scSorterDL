#!/bin/bash
module load r/4.1.0
source ~/envs/enslda/bin/activate
numldas=$1
random_seed=$2
optimizer=${3:-'NAdam'}
drspec=${4:-'/scratch/kbai/results/web_ensembleLDA/cross_platform/dataset1/nonof'} #results directory
datadir=${5:-"/project/rrg-ubcxzh/kbai/singlecell_data/cross_platform/dataset1/wilcox_res"} #data folder, where train and test are located
ldareg=${6:-0.1}
epochs=${7:-0}
gene_sampling=${8:-'uniform'} 
cell_sampling=${9:-'uniform'} #'-s'
weblayers=${10:-'noweblayers'} #'-s'
if [ "$weblayers" = "noweblayers" ]
then
   weblayers=''
else
   weblayers='--web_layer_size "${@:11}"'
fi

drname="${drspec}"
resX=10
resY=10
spec=$(basename $drspec)
dtname="$(basename $(dirname $datadir))"

global_reg='--global_reg'
fine_aggr='--fine_aggr'
#noshrinkage='--noshrinkage'
group_split='' #'--group_split'


batch_size=200
incrEpochs=300
if [ "${dtname}" == "dataset29" ] || [ "${dtname}" == "dataset32" ]
then
  batch_size=700
fi
longtrain=false
#for random_seed in {1..10}
#do
runID="${dtname}_${numldas}_${random_seed}_${spec}"
echo "run ID: $runID"
mkdir -p "${drname}/ldas_${numldas}"
cd "${drname}/ldas_${numldas}"
if [ -f "ckpt_${runID}.bz2" ]
then
  rstrt=1
  lr=0.01
else
  rstrt=0
  lr=0.05
fi
if [ $epochs -eq 0 ]
then
  rstrt=1
# cp "${drname}/ldas_${numldas}/ckpt_${runID}"* ./
elif [ $longtrain = true ]
then
  oldEpochs=0
  if [ $rstrt -eq 1 ] && [ -f "ValResults_${runID}.csv" ]
  then
     oldEpochs=$((`grep ",$incrEpochs," "ValResults_${runID}.csv"|wc -l`*$incrEpochs))
  fi
  epochs=$((epochs - oldEpochs)) 
  while [ $epochs -gt 0 ]
  do
      echo python /project/def-ubcxzh/kbai/Belaid/web_ensembleLDA/localwebnnLDA.py ${fine_aggr} ${global_reg} --cell_sampling="${cell_sampling}" --gene_sampling="${gene_sampling}" --retrain --restart=$rstrt --ldareg=$ldareg --optimizer=$optimizer --epochs=$incrEpochs --numldas=$numldas --random_seed=${random_seed} --batch_size=${batch_size} --device='cuda' --lr=$lr --log_every=10 --train_data="$datadir/train.pkl" --runID="$runID" --resX=$resX --resY=$resY 2>&1 | tee -a log_${runID}_${SLURM_JOBID}.out
      python /home/kbai/Belaid/web_ensembleLDA/scSorterDL_final/train.py ${fine_aggr} ${global_reg} ${weblayers} --cell_sampling="${cell_sampling}" --gene_sampling="${gene_sampling}" --retrain --restart=$rstrt --ldareg=$ldareg --optimizer=$optimizer --epochs=$incrEpochs --numldas=$numldas --random_seed=${random_seed} --batch_size=${batch_size} --device='cuda' --lr=$lr --log_every=10 --train_data="$datadir/train.pkl" --runID="$runID" --resX=$resX --resY=$resY 2>&1 | tee -a log_${runID}_${SLURM_JOBID}.out
      rstrt=1
      epochs=$((epochs-incrEpochs))
      # lr=0.005
      # incrEpochs=4000
 done
 epochs=0
fi

echo python /home/kbai/Belaid/web_ensembleLDA/scSorterDL_final/train.py ${fine_aggr} ${global_reg} ${weblayers} --cell_sampling="${cell_sampling}" --gene_sampling="${gene_sampling}" --retrain --restart=$rstrt --ldareg=$ldareg --optimizer=$optimizer --epochs=$epochs --numldas=$numldas --random_seed=${random_seed} --batch_size=${batch_size} --device='cuda' --lr=$lr --log_every=10 --train_data="$datadir/train.pkl" --test_data="$datadir/test.pkl" --runID="$runID" --resX=$resX --resY=$resY 2>&1 | tee -a log_${runID}_${SLURM_JOBID}.out
python /home/kbai/Belaid/web_ensembleLDA/scSorterDL_final/train.py ${fine_aggr} ${global_reg} ${web} ${group_split} $stratified --retrain --restart=$rstrt --ldareg=$ldareg --optimizer=$optimizer --epochs=$epochs --numldas=$numldas --random_seed=${random_seed} --batch_size=${batch_size} --device='cuda' --lr=$lr --log_every=10 --train_data="$datadir/train.pkl" --test_data="$datadir/test.pkl" --runID="$runID" --resX=$resX --resY=$resY 2>&1 | tee  -a log_${runID}_${SLURM_JOBID}.out

