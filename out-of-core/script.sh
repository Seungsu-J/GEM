#! /bin/bash
GPU_NUM=0
dataset=datagen-9_0-fb
query_size=4
query_folder=query_$query_size

dataset_list=("dblp" "enron_16" "github" "gowalla" "patents" "orkut" "amazon" "livejournal" "youtube" "soc" "datagen-9_0-fb")
query_folder_list=("query_4" "query_6" "query_8" "query_10" "query_12")

for label in 4 8 12
do
  dset=${dataset}_${label}
  for qs in 3 4 5 6
  do
    file=output_all_${dset}_${qs}.txt
    echo $dset > $file
      # echo $i >> $file
      ./build/release/GEM -q ~/codes/datasets/$dset/query_${qs} -d ~/codes/datasets/$dset/data.graph --threshold 0 --gpu $GPU_NUM >> $file
      # echo "" >> $file
  done
done

