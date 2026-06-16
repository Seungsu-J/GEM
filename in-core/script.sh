#! /bin/bash
GPU_NUM=2
dataset=amazon
query_size=3
query_folder=query_$query_size

dataset_list=("amazon" "github" "dblp" "patents" "gowalla" "livejournal" "youtube")
query_folder_list=("query_4" "query_6" "query_8" "query_10" "query_12")




for ds in ${dataset_list[@]}
do
  for qs in 3 4 5 6 
  do  
    qf=query_$qs
    output_file=recompute/output_all_${ds}_${qs}_new.txt
    echo "" > $output_file
    ./build/release/GEM -q ~/codes/datasets/$ds/$qf/ -d ~/codes/datasets/$ds/data.graph --threshold 0 --gpu $GPU_NUM >> $output_file
  done
done
