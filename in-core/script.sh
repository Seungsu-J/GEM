#! /bin/bash
GPU_NUM=0
dataset=gowalla
query_size=5
query_folder=query_$query_size

dataset_list=("gowalla" "patents" "orkut" "amazon" "livejournal" "youtube" "soc" "datagen-9_0-fb")
query_folder_list=("query_4" "query_6" "query_8" "query_10" "query_12")


output_file=output_all_${dataset}_$query_size.txt
echo $dataset > $output_file
for i in $(seq 0 99)
do
  echo $i >> $output_file
  ./build/release/GEM -q ~/codes/datasets/$dataset/$query_folder/Q_$i.in -d ~/codes/datasets/$dataset/data.graph --threshold 0 --gpu $GPU_NUM >> $output_file
  echo "" >> $output_file
done
#all-in-one

