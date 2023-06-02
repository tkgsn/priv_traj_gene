#!/usr/bin/bash

save_name=art_data_clipping
cuda_number=1
data_name=simple_variable
accountant_mode=rdp
epsilon=0
# if the directory /data/results/test/meta_learning_variable/data_name exists, then remove it
if [ -d "/data/results/test/$save_name" ]; then
    echo "remove /data/results/test/$save_name"
    rm -rf /data/results/test/$save_name
fi
echo "save_name: $save_name"

epoch=10
n_epoch=$(($epoch+1))

python3 pseuddata_gene_variable.py --is_variable --size 10000

for i in {0..0}
do
python3 run_meta_learning.py --gru --data_name $data_name --save_name $save_name --is_traj_type --is_time --is_evaluation --without_end --cuda_number $cuda_number --n_generated 0 --print_epoch $epoch --n_epochs $n_epoch --clipping_bound 1 --dp_noise_multiplier 0 --batch_size 100 --loss_reduction mean --is_dp --meta_n_iter 50 --seed $i --accountant_mode $accountant_mode --epsilon $epsilon
python3 workspace.py --save_name $save_name
done