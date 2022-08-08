python main.py \
--data_dir ../data/train \
--data_list "data_train.list"
--save ${1} \
--epochs 100 \
--batch_size 20 \
--lr 0.001 \
--decay 10,20,30,40,50 \
--gamma 1.0,0.5,0.25,0.125,0.0625 \
--batch_size 20 \
--max_depth 10.0 \
--cut_mask \
--rgb_noise 0.05 \
--noise 0.01 \
--num_threads 5