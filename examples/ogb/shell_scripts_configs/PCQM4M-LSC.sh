exp_name="PCQM4M-LSC--pretrain"
seed="1"
dataset_name="PCQM4M-LSC"
# TODO: what is multi_hop and the meaning of the last two parameters
arch="\
--ffn_dim 512 \
--hidden_dim 512 \
--intput_dropout_rate 0.0 \
--attention_dropout_rate 0.1 \
--dropout_rate 0.1 \
--weight_decay 0.0 \
--n_layers 6 \
--edge_type multi_hop \
--multi_hop_max_dist 5 \
"
batch_size=1024
epoch=400
peak_lr="3e-4"
end_lr="1e-9"
# TODO
flag_m=2
flag_step_size="0.2"
flag_mag="0"
n_gpu=8
tot_updates=$((33000*epoch/batch_size/n_gpu))
warmup_updates=$((tot_updates/10))
max_epochs=$((epoch+1))

default_root_dir="../../exps/PCQM4M-LSC/$exp_name/$seed"
pretrain_checkpoint_dir="<pretrain_checkpoint_dir>"
checkpoint_dir="$default_root_dir/lightning_logs/checkpoints"