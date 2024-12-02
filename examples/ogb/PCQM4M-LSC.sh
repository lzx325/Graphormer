{
    set -e
    source "/sw/csi/anaconda3/4.4.0/binary/anaconda3/etc/profile.d/conda.sh"
    module purge
    module load cuda/11.2.2
    conda deactivate
    conda activate graphormer

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
    n_gpu=4
    tot_updates=$((3045360*epoch/batch_size/n_gpu))
    warmup_updates=$((tot_updates/10))
    max_epochs=$((epoch+1))

    default_root_dir="../../exps/PCQM4M-LSC/$exp_name/$seed"
    pretrain_checkpoint_dir="<pretrain_checkpoint_dir>"
    checkpoint_dir="$default_root_dir/lightning_logs/checkpoints"

    mkdir -p $default_root_dir

    # train
    # python ../../graphormer/entry.py \
    # --num_workers 8 \
    # --seed $seed \
    # --batch_size $batch_size \
    # --dataset_name "$dataset_name" \
    # --gpus $n_gpu \
    # --accelerator ddp \
    # --precision 16 \
    # $arch \
    # --default_root_dir $default_root_dir \
    # --tot_updates $tot_updates \
    # --warmup_updates $warmup_updates \
    # --max_epochs $max_epochs \
    # --peak_lr $peak_lr \
    # --end_lr $end_lr \
    # --progress_bar_refresh_rate 10 \
    # --checkpoint_path "$pretrain_checkpoint_dir"

    # test
    eval_checkpoint_path="../../exps/PCQM4M-LSC/PCQM4M-LSC--pretrain/1/lightning_logs/checkpoints/PCQM4M-LSC-epoch=324-valid_mae=0.1300.ckpt"

    python ../../graphormer/entry.py \
    --num_workers 8 \
    --seed "$seed" \
    --batch_size $batch_size \
    --dataset_name "$dataset_name" \
    --gpus 1 \
    --accelerator ddp \
    --precision 16 \
    $arch \
    --default_root_dir tmp/ \
    --checkpoint_path "$eval_checkpoint_path" \
    --test \
    --progress_bar_refresh_rate 100

    exit 0;
}
