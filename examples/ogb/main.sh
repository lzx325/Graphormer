{
    set -e
    source "/sw/csi/anaconda3/4.4.0/binary/anaconda3/etc/profile.d/conda.sh"
    source shell_scripts_configs/PCQM4M-LSC.sh
    module purge
    module load cuda/11.2.2
    conda deactivate
    conda activate graphormer
    mkdir -p $default_root_dir
    python ../../graphormer/entry.py \
    --num_workers 8 \
    --seed $seed \
    --batch_size $batch_size \
    --dataset_name "$dataset_name" \
    --gpus $n_gpu \
    --accelerator ddp \
    --precision 16 $arch \
    --default_root_dir $default_root_dir \
    --tot_updates $tot_updates \
    --warmup_updates $warmup_updates \
    --max_epochs $max_epochs \
    --peak_lr $peak_lr \
    --end_lr $end_lr \
    --progress_bar_refresh_rate 10 \
    --flag \
    --flag_m $flag_m \
    --flag_step_size $flag_step_size \
    --flag_mag $flag_mag \
    --checkpoint_path $ckpt_path
     
    exit 0;
}

