set -x

read -r -d '' training_commands <<EOF
train_sft.py \
    --max_len 2048 \
    --dataset Dahoas/full-hh-rlhf \
    --dataset_probs 1.0 \
    --train_batch_size 256 \
    --micro_train_batch_size 32 \
    --max_samples 500000 \
    --pretrain ./Tinyllama \
    --save_path ./tinyllama_sft \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 5e-6 \
    --gradient_checkpointing \
    --save_hf_model
EOF
    # --wandb [WANDB_TOKENS]
    # --use_wandb 85b307e786518dda002091933c29896ac1180232

if [[ ${1} != "slurm" ]]; then
    #export PATH=$HOME/.local/bin/:$PATH
    deepspeed $training_commands
fi
