#export CUDA_VISIBLE_DEVICES='0,2'

set -x 

read -r -d '' training_commands <<EOF
train_dpo.py \
     --save_path ./llama3-8B-DPO \
     --save_steps 275 \
     --logging_steps 1 \
     --eval_steps 275 \
     --train_batch_size 256 \
     --micro_train_batch_size 2 \
     --pretrain ./llama3-8B-it \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset Anthropic/hh-rlhf \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --save_hf_model \
     --use_wandb 85b307e786518dda002091933c29896ac1180232
EOF
     # --wandb [WANDB_TOKENS]
     # --ipo [for IPO]
     # --label_smoothing 0.1 [for cDPO]
     # --use_wandb 85b307e786518dda002091933c29896ac1180232


if [[ ${1} != "slurm" ]]; then
    #export PATH=$HOME/.local/bin/:$PATH
    #deepspeed --include localhost:0,1 $training_commands
    deepspeed $training_commands
fi
