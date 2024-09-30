#export CUDA_VISIBLE_DEVICES='0,2,3'

set -x 

read -r -d '' training_commands <<EOF
train_selective.py \
     --save_path ./llama2-chat-7B-simpo-baseline \
     --save_steps -1 \
     --logging_steps 1 \
     --eval_steps -1 \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --pretrain /mnt/iusers01/nactem01/g36374ky/scratch/FastChat/llama2-chat-7B \
     --bf16 \
     --max_epochs 1 \
     --max_len 4096 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --dataset ./ultrafeedback_100p_new_selected_data/k0.9.csv \
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
     # pretrain /mnt/iusers01/nactem01/g36374ky/scratch/FastChat/llama2-chat-13B
     # --pretrain /mnt/iusers01/nactem01/g36374ky/scratch/FastChat/llama2-chat-7B

if [[ ${1} != "slurm" ]]; then
    #export PATH=$HOME/.local/bin/:$PATH
    deepspeed --include localhost:0 $training_commands
    #deepspeed $training_commands
fi
