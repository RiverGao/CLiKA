#!/bin/bash

REQUIRED_GPUS=4  # 你希望使用的GPU数量
datasets=("ZH" "PL" "HE" "DE" "AR" "IT" "JA" "RU" "FR") 
modelSet=("/home/nfs01/llama/model/llama-hf/llama-7b-hf")

for model in "${modelSet[@]}"; do
    model_name=${model##*/}
    declare -A modelTOTemplate=(
        ["Llama-2-13b-chat-hf"]="llama2"
        ["bloomz-7b1-mt"]="empty"
        ["llama-13b-hf"]="alpaca"
        ["chinese_llama-13b"]="alpaca"
        ["meta-llama_Llama-2-13b-hf"]="alpaca"
        ["baichuan-inc_Baichuan2-13B-Base"]="alpaca"
        ["hfl_chinese-llama-2-13b"]="alpaca"
        ["alpaca-13b"]="alpaca"
        ["bayling-13b-v1.1"]="BayLing"
        ["Baichuan2-13B-Chat"]="baichuan2"
        ["chinese_alpaca-13b"]="alpaca"
        ["chinese-alpaca-2-13b-hf"]="llama2_zh"
        ["vicuna-13b-v1.5"]="vicuna"
        ["LeoLM_leo-hessianai-13b"]="llama2"
        ["LeoLM_leo-hessianai-13b-chat"]="hessianai"
        ['llama-7b-hf']='alpaca'
    )

    echo "model_name: $model_name"
    template=${modelTOTemplate[$model_name]}
    echo "Using template: $template"
    #    query_key_value  q_proj,v_proj    W_pack
    declare -A lora_targetS=(["llama-7b-hf"]="q_proj,v_proj" ["llama-13b-hf"]="q_proj,v_proj" ["Llama-2-7b-hf"]="q_proj,v_proj" ["llama-2-13b-hf"]="q_proj,v_proj" ["Llama-2-13b-chat-hf"]="q_proj,v_proj" ["Llama-2-7b-chat-hf"]="q_proj,v_proj" ["bloomz-7b1-mt"]="query_key_value" ["bayling-13b-v1.1"]="q_proj,v_proj" ["chinese-llama-7b"]="q_proj,v_proj" ["chinese-llama-13b"]="q_proj,v_proj" ["hfl_chinese-llama-2-7b"]="q_proj,v_proj" ["hfl_chinese-llama-2-13b"]="q_proj,v_proj" ["chinese_llama2_7b"]="q_proj,v_proj" ["m-llama"]="q_proj,v_proj" ['bloom-7b1']="query_key_value" ['Baichuan2-13B-Chat']="W_pack" ['meta-llama_Llama-2-13b-hf']='q_proj,v_proj' ["baichuan-inc_Baichuan2-13B-Base"]="W_pack" ['alpaca-13b']="q_proj,v_proj" ['chinese_alpaca-13b']="q_proj,v_proj" ['chinese_llama-13b']='q_proj,v_proj' ['chinese-alpaca-2-13b-hf']='q_proj,v_proj' ['vicuna-13b-v1.5']='q_proj,v_proj' ['LeoLM_leo-hessianai-13b']='q_proj,v_proj' ['LeoLM_leo-hessianai-13b-chat']='q_proj,v_proj')
    lora_target=${lora_targetS[$model_name]}
    echo "lora_target: $lora_target"
    MASTER_PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
    FREE_GPUS=$(python -c "
import subprocess
import re
import sys

def find_free_gpus():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free', '--format=csv,nounits,noheader'], universal_newlines=True)
    lines = output.strip().split('\\n')
    free_gpus = []
    for i, line in enumerate(lines):
        util, mem = map(int, re.split(',\\s*', line))
        if mem > 25000:
            free_gpus.append(str(i))

    if len(free_gpus) < $REQUIRED_GPUS:
        sys.stderr.write(f'Error: Not enough free GPUs. Required: $REQUIRED_GPUS, Available: {len(free_gpus)}\\n')
        sys.exit(1)

    print(','.join(free_gpus[:$REQUIRED_GPUS]))

find_free_gpus()
")
    echo "Using GPUs: $FREE_GPUS"


    learning_rateS=(2e-5)
    for lr in "${learning_rateS[@]}"; do
        for dataset in "${datasets[@]}"; do
            output_dir="./rebutall/$model_name/$dataset$lr"
            deepspeed --include localhost:$FREE_GPUS --master_port=$MASTER_PORT Your PATH/src/train_bash.py \
                --deepspeed Your PATH/ds_config.json \
                --stage sft \
                --model_name_or_path $model \
                --do_train \
                --dataset  $dataset \
                --template $template \
                --finetuning_type  full \
                --output_dir  $output_dir \
                --overwrite_cache \
                --per_device_train_batch_size 8 \
                --gradient_accumulation_steps 2 \
                --lr_scheduler_type cosine \
                --logging_steps 20 \
                --learning_rate $lr \
                --num_train_epochs 3 \
                --plot_loss \

        done
    done
done


