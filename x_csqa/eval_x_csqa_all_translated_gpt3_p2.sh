#!/bin/bash  
arr=('DE' 'JA' 'FR' 'IT')
for num in {800..0..1}
do
    for i in ${arr[*]} 
    do  
        echo $i  
        lang=$i
        python eval_x_csqa_performance.py \
            --data_language  $lang \
            --data_path "./x_csqa/${lang}_trans/validation.jsonl" \
            --res_path "./x_csqa/${lang}_trans/chatgpt_res.jsonl" \
            --proxy "" \
            --api_key_idx 4 \
            >> ./x_csqa/${lang}_trans/eval_x_csqa_performance_${lang}_gpt3.log
    done
done 