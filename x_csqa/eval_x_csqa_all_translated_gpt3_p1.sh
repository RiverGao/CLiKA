#!/bin/bash  
arr=('PL' 'RU' 'ZH')
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
            >> ./x_csqa/${lang}_trans/eval_x_csqa_performance_${lang}_gpt3.log
    done
done 