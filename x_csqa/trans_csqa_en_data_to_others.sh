#!/bin/bash  
arr=('DE' 'JA' 'FR' 'IT' 'PL' 'RU' 'ZH')
idx_list=(1 2 3 4)
for i in ${arr[*]} 
do  
    for j in ${idx_list[*]} 
    do
        echo $i  
        lang=$i
        if [ ! -d "./${lang}_trans/" ];then
            mkdir ./${lang}_trans/
        else
            echo "./${lang}_trans/ Exists!"
        fi
        python csqa_translate.py \
            --tgt_language  $lang \
            --tgt_path "./${lang}_trans/validation.jsonl" \
            --proxy "" \
            --api_key_idx $j \
            >> ./${lang}_trans/trans_en_to_${lang}.log
    done
done
