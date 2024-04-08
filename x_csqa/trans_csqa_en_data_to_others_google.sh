#!/bin/bash  
arr=('ar' 'he')
for num in {100..0..1}
do
    for i in ${arr[*]} 
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
            >> ./${lang}_trans/trans_en_to_${lang}.log
    done
done
