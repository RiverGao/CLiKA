#!/bin/bash  
arr=('DE' 'JA' 'FR' 'PL' 'RU' )
# arr=('ZH')
idx_list=(0 1 2 3 4)
for i in ${arr[*]} 
do  
    for j in ${idx_list[*]}
    do
        echo $i  
        lang=$i
        if [ ! -d "./${lang}/" ];then
            mkdir ./${lang}/
        else
            echo "./${lang}/ Exists!"
        fi
        python x_copa_translate.py \
            --tgt_language  $lang \
            --tgt_path "./${lang}/test.jsonl" \
            --proxy "" \
            --api_key_idx $j \
            >> ./${lang}/trans_en_to_${lang}.log
    done
done
