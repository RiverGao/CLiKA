#!/bin/bash  
#'DE' 'JA' 'FR' 'PL' 'RU' 'ar' 'he' 'it' 'en' 'zh'
arr=('DE' 'JA' 'FR' 'PL' 'RU' 'ar' 'he' 'it' 'en' 'zh')

for i in ${arr[*]} 
do  
    echo $i  
    lang=$i
    python generate_input_data.py \
        --data_path "./${lang}/test.jsonl" \
        --res_path "./${lang}/generate_input.jsonl" \
        --proxy "" \
        --api_key_idx 0 
done
