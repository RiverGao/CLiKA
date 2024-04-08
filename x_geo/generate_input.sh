#!/bin/bash  
arr=( 'it' 'pl' 'ru' 'en' 'de' 'ja' 'fr' 'zh' 'ar' 'he')
# arr=('en' )
for i in ${arr[*]} 
do  
    echo $i  
    lang=$i
    python eval_x_geo_generate_input.py \
        --data_path "./${lang}/${lang}_geo_data.jsonl" \
        --res_path "./${lang}/generate_input.jsonl" \
        --proxy "" \
        --api_key_idx 5 \
        > ./${lang}/generate_input_${lang}_gpt3.log
done
