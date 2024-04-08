#!/bin/bash  
#'DE' 'JA' 'FR' 'PL' 'RU' 'ar' 'he' 'it' 'en' 'zh'
arr=('DE' 'JA' 'FR' )
for num in {1000..0..1}
do
    for i in ${arr[*]} 
    do  
        echo $i  
        lang=$i
        python eval_xcopa_performance_fixed_en_prompt.py \
            --data_path "./${lang}/test.jsonl" \
            --res_path "./${lang}/chatgpt_res.jsonl" \
            --proxy "" \
            --api_key_idx 0 \
            >> ./${lang}/eval_x_copa_performance_${lang}_gpt3.log
    done
done 