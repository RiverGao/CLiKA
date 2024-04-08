#!/bin/bash  
arr=('en' 'de' 'ja' 'fr' )
# arr=('en' )
for num in {2000..0..1}
do
    for i in ${arr[*]} 
    do  
        echo $i  
        lang=$i
        python eval_x_name_performance.py \
            --data_path "./${lang}/${lang}_name_data_generate.jsonl" \
            --res_path "./${lang}/chatgpt_res.jsonl" \
            --proxy "" \
            >> ./${lang}/eval_x_name_performance_fixed_en_prompt_${lang}_gpt3.log
    done
done 