#!/bin/bash  
arr=('ar' 'he')
for num in {1000..0..1}
do
    for i in ${arr[*]} 
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
            >> ./${lang}/trans_en_to_${lang}.log
    done
done