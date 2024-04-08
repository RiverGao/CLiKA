models=('baichuan2_13b_base'   'baichuan2_7b_base'  )


langs=( 'PL_trans' 'RU_trans' 'ZH_trans' 'ar_trans' 'he_trans' 'en' 'DE_trans' 'JA_trans' 'FR_trans'  'IT_trans')
task="x_csqa"
for model in ${models[*]} 
do
    for lang in ${langs[*]} 
    do  
        echo $task
        echo $lang
        echo $model
        CUDA_VISIBLE_DEVICES=3 python inference.py \
            --input_path "../${task}/${lang}/chatgpt_res.jsonl" \
            --output_path "../${task}/${lang}/${model}_res_bf16_add_simple_example.jsonl" \
            --model_name ${model}  \
            --add_simple_example \
            --task ${task} \
            >> ../${task}/${lang}/${model}_inference_bf16_add_simple_example.log
    done
done


langs=( 'it' 'pl' 'ru' 'en' 'de' 'ja' 'fr' 'zh' 'ar' 'he')
task="x_geo"
for model in ${models[*]} 
do
    for lang in ${langs[*]} 
    do  
        echo $task
        echo $lang
        echo $model
        CUDA_VISIBLE_DEVICES=3 python inference.py \
            --input_path "../${task}/${lang}/generate_input.jsonl" \
            --output_path "../${task}/${lang}/${model}_res_bf16_add_simple_example.jsonl" \
            --model_name ${model}  \
            --add_simple_example \
            --task ${task} \
            >> ../${task}/${lang}/${model}_inference_bf16_add_simple_example.log
    done
done

langs=( 'it' 'pl' 'ru' 'en' 'de' 'ja' 'fr' 'zh' 'ar' 'he')
task="x_name"
for model in ${models[*]} 
do
    for lang in ${langs[*]} 
    do  
        echo $task
        echo $lang
        echo $model
        CUDA_VISIBLE_DEVICES=3 python inference.py \
            --input_path "../${task}/${lang}/generate_input.jsonl" \
            --output_path "../${task}/${lang}/${model}_res_bf16_add_simple_example.jsonl" \
            --model_name ${model}  \
            --add_simple_example \
            --task ${task} \
            >> ../${task}/${lang}/${model}_inference_bf16_add_simple_example.log
    done
done



langs=( 'DE' 'JA' 'FR' 'PL' 'RU' 'ar' 'he' 'it' 'en' 'zh')
task="x_copa"
for model in ${models[*]} 
do
    for lang in ${langs[*]} 
    do 
        echo $task
        echo $lang
        echo $model
        CUDA_VISIBLE_DEVICES=3 python inference.py \
            --input_path "../${task}/${lang}/generate_input.jsonl" \
            --output_path "../${task}/${lang}/${model}_res_bf16_add_simple_example.jsonl" \
            --model_name ${model}  \
            --add_simple_example \
            --task ${task} \
            >> ../${task}/${lang}/${model}_inference_bf16_add_simple_example.log
    done
done

