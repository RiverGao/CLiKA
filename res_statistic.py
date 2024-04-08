import json
import os
import csv
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
model_res = defaultdict(dict)
# ar de en fr he it ja pl ru zh
def eval_res(data_path):
    with open(data_path,"r+",encoding="utf-8") as f:
        lines = f.readlines()
        total_num = len(lines)
        gold_cnt = 0
        for idx,line in enumerate(lines):
            line = json.loads(line)
            label = line['label']
            res = line['res'].strip("\n").strip()
            if len(res) > 0:
                res = res[0]
            else:
                print(f"res_path : {res_path} idx : {idx} ; line:{line} Empty!")
                res = "F"
            if res not in ['A','B','C','D','E']:
                print(f"res_path : {res_path} idx : {idx} ; line:{line} output Error!")
                have_error_model.add(model)
            if res == label:
                gold_cnt +=1
    return gold_cnt/total_num

langs=[ 'PL_trans', 'RU_trans', 'ZH_trans' ,'ar_trans', 'he_trans', 'en', 'DE_trans', 'JA_trans', 'FR_trans',  'IT_trans']# x_csqa 
copa_langs = ['DE','JA','FR','PL','RU','ar','he','it','en','zh']#x_copa
name_geo_langs = [ 'it', 'pl' ,'ru', 'en', 'de', 'ja' ,'fr', 'zh', 'ar' ,'he']# x_name,x_geo

models=['chatgpt','baichuan2_13b_base','baichuan2_7b_base',]



task_zu =[['x_csqa'],['x_name','x_geo'],['x_copa']]
lans_zu = [langs,name_geo_langs,copa_langs]
for x,y in zip (task_zu,lans_zu):
    tasks=x
    use_langs = y
    have_error_model = set()

    for task in tasks:
        statistic_data = []
        for model in models:
            single_data = {"model": model, "ar":0, "de":0, "en":0, "fr":0, "he":0, "it":0, "ja":0,"pl":0, "ru":0, "zh":0}
            for lang in use_langs:
                if model == "chatgpt":
                    res_path = f"../{task}/{lang}/{model}_res.jsonl"
                else:
                    res_path = f"../{task}/{lang}/{model}_res_bf16_add_simple_example.jsonl"
                ratio = eval_res(res_path)
                print(f"Task : {task} Lang:{lang} Model:{model} F1: {ratio}")
                # print(f"lang : {lang.lower().replace('_trans','')}")
                single_data[lang.lower().replace("_trans",'')] = ratio *100
            statistic_data.append(single_data)
        csv_file_path = f'./{task}.csv'
        csv_T_file_path = f'./{task}_T.csv'
        print(f"statistic_data: {statistic_data}")
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["model", "ar", "de", "en","fr", "he", "it", "ja","pl", "ru", "zh"]
            
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            csv_writer.writeheader()
            
            for row in statistic_data:
                csv_writer.writerow(row)
    for task in tasks:
        csv_file_path = f'./{task}.csv'
        csv_T_file_path = f'./{task}_T.csv'
        
        df = pd.read_csv(csv_file_path)

        df_transposed = df.transpose()


        df_transposed.to_csv(csv_T_file_path)
                
    for task in tasks:
        for lang in use_langs:
            res_path = f"../{task}/{lang}/chatgpt_res.jsonl"
            ratio = eval_res(res_path)
            print(f"Task : {task} Lang:{lang} Chatgpt F1: {ratio}")
