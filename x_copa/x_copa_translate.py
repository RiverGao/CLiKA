import deepl
import os
import time
from tqdm import tqdm
import argparse
import sys
import json
import requests



deepl_key_pool = [

]
source_data_path = ""

def get_start_idx(data_path):
    if not os.path.exists(data_path):
        return 0
    return len(open(data_path,"r+",encoding="utf-8").readlines())

def translate_deepl(texts,tar_lang='EN-US',key = ""):
    translator = deepl.Translator(key)
    result = translator.translate_text(texts, target_lang=tar_lang) 
    return [item.text for item in result]


def translate_google(texts,tar_lang = "en"):
    res = []
    print(f"en text : {texts}",flush = True)
    for item in texts:
        if item.isdigit():
            res.append(item)
        else:
            data = {"keywords": item, "targetLanguage": tar_lang,"sourceLanguage":"en"}
            r = requests.post('',json=data)
            print(r.json(),flush = True)
            res.append(r.json()['data']['text'])
            time.sleep(3)
    return res

def translate_data(source_data_path,tgt_path,tgt_lang,start_idx,key):
    with open(source_data_path,"r+",encoding= "utf-8") as rf, open(tgt_path,"a+",encoding="utf-8") as wf :
        r_lines =  rf.readlines()
        with  tqdm(total=len(r_lines) - start_idx) as pbar:
            for idx,r_line in enumerate(r_lines):
                if idx < start_idx:
                    continue
                r_line = json.loads(r_line)
                w_line = r_line.copy() 
                
                input_text = [r_line['premise'],r_line['choice1'],r_line['choice2']]
                
                assert len(input_text) == 3
                
                if tgt_lang in ['DE', 'JA', 'FR', 'IT', 'PL', 'RU', 'ZH','it','zh']:
                    output_text = translate_deepl(input_text,tgt_lang,key)
                elif tgt_lang in ['ar','he']:
                    output_text = translate_google(input_text,tgt_lang)
                    
                w_line['premise'],w_line['choice1'],w_line['choice2'] = output_text[0],output_text[1],output_text[2]
                
                
                print(f"idx : {idx} ; en : {input_text} ;{tgt_lang} :{w_line} \n")
                wf.write(json.dumps(w_line)+ "\n")
                pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_language',type = str)
    parser.add_argument('--tgt_path',type = str)
    parser.add_argument('--api_key_idx',type =int,default=1)
    parser.add_argument('--proxy',type=str,default="")
    args = parser.parse_args()
    os.environ["http_proxy"] = args.proxy
    os.environ["https_proxy"] = args.proxy
    tgt_start_idx  = get_start_idx(args.tgt_path)
    print(f"args: {args}",flush = True)
    api_key_val = deepl_key_pool[args.api_key_idx%len(deepl_key_pool)]
    translate_data(source_data_path,args.tgt_path,args.tgt_language,tgt_start_idx,api_key_val)
    
    