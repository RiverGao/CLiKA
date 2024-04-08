import openai
import os
import time
from tqdm import tqdm
import argparse
import sys
import json


openai.api_base = ""
label_2_option = ['A','B']
api_key_pool = [

]


def clean_output(return_output):
    if "\n" in return_output:
        return_output = return_output.strip("\n")[0]
    return return_output

def get_responese(input):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": input}],
    temperature = 0
    )
    return completion.choices[0].message.content

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def tokenizer_static(inputs):
    print(f"total len:{len(encoding.encode(inputs))}",flush=True)
# ('DE' 'JA' 'FR' 'IT' 'PL' 'RU' 'ZH')
def inputs_generating(data_path,start_idx = 0):
    uuids = []
    
    '''
    Answer the following multiple choice questions by selecting the correct option according to the Premise.Please choose the correct one from the two options A and B.
    Premise:
    Question:
    A. 
    B.
    Answer: 
    '''
    key_2_question = {'cause': 'What was the cause of this?',
                  'effect': 'What happened as a result?'}
    current_initial_prompt = "Answer the following multiple choice questions by selecting the correct option according to the Premise.Please choose the correct one from the two options A and B.\n"
    inputs , labels = [] ,[]
    with open(data_path,"r+",encoding="utf-8") as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if idx < start_idx:
                continue
            item = json.loads(line)
            inputs.append(current_initial_prompt + f"Premise:{item['premise']}\nQuestion:{key_2_question[item['question']]}\nA. {item['choice1']}\nB. {item['choice2']}\nAnswer:")
            labels.append(label_2_option[item["label"]])
            uuids.append(item['id'])
            print(f"idx : {idx} ,Input: {inputs[-1]} ; Label: {labels[-1]}")
    tokenizer_static("".join(inputs))
    return inputs,labels,uuids
            
def generate_ouptut_data(inputs,output_file,gold_labels,start_idx,uuids):
    outputs= []
    with tqdm(total=len(inputs)) as pbar:
        pbar.set_description('Generate Processing')
        with open(output_file,"a+",encoding="utf-8") as f:
            for idx,input in enumerate(inputs):
                res = get_responese(input)
                outputs.append(res)
                temp = {
                            "id":idx+start_idx,
                            "input": inputs[idx],
                            'res':res,
                            'label':gold_labels[idx],
                            'uuid':uuids[idx]
                        }
                print(f"idx: {idx} temp: {temp}",flush=True)
                f.write(json.dumps(temp)+"\n")
                if idx >= 30:
                    time.sleep(4)
                else:
                    time.sleep(3)
                if idx % 80 == 60 :
                    time.sleep(15)
                    
                pbar.update(1)
    return outputs

def get_start_idx(data_path):
    if not os.path.exists(data_path):
        return 0
    return len(open(data_path,"r+",encoding="utf-8").readlines())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type = str)
    parser.add_argument('--res_path',type=str)
    parser.add_argument('--start_idx',type=int,default=0)
    parser.add_argument('--api_key_idx',type =int,default=2)
    parser.add_argument('--proxy',type=str,default="")
    args = parser.parse_args()
    openai.api_key = api_key_pool[args.api_key_idx%len(api_key_pool)]
    os.environ["http_proxy"] = args.proxy
    os.environ["https_proxy"] = args.proxy
    print(f"openai.api_key : {openai.api_key }",flush=True)
    args.start_idx = get_start_idx(args.res_path)
    print(f"args :{args}",flush=True)
    inputs,gold_labels,uuids = inputs_generating(args.data_path,args.start_idx)
    outputs = generate_ouptut_data(inputs,args.res_path,gold_labels,args.start_idx,uuids)
