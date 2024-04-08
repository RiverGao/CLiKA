import openai
import os
import time
from tqdm import tqdm
import argparse
import sys
import json


openai.api_base = ""

api_key_pool = [

]
def clean_output(return_output):
    if "\n" in return_output:
        print(f"return_output : {return_output} HAS NOISE!")
        return_output = return_output.strip("\n")[0]
    return return_output

def get_responese(input):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": input}],
    temperature = 0
    )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def tokenizer_static(inputs):
    print(f"total len:{len(encoding.encode(inputs))}")
# ('DE' 'JA' 'FR' 'IT' 'PL' 'RU' 'ZH')
def inputs_generating(data_path,data_language = "en",start_idx = 0):
    uuids = []
    data_language = data_language.lower()
    prompt_dict = {'zh':"以下是单项选择题，请从下列选项中选出最合理的一项。\n",
                   'en':"The following is a multiple choice question. Please choose the most reasonable one from the following options.\n",
                   'ar':'فيما يلي أسئلة متعددة الخيارات ، يرجى اختيار السؤال الأكثر منطقية من الخيارات التالية.\n',
                   'de':'Bei der folgenden Frage handelt es sich um eine Multiple-Choice-Frage. Bitte wählen Sie aus den folgenden Optionen die sinnvollste aus.\n',
                   'es':'La siguiente es una pregunta de opción múltiple. Elija la más razonable de las siguientes opciones.\n',
                   'fr':'La question suivante est une question à choix multiples. Veuillez choisir la plus raisonnable parmi les options suivantes.\n',
                   'hi':'निम्नलिखित एक बहुविकल्पीय प्रश्न है। कृपया निम्नलिखित विकल्पों में से सबसे उचित प्रश्न चुनें।\n',
                   'it':'La seguente è una domanda a scelta multipla. Scegli quella più ragionevole tra le seguenti opzioni.\n',
                   'jap':'以下は多肢選択式の質問です。次の選択肢から最も妥当なものを選択してください。\n',
                   'ja':'以下は多肢選択式の質問です。次の選択肢から最も妥当なものを選択してください。\n',
                   'nl':'De volgende vraag is een meerkeuzevraag. Kies de meest redelijke vraag uit de volgende opties.\n',
                   'pl':'Poniżej znajduje się pytanie wielokrotnego wyboru. Wybierz najbardziej rozsądne z poniższych opcji.\n',
                   'pt':'A questão a seguir é de múltipla escolha. Escolha a mais razoável dentre as opções a seguir.\n',
                   'ru':'Ниже приведен вопрос с несколькими вариантами ответов. Пожалуйста, выберите наиболее разумный из следующих вариантов.\n',
                   'sw':'Hii ni swali la kuchagua mojawapo, tafadhali chagua chaguo sahihi kutoka kwenye chaguo zilizotolewa.\n',
                   'ur':'درج ذیل ایک کثیر انتخابی سوال ہے۔ براہ کرم درج ذیل اختیارات میں سے سب سے زیادہ معقول سوال کا انتخاب کریں۔\n',
                   'vi':'Sau đây là câu hỏi trắc nghiệm, vui lòng chọn câu trả lời hợp lý nhất trong các phương án dưới đây.\n',
                   'he':'להלן שאלה רב ברירה. אנא בחר את הסבירה ביותר מבין האפשרויות הבאות.\n'}
    if data_language not in prompt_dict.keys():
        print(f"Language Error!")
        sys.exit()
    current_initial_prompt = prompt_dict[data_language]
    print(f"current_initial_prompt : {current_initial_prompt}")
    inputs , labels = [] ,[]
    with open(data_path,"r+",encoding="utf-8") as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if idx < start_idx:
                continue
            item = json.loads(line)
            # print(f"item['question']['stem']: {type(item['question']['stem'])}item['question']['choices']['text'][0]  : {item['question']['choices']['text'][0]}")
            inputs.append(current_initial_prompt + f"{item['question']['stem']}\nA. {item['question']['choices']['text'][0]}\nB. {item['question']['choices']['text'][1]}\nC. {item['question']['choices']['text'][2]}\nD. {item['question']['choices']['text'][3]}\nE. {item['question']['choices']['text'][4]}\nAnswer:")
            labels.append(item["answerKey"])
            uuids.append(item['id'])
            # print(f"idx : {idx} ,Input: {inputs[-1]} ; Label: {labels[-1]}")
    tokenizer_static("".join(inputs))
    print(f"len(inputs) : {len(inputs)}")
    return inputs,labels,uuids
            
def generate_ouptut_data(inputs,output_file,gold_labels,start_idx,uuids):
    outputs= []
    # decode_ner_result = [] 
    with tqdm(total=len(inputs)) as pbar:
        pbar.set_description('Generate Processing:')
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
                print(f"idx: {idx} temp: {temp}")
                # decode_ner_result.append(decode_res(res))
                f.write(json.dumps(temp)+"\n")
                if idx >= 30:
                    time.sleep(5)
                else:
                    time.sleep(4)
                if idx % 80 == 40 :
                    time.sleep(25)
                    
                pbar.update(1)
    return outputs

def get_start_idx(data_path):
    if not os.path.exists(data_path):
        return 0
    return len(open(data_path,"r+",encoding="utf-8").readlines())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_language',type = str)
    parser.add_argument('--data_path',type = str)
    parser.add_argument('--res_path',type=str)
    parser.add_argument('--start_idx',type=int,default=0)
    parser.add_argument('--api_key_idx',type =int,default=2)
    parser.add_argument('--proxy',type=str,default="")
    args = parser.parse_args()
    openai.api_key = api_key_pool[args.api_key_idx%len(api_key_pool)]
    os.environ["http_proxy"] = args.proxy
    os.environ["https_proxy"] = args.proxy
    print(f"openai.api_key : {openai.api_key }")
    args.start_idx = get_start_idx(args.res_path)
    print(f"args :{args}")
    inputs,gold_labels,uuids = inputs_generating(args.data_path,args.data_language,args.start_idx)
    outputs = generate_ouptut_data(inputs,args.res_path,gold_labels,args.start_idx,uuids)