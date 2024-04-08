import argparse
import sys
from abc import ABC, abstractmethod
import json
import os
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel,LlamaForCausalLM, LlamaTokenizer
from transformers.generation.utils import GenerationConfig

total_num = {
    'x_csqa':1000,
    'logiqa2':1572,
    'x_copa':500,
    'x_reclor':500,
    'x_name':180,
    'x_geo':200
}
task_instruction ={
    'x_csqa':'The following are multiple choice questions. Please choose the most reasonable one from the following options.\n',
    'logiqa2':'Answer the following multiple choice questions by selecting the correct option according to the Content.Please choose the most correct one from the four options A, B, C, and D.\n',
    'x_copa':'Answer the following multiple choice questions by selecting the correct option according to the Premise.Please choose the correct one from the two options A and B.\n',
    'x_reclor':'Answer the following multiple choice questions by selecting the correct option according to the Content.Please choose the most correct one from the four options A, B, C, and D.\n',
    'x_name':'The following are multiple choice questions. Please choose the most reasonable one from the following options.\n',
    'x_geo':'The following are multiple choice questions. Please choose the most reasonable one from the following options.\n'
}

single_prompt = {
    'x_csqa':"Question:2+3=?\nA. 1\nB. 2\nC. 3\nD. 5\nE. 6\nAnswer:D. 5\nQuestion:",
    'logiqa2':"Content:Joe has 2 red apples and 3 green apples.\nQuestion:How many apples dose Joe have?\nA. 2\nB. 3\nC. 4\nD. 5\nAnswer:D. 5\n",
    'x_copa':"Premise:The sky turns dark.\nQuestion:What was the cause of this?\nA. It's night.\nB. It's noon.\nAnswer:A. It's night.\n\n",
    'x_reclor':"Content:Joe has 2 red apples and 3 green apples.\nQuestion:How many apples dose Joe have?\nA. 2\nB. 3\nC. 4\nD. 5\nAnswer:D. 5\n",
    'x_name':'Question:2+3=?\nA. 1\nB. 2\nC. 3\nD. 5\nAnswer:D. 5\n',
    'x_geo':'Question:2+3=?\nA. 1\nB. 2\nC. 3\nD. 5\nAnswer:D. 5\n'
    }

name_2_path = {
    'baichuan2_7b_chat':"Baichuan2-7B-Chat",
}
device = "cuda"
class LLM_model(ABC):
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        pass
    
    @abstractmethod
    def model_load(self,model_path):
        pass
    
    @abstractmethod
    def model_inference(self,input_texts):
        pass


class Baichuan_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
        
    def model_load(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.model = model

        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2, self.model.generation_config.num_beams) 

    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                messages = [{"role": "user", "content":input}]
                self.model.generation_config.max_new_tokens = 2
                response = self.model.chat(self.tokenizer,messages)
                res.append(response.strip())
                print(f"Input:{input} ; Result:{res[-1]}",flush=True)
                pbar.update(1)
        return res
    
    
class Baichuan_base_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
        
    def model_load(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

        self.model = model
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2, self.model.generation_config.num_beams) 
        print(f"self.model.generation_config: {self.model.generation_config}")

    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                model_input = self.tokenizer.encode(input, return_tensors="pt").to("cuda")
                outputs = self.model.generate(model_input, max_new_tokens=2,
                                              do_sample=False)
                res.append(self.tokenizer.decode(outputs[0]).replace(input,'').strip())
                print(f"Input:{input} ; Result:{res[-1]}",flush=True)
                pbar.update(1)
        return res
    
class Bloomz_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
    def model_load(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2, self.model.generation_config.num_beams) 
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                model_input = self.tokenizer.encode(input, return_tensors="pt").to("cuda")
                outputs = self.model.generate(model_input, max_new_tokens=2,
                                              do_sample=False)
                res.append(self.tokenizer.decode(outputs[0]).replace(input,'').strip())
                pbar.update(1)
        return res
    
class Chatglm_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
    
    def model_load(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,trust_remote_code=True).half().cuda()
        self.model = self.model.eval()
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2, self.model.generation_config.num_beams) 
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                _inputs = self.tokenizer(input,return_tensors="pt")
                input_length =  len(_inputs['input_ids'][0])
                response, history = self.model.chat(self.tokenizer, input, history=[],max_length=input_length + 20)
                res.append(response.strip())
                pbar.update(1)
        return res

class Polylm_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
    
    def model_load(self, model_path):

        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2,self.model.generation_config.num_beams) 
        
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                inputs = self.tokenizer.encode(input, return_tensors="pt").to("cuda")
                outputs = self.model.generate(inputs, max_new_tokens=4,pad_token_id =self.tokenizer.eos_token_id)
                output = self.tokenizer.decode(outputs[0]).replace(input,'').strip().strip("\n")
                res.append(output)
                pbar.update(1)
        
        return res

class Bayling_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
    
    def model_load(self, model_path):

        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype="auto", device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2,self.model.generation_config.num_beams) 
                
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                model_input = self.tokenizer.encode(input, return_tensors="pt").to("cuda")
                output_ids = self.model.generate(
                    model_input,
                    do_sample=False,
                    max_new_tokens = 4
                )
                _inputs = self.tokenizer(input,return_tensors="pt")
                input_length =  len(_inputs['input_ids'][0])
                if self.model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][input_length:]
                outputs = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                )
                
                res.append(outputs)
                pbar.update(1)
        
        return res
    

class Bigtranslate_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
    
    def model_load(self, model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype="auto", device_map="auto")

        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2,self.model.generation_config.num_beams) 
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                inputs = self.tokenizer.encode(input, return_tensors="pt").to("cuda")
                _inputs = self.tokenizer(input,return_tensors="pt")
                input_length =  len(_inputs['input_ids'][0])
                outputs = self.model.generate(inputs, max_new_tokens=4)
                output_ids = outputs[0][input_length:]
                outputs = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                )
                res.append(outputs)
                pbar.update(1)
        
        return res

class Mistral_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
        
    def model_load(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2, self.model.generation_config.num_beams)
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                messages = [
                    {"role": "user", "content": input}
                ]
                encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
                model_inputs = encodeds.to(device)
                self.model.to(device)
                generated_ids = self.model.generate(model_inputs, max_new_tokens=4, do_sample=False)
                decoded = self.tokenizer.batch_decode(generated_ids)
                decoded_res = decoded[0].strip("<s>").strip("</s>").replace(input,"").replace("[/INST]","").replace("[INST]","").strip()
                res.append(decoded_res)
                pbar.update(1)
        return res


class Llama_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
        
    def model_load(self, model_path):
        self.tokenizer =  LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto',
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2, self.model.generation_config.num_beams) 
        
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                origin_len = len(input)
                inputs = self.tokenizer(input,return_tensors="pt") 
                generation_output = self.model.generate(
                    input_ids = inputs["input_ids"].cuda(),
                    attention_mask = inputs['attention_mask'].cuda(),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens = 2,
                    do_sample=False,
                )
                s = generation_output[0]
                output = self.tokenizer.decode(s,skip_special_tokens=True)
                res.append(output[origin_len:].strip())
                print(f"Input:{input} ;Result:{res[-1]}",flush=True)
                pbar.update(1)
        return res

class leo_base_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
        
    def model_load(self, model_path):
        self.tokenizer =  AutoTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto',
        )
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.force_word_ids = [self.tokenizer(['A','B','C','D','E'], add_special_tokens=False)["input_ids"]]
        self.model.generation_config.num_beams=max(2, self.model.generation_config.num_beams) 
        
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                origin_len = len(input)
                inputs = self.tokenizer(input,return_tensors="pt") 
                generation_output = self.model.generate(
                    input_ids = inputs["input_ids"].cuda(),
                    attention_mask = inputs['attention_mask'].cuda(),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens = 2,
                    do_sample=False,
                )
                s = generation_output[0]
                output = self.tokenizer.decode(s,skip_special_tokens=True)
                res.append(output[origin_len:].strip())
                pbar.update(1)
        return res
                

        
class Palm_model(LLM_model):
    def __init__(self) -> None:
        super().__init__()
        
    def model_load(self, model_path):
        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
        # print(f"models: {[x.name for x in models]}")
        self.model = models[0].name
    def model_inference(self, input_texts):
        res = []
        with tqdm(total = len(input_texts)) as pbar:
            for input in input_texts:
                completion = palm.generate_text(
                    model=self.model,
                    prompt=input,
                    temperature=0,
                    # The maximum length of the response
                    max_output_tokens=4,
                )
                res.append(completion.result.strip())
                print(f"Input:{input} ;Result:{res[-1]}",flush=True)
                pbar.update(1)
        return res
        
Processers = {
    'baichuan2_7b_chat':Baichuan_model
}

def add_single_prompt(text,task):
    others = '\n'.join(text.split('\n')[1:])
    return task_instruction[task]+single_prompt[task]+others

def generate_inputs(input_path,task,add_simple_example = False,start_idx = 0):
    inputs = []    
    cp_data = []
    with open(input_path,"r+",encoding="utf-8") as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            if idx < start_idx:
                continue
            line = json.loads(line)
            cp_data.append(line.copy())
            origin_input = line['input']
            inputs.append(add_single_prompt(origin_input,task))
            cp_data[-1]['input'] = inputs[-1]
    return inputs,cp_data

def get_start_idx(data_path):
    if not os.path.exists(data_path):
        return 0
    return len(open(data_path,"r+",encoding="utf-8").readlines())

def write_output(outputs,cp_data,output_path):
    with open(output_path,"a+",encoding="utf-8") as wf:
        for output,origin_data in zip(outputs,cp_data):
            origin_data['res']  = output
            wf.write(json.dumps(origin_data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',type = str)
    parser.add_argument('--task',type = str)
    parser.add_argument('--output_path',type = str)
    parser.add_argument('--model_name',type = str)
    parser.add_argument("--add_simple_example", action='store_true')#add 2+3 =5 
    args = parser.parse_args()
    start_idx = get_start_idx(args.output_path)
    print(f"args : {args} ;start_idx: {start_idx}",flush=True)
    if start_idx == total_num[args.task]:
        print(f"Finished!")
        sys.exit()
    inference_model = Processers[args.model_name]()
    inference_model.model_load(name_2_path[args.model_name])
    inputs,cp_data = generate_inputs(args.input_path,args.task,args.add_simple_example,start_idx)
    outputs = inference_model.model_inference(inputs)
    write_output(outputs,cp_data,args.output_path)