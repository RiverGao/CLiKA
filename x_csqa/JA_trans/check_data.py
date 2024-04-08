import os
import json
with open('./baichuan2_7b_chat_res_bf16_add_simple_example.jsonl',"r+",encoding="utf-8") as f:
    lines = f.readlines()
    for idx,line in enumerate(lines):
        line = json.loads(line)
        if line['res'].strip()[0] not in ['A','B','C','D','E']:
            print(f"idx : {idx} line: {line}")