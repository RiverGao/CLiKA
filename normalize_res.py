import pandas as pd


# input_file = 'logiqa2.csv'
# output_file = 'logiqa2_normalize.csv'
'''
x_csqa:74.6 5 
x_copa:96.4 2
logiqa2:53.6 4
x_reclor:61.6 4
x_name:92.2 4
x_geo:97.0 4
'''
def normalize_data(task_name,input_file,output_file):
    
    nor_val  ={
        "x_csqa":[74.6,20],
        "x_copa":[96.4,50],
        "logiqa2":[53.6,25],
        "x_reclor":[61.6,25],
        "x_name":[92.2,25],
        "x_geo":[97.0,25],
    }
    output_data  = []
    langs = ['zh','ja','de','en','fr','it','pl','ru','ar','he']
    df = pd.read_csv(input_file)
    for index, row in df.iterrows():
        for lang in langs:
            row[lang] = (row[lang] - nor_val[task_name][1] ) / (nor_val[task_name][0] - nor_val[task_name][1])
            if row[lang] < 0:
                row[lang] = 0
        print(f"idx: {index}",row)
        output_data.append(row)

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file, index=False)



normalize_data("x_csqa","x_csqa.csv","x_csqa_normalize.csv")
normalize_data("x_copa","x_copa.csv","x_copa_normalize.csv")
normalize_data("x_name","x_name.csv","x_name_normalize.csv")
normalize_data("x_geo","x_geo.csv","x_geo_normalize.csv")