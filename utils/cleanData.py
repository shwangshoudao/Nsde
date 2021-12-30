import re
import pandas as pd
import os
import datetime 

root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_path = root_path + "/data/option"

for filename in os.listdir(data_path):
    if filename == ".DS_Store" or "revised" in filename:
        continue
    print(filename)
    file_path = data_path+"/"+filename
    date_start_index = filename.find("2")
    file_date = datetime.datetime.strptime(filename[date_start_index:date_start_index+10],'%Y-%m-%d')
    file_date_before = file_date - datetime.timedelta(days=1)
    
    origin_data = pd.read_excel(file_path)
    origin_data = origin_data[origin_data["lastTradeDate"].between(str(file_date_before),str(file_date))]
    revised_data = origin_data.loc[:,["strike","Maturity","lastPrice","volume"]]
    revised_data["Maturity"] = revised_data["Maturity"] + 1
    revised_data.index = list(range(len(revised_data)))
    revised_data.to_excel(data_path+"/"+filename[:-5]+"_revised"+".xlsx")
